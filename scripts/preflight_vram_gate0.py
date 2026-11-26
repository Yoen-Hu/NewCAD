"""Two-sample Gate-0 benchmark for the locked Windows Ampere training stack."""

import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import bitsandbytes as bnb
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration,
    StoppingCriteria, StoppingCriteriaList,
)


ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION = ROOT.parent / "reproduction"
CHECKPOINT = REPRODUCTION / "checkpoint" / "Qwen3-VL-8B-Instruct"
MAX_LENGTH = 2048
NEW_TOKENS = 1280
GROUP_SIZE = 4
GRADIENT_ACCUMULATION = 16
MIN_PIXELS = 3136
MAX_PIXELS = 200704
GIB = 1024**3
GB = 10**9
LIMITS = {"sft": 10.2, "cf_rpo": 10.2, "grpo_online": 10.5}


def log(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=True), flush=True)


def cuda_memory():
    free, total = torch.cuda.mem_get_info()
    peak = torch.cuda.max_memory_allocated()
    return {"max_memory_allocated_bytes": peak, "max_memory_allocated_gib": round(peak / GIB, 4),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "allocated_bytes": torch.cuda.memory_allocated(), "device_free_bytes": free,
            "device_total_bytes": total}


def load_model():
    sys.path.insert(0, str(REPRODUCTION))
    from readonly_weights import ReadOnlyShard

    config = json.loads((CHECKPOINT / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", {})
    if (config.get("model_type") != "qwen3_vl" or config.get("quantization_config")
            or text_config.get("dtype", text_config.get("torch_dtype")) != "bfloat16"):
        raise RuntimeError("standard BF16 Qwen3-VL checkpoint required. FP8 is prohibited on sm_86.")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=False,
    )
    with patch("transformers.modeling_utils.safe_open", ReadOnlyShard):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(CHECKPOINT), local_files_only=True, device_map={"": 0}, torch_dtype=torch.bfloat16,
            quantization_config=quantization, attn_implementation="sdpa",
        )
    if not model.is_loaded_in_4bit or any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("NF4 model must be fully CUDA resident")
    # Freeze all base tensors and preserve BF16 for frozen embeddings, head and vision.
    # PEFT's generic k-bit helper upcasts those large frozen tensors to FP32.
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    targets = r".*language_model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=targets, task_type="CAUSAL_LM"))
    if any(parameter.requires_grad for parameter in model.get_base_model().visual.parameters()):
        raise RuntimeError("vision encoder must stay frozen")
    trainables = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainables or not model.is_gradient_checkpointing:
        raise RuntimeError("LoRA trainables and gradient checkpointing are required")
    optimizer = bnb.optim.PagedAdamW8bit(trainables, lr=1e-5)
    log("model_loaded", trainable_parameters=sum(p.numel() for p in trainables), **cuda_memory())
    return model, optimizer


def make_sample(index, processor):
    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    for x, y in [(35, 40), (240, 40), (35, 230)]:
        draw.rectangle((x, y, x + 155, y + 120), outline="black", width=3)
        for delta in (0, 25):
            for dx in range(9, 145, 12):
                draw.line((x + dx, y + 47 + delta, x + dx + 6, y + 47 + delta), fill="black", width=2)
    if index == 0:
        draw.ellipse((294, 76, 338, 120), outline="black", width=3)
    else:
        draw.rectangle((283, 78, 350, 122), outline="black", width=3)
    labels = ("front", "top", "side")
    for label, xy in zip(labels, [(35, 163), (240, 163), (35, 353)]):
        draw.text(xy, label, fill="black")
    category, method, shape = (("round_hole_plate", "circle", "circle(4)") if index == 0
                               else ("rectangular_culvert", "rect", "rect(12, 8)"))
    task = ("Read the three orthographic projections including the hidden dashed internal opening. "
            "Return evidence-grounded EG-PIR JSON followed by an ETC-CQ program. The workplane is XY "
            "and all dimensions use millimetres. Outer length 40, width 30, height 20. " * 9)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": task}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = processor(text=[chat], images=[image], return_tensors="pt")
    if prompt.input_ids.shape[1] > MAX_LENGTH - NEW_TOKENS:
        raise RuntimeError(f"prompt exceeds {MAX_LENGTH - NEW_TOKENS} tokens")
    pixel_count = int(prompt.image_grid_thw[0, 1:].prod()) * processor.image_processor.patch_size**2
    if not MIN_PIXELS <= pixel_count <= MAX_PIXELS:
        raise RuntimeError(f"image exceeds visual budget: {pixel_count}")
    pir = {"part_category": category,
           "outer_solid_params": {name: {"value": size, "unit": "mm", "evidence": name}
                                  for name, size in zip("WLH", (40, 30, 20))},
           "hollow_features": [{"hollow_id": "hole", "cadquery_mode": "extrude_cut",
                                "spatial_pose": {"base_plane": "XY", "plane_offset": -10, "center_2d": [0, 0],
                                                 "extrude_axis": "Z", "through_type": "through_all",
                                                 "overshoot_ends": ["start", "end"]},
                                "profile_shape": {"shape_type": method, "dimensions": [4] if index == 0 else [12, 8]},
                                "evidence": ["hidden-opening"]}],
           "constraints": [], "underconstrained_params": []}
    script = (
        "import cadquery as cq\n# 1. Outer Solid Parameters\nW = 40\nL = 30\nH = 20\n"
        "# 2. Hollow Feature Parameters\nCUT_MARGIN = 10.0\n# 3. Underconstrained Parameters\n"
        'solid = cq.Workplane("XY").rect(W, L).extrude(H, combine=False)\n'
        f'tool = cq.Workplane("XY").workplane(offset=-CUT_MARGIN).center(0, 0).{shape}'
        '.extrude(H + 2 * CUT_MARGIN, combine=False)\nsolid = solid.cut(tool)\nresult = solid\n')
    positive = f"<eg_pir>{json.dumps(pir)}</eg_pir>\n<cadquery>\n{script}</cadquery>"
    # Pad inside CadQuery comments to keep the XML response valid at full context.
    suffix = "\nresult = solid\n</cadquery>"
    pad = processor.tokenizer.encode("\n# Synthetic context stress for the Gate-0 benchmark.",
                                     add_special_tokens=False)
    end = processor.tokenizer.encode(suffix, add_special_tokens=False) + [processor.tokenizer.eos_token_id]
    budget = MAX_LENGTH - prompt.input_ids.shape[1]

    def encode(response):
        if not response.endswith(suffix):
            raise RuntimeError("unexpected synthetic ETC-CQ footer")
        prefix = processor.tokenizer.encode(response[:-len(suffix)], add_special_tokens=False)
        remaining = budget - len(prefix) - len(end)
        if remaining < 0:
            raise RuntimeError("completion overflow")
        extra = (pad * (1 + remaining // len(pad)))[:remaining]
        return torch.tensor([prefix + extra + end])

    chosen = encode(positive)
    rejected = encode(positive.replace(".center(0, 0)", ".center(6, 0)"))
    return {key: value.to("cuda") for key, value in prompt.items()}, chosen.to("cuda"), rejected.to("cuda")


def completion_logps(model, prompt, completion):
    ids = torch.cat((prompt["input_ids"], completion), dim=-1)
    inputs = {**prompt, "input_ids": ids, "attention_mask": torch.ones_like(ids)}
    length = completion.size(1)
    if length < 1 or ids.size(1) > MAX_LENGTH:
        raise RuntimeError(f"invalid completion length: {length}")
    base = model.get_base_model()
    hidden = base.model(**inputs, use_cache=False).last_hidden_state[:, -length - 1:-1]

    def token_logps(states, tokens):
        logits = base.lm_head(states)
        return -F.cross_entropy(logits.flatten(0, 1).float(), tokens.flatten(), reduction="none")

    # Checkpoint the vocabulary projection and FP32 softmax in each token chunk.
    # This computes the same log-probabilities without keeping a full vocabulary
    # tensor (and its softmax) live across the whole completion.
    pieces = []
    for states, tokens in zip(hidden.split(128, dim=1), completion.split(128, dim=1)):
        if torch.is_grad_enabled():
            pieces.append(checkpoint(token_logps, states, tokens, use_reentrant=False))
        else:
            pieces.append(token_logps(states, tokens))
    return torch.cat(pieces)


class PeakExceeded(RuntimeError):
    pass


def enforce_peak(stage):
    peak = torch.cuda.max_memory_allocated()
    if peak > LIMITS[stage] * GB:
        raise PeakExceeded(f"{stage} exceeded ceiling at {peak} bytes")


def preference_terms(model, samples):
    """All 16 microbatches share the policy weights before the optimizer step."""
    beta, nll_weight = 0.1, 0.15
    terms = []
    if model.get_base_model().config.text_config.attention_dropout != 0:
        raise RuntimeError("the deterministic split-gradient CF-RPO step requires attention dropout = 0")
    for prompt, chosen, rejected in samples:
        with torch.no_grad():
            with model.disable_adapter():
                ref_positive = completion_logps(model, prompt, chosen).sum()
                ref_negative = completion_logps(model, prompt, rejected).sum()
            positive = completion_logps(model, prompt, chosen)
            negative = completion_logps(model, prompt, rejected)
            delta = beta * ((positive.sum() - negative.sum()) - (ref_positive - ref_negative))
            loss = -F.logsigmoid(delta) - nll_weight * positive.mean()
            slope = -beta * torch.sigmoid(-delta)
        terms.append((loss, slope))
        enforce_peak("cf_rpo")
    return terms


def offline_step(model, optimizer, samples, stage):
    total = torch.zeros((), device="cuda")
    model.train()
    model.config.use_cache = False
    terms = preference_terms(model, samples) if stage == "cf_rpo" else None
    for index in range(GRADIENT_ACCUMULATION):
        sample = index % len(samples)
        prompt, chosen, rejected = samples[sample]
        if terms is None:
            loss = -completion_logps(model, prompt, chosen).mean()
            (loss / GRADIENT_ACCUMULATION).backward()
        else:
            loss, slope = terms[sample]
            # The DPO chain rule needs a shared scalar slope, not two live graphs.
            positive = completion_logps(model, prompt, chosen)
            ((slope * positive.sum() - 0.15 * positive.mean()) / GRADIENT_ACCUMULATION).backward()
            del positive
            negative = completion_logps(model, prompt, rejected)
            ((-slope * negative.sum()) / GRADIENT_ACCUMULATION).backward()
            del negative
        total += loss.detach() / GRADIENT_ACCUMULATION
        enforce_peak(stage)
        if index == 0 or (index + 1) % 4 == 0:
            log("microbatch", stage=stage, completed=index + 1, **cuda_memory())
    if not torch.isfinite(total):
        raise RuntimeError(f"non-finite {stage} loss")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    enforce_peak(stage)
    return {"loss": float(total), "backward_microbatches": GRADIENT_ACCUMULATION}


class RolloutProgress(StoppingCriteria):
    def __init__(self, input_length, sample, candidate):
        self.input_length = input_length
        self.sample = sample
        self.candidate = candidate

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids.size(1) - self.input_length
        if generated % 128 == 0:
            enforce_peak("grpo_online")
            log("generation_progress", sample=self.sample, candidate=self.candidate,
                completion_tokens=generated, **cuda_memory())
        return False


def online_step(model, optimizer, samples, processor):
    sys.path.insert(0, str(ROOT))
    from src.topocq.sandbox_client import run_trial

    model.eval()
    records = []
    for index, (prompt, _, _) in enumerate(samples):
        completions, rewards, old_logps, reference_logps = [], [], [], []
        for candidate in range(GROUP_SIZE):
            progress = RolloutProgress(prompt["input_ids"].size(1), index + 1, candidate + 1)
            with torch.no_grad():
                sequence = model.generate(**prompt, do_sample=True, temperature=0.9, top_p=0.95,
                                          max_new_tokens=NEW_TOKENS,
                                          min_new_tokens=NEW_TOKENS if candidate == 0 else 0, use_cache=True,
                                          stopping_criteria=StoppingCriteriaList([progress]))
                completion = sequence[:, prompt["input_ids"].shape[1]:]
                old_logps.append(completion_logps(model, prompt, completion))
                with model.disable_adapter():
                    reference_logps.append(completion_logps(model, prompt, completion))
            enforce_peak("grpo_online")
            text = processor.tokenizer.decode(completion[0], skip_special_tokens=True)
            verdict = run_trial(text, project=False)
            reward = float(verdict["ok"])
            completions.append(completion)
            rewards.append(reward)
            log("rollout", sample=index + 1, candidate=candidate + 1, completion_tokens=completion.numel(),
                forced_length=candidate == 0, sandbox_status=verdict["status"], reward=reward, **cuda_memory())
        values = torch.tensor(rewards, device="cuda")
        advantages = (values - values.mean()) / (values.std(unbiased=False) + 1e-4)
        records.extend(zip([prompt] * GROUP_SIZE, completions, old_logps, reference_logps, advantages))
    model.train()
    total = torch.zeros((), device="cuda")
    for index in range(GRADIENT_ACCUMULATION):
        prompt, completion, old, ref, advantage = records[index % len(records)]
        current = completion_logps(model, prompt, completion)
        ratio = (current - old).exp()
        clipped = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
        delta = (ref - current).clamp(-20, 20)
        loss = (-clipped + 0.02 * (torch.expm1(delta) - delta)).mean()
        (loss / GRADIENT_ACCUMULATION).backward()
        total += loss.detach() / GRADIENT_ACCUMULATION
        enforce_peak("grpo_online")
        if index == 0 or (index + 1) % 4 == 0:
            log("microbatch", stage="grpo_online", completed=index + 1, **cuda_memory())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    enforce_peak("grpo_online")
    if not torch.isfinite(total):
        raise RuntimeError("non-finite online GRPO loss")
    return {"loss": float(total), "rollouts": len(records), "backward_microbatches": GRADIENT_ACCUMULATION}


def main():
    if sys.executable.lower() != str(REPRODUCTION / "env" / "python.exe").lower():
        raise RuntimeError("launch this script from the contract's ML_PYTHON")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (8, 6):
        raise RuntimeError("one sm_86 CUDA device is required")
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    log("start", device=torch.cuda.get_device_name(), checkpoint=str(CHECKPOINT), torch=torch.__version__,
        compute_capability="sm_86", cpu_weight_offload=False, sft_cf_ceiling_gb=10.2,
        online_grpo_ceiling_gb=10.5, limits_unit="GB", **cuda_memory())
    stage = "load"
    started = time.monotonic()
    try:
        model, optimizer = load_model()
        processor = AutoProcessor.from_pretrained(str(CHECKPOINT), local_files_only=True)
        processor.image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
        samples = [make_sample(index, processor) for index in range(2)]
        log("workload", samples=2, prompt_tokens=[p["input_ids"].numel() for p, _, _ in samples],
            chosen_tokens=[p.numel() for _, p, _ in samples], rejected_tokens=[r.numel() for _, _, r in samples],
            image_grids=[p["image_grid_thw"].tolist() for p, _, _ in samples],
            max_length=MAX_LENGTH, gradient_accumulation=GRADIENT_ACCUMULATION,
            group_size=GROUP_SIZE, max_new_tokens=NEW_TOKENS)
        for stage, limit in LIMITS.items():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            log("stage_start", stage=stage, **cuda_memory())
            measurements = (online_step(model, optimizer, samples, processor)
                            if stage == "grpo_online" else offline_step(model, optimizer, samples, stage))
            torch.cuda.synchronize()
            peak = cuda_memory()
            outcome = "pass" if peak["max_memory_allocated_bytes"] <= limit * GB else "over_ceiling"
            log("stage_end", stage=stage, outcome=outcome, ceiling_gb=limit,
                elapsed_seconds=round(time.monotonic() - started, 2), **measurements, **peak)
            if outcome != "pass":
                return 2
    except PeakExceeded as exc:
        log("stage_end", stage=stage, outcome="over_ceiling", ceiling_gb=LIMITS[stage],
            elapsed_seconds=round(time.monotonic() - started, 2), detail=str(exc), **cuda_memory())
        return 2
    except torch.cuda.OutOfMemoryError as exc:
        log("stage_end", stage=stage, outcome="oom", ceiling_gb=LIMITS.get(stage),
            elapsed_seconds=round(time.monotonic() - started, 2), detail=str(exc), **cuda_memory())
        return 2
    except Exception as exc:
        log("stage_error", stage=stage, exception=type(exc).__name__, detail=str(exc), **cuda_memory())
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
