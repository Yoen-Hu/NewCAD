"""Execute geometry through the real CAD interpreter and a strict five-second boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.topocq import sandbox_client
from src.topocq.eg_pir_schema import parse_etc_cq


# Constructed engineering fixtures. Mini-Gold records require reviewed railway provenance.
CASES = {
    "bearing_pad_hole": [("hole", "circle", "circle(4)", 0, 0)],
    "fastener_two_holes": [("hole1", "circle", "circle(3)", -10, 0),
                           ("hole2", "circle", "circle(3)", 10, 0)],
    "box_culvert": [("opening", "rect", "rect(24, 12)", 0, 0)],
    "slotted_plate": [("slot", "slot2D", "slot2D(16, 5)", 2, 0)],
    "tapered_void_pier": [("void", "rect", "rect(18, 12)", 0, 0)],
}


def make_case(case_name):
    features = []
    lines = [
        "import cadquery as cq",
        "# 1. Outer Solid Parameters", "W = 40", "L = 30", "H = 20",
        "# 2. Hollow Feature Parameters", "CUT_MARGIN = 10.0",
        "# 3. Underconstrained Parameters",
        'solid = cq.Workplane("XY").rect(W, L).extrude(H, combine=False)',
    ]
    dimensions = {"circle(4)": [4], "circle(3)": [3], "rect(24, 12)": [24, 12],
                  "slot2D(16, 5)": [16, 5], "rect(18, 12)": [[18, 12], [14, 10]]}
    for i, (identity, kind, profile, x, y) in enumerate(CASES[case_name]):
        loft = case_name == "tapered_void_pier"
        lines.append(f'tool{i} = (cq.Workplane("XY").workplane(offset=-CUT_MARGIN)'
                     f'.center({x}, {y}).{profile}'
                     + ('.workplane(offset=H + 2 * CUT_MARGIN).center(0, 0).rect(14, 10).loft(combine=False))'
                        if loft else '.extrude(H + 2 * CUT_MARGIN, combine=False))'))
        lines.append(f"solid = solid.cut(tool{i})")
        features.append({
            "hollow_id": identity,
            "cadquery_mode": "loft_cut" if loft else "extrude_cut",
            "spatial_pose": {"base_plane": "XY", "plane_offset": -10.0, "center_2d": [x, y],
                             "extrude_axis": "Z", "through_type": "through_all",
                             "overshoot_ends": ["start", "end"]},
            "profile_shape": {"shape_type": kind, "dimensions": dimensions[profile]},
            "evidence": ["fixture-section"],
        })
    lines.append("result = solid")
    eg_pir = {"part_category": case_name,
              "outer_solid_params": {key: {"value": value, "unit": "mm", "evidence": f"fixture-{key}"}
                                     for key, value in zip("WLH", (40, 30, 20))},
              "hollow_features": features, "constraints": [], "underconstrained_params": []}
    return pack(eg_pir, "\n".join(lines))


def pack(eg_pir, source):
    return f"<eg_pir>\n{json.dumps(eg_pir)}\n</eg_pir>\n<cadquery>\n{source}\n</cadquery>"


@pytest.mark.parametrize("case", CASES)
def test_constructed_etc_cq_cases(case):
    response = make_case(case)
    result = sandbox_client.run_trial(response, response)
    assert result["ok"], result
    assert result["elapsed_seconds"] < 5
    assert result["geometry"]["num_solids"] == 1 and result["geometry"]["is_watertight"]
    assert result["R_cut_health"] == 1
    assert len(result["cut_steps"]) == len(CASES[case])
    assert all(cut["volume_removed"] > 0 and cut["overshoot_ok"] for cut in result["cut_steps"])
    assert result["iou"] == pytest.approx(1.0)
    assert len(result["iou_attempts"]) == 4
    assert result["2D-Reproj-IoU"] == pytest.approx(1.0)
    assert result["Vis-CD2D"] == pytest.approx(0.0) and result["Hid-CD2D"] == pytest.approx(0.0)
    assert set(result["views"]) == {"front", "top", "side"}
    assert result["views"]["front"]["hidden"]


@pytest.mark.parametrize(("name", "old", "new", "expected"), [
    ("coplanar_flush", "offset=-CUT_MARGIN", "offset=0", "invalid_geometry"),
    ("infinite_loop", "result = solid", "while True:\n    pass\nresult = solid", "static_rejected"),
    ("disjoint_solids", "solid = solid.cut(tool0)", "solid = solid.union(tool0) # [EMBEDDED_SUBASSEMBLY]", "invalid_geometry"),
    ("syntax_error", "result = solid", "result = (", "static_rejected"),
    ("shifted_cavity", ".center(0, 0)", ".center(6, 0)", "ok"),
])
def test_adversarial(name, old, new, expected):
    reference = make_case("bearing_pad_hole")
    bad = reference.replace(old, new)
    if name == "disjoint_solids":
        bad = bad.replace(".center(0, 0)", ".center(100, 0)")
        _, source = parse_etc_cq(reference)
        source = source.replace(old, new).replace(".center(0, 0)", ".center(100, 0)")
        result = sandbox_client.run_script(source)
    else:
        result = sandbox_client.run_trial(bad, reference)
    assert result["status"] == expected, result
    if name == "coplanar_flush":
        assert result["R_cut_health"] == 0 and not result["cut_steps"][0]["overshoot_ok"]
    if name == "shifted_cavity":
        assert result["2D-Reproj-IoU"] < 0.95 and result["iou"] < 1
        assert result["Hid-CD2D"] > 0


def test_schema_rejects_reversed_hollow_order():
    response = make_case("bearing_pad_hole")
    pir, source = parse_etc_cq(response)
    data = pir.model_dump()
    item = data["hollow_features"][0]
    pose = item.pop("spatial_pose")
    item["spatial_pose"] = pose
    assert sandbox_client.run_trial(pack(data, source))["status"] == "static_rejected"


def test_subprocess_timeout_kills_child(monkeypatch):
    # Force a worker-side stall independently of the generated-code static guard.
    real_run = subprocess.run

    def stalled(command, **kwargs):
        assert kwargs["timeout"] == 5.0
        return real_run([command[0], "-c", "import time; time.sleep(60)"], **kwargs)

    monkeypatch.setattr(sandbox_client.subprocess, "run", stalled)
    result = sandbox_client.run_trial(make_case("bearing_pad_hole"))
    assert result["status"] == "timeout" and result["timeout_seconds"] == 5.0
    assert sandbox_client.run_trial("not a valid response")["status"] == "static_rejected"



def test_mini_gold_5():
    source = os.environ.get("TOPOCQ_MINI_GOLD")
    default_directory = sandbox_client.REPOSITORY.parents[1] / "data" / "mini_gold_5"
    if not source and not default_directory.is_dir():
        pytest.skip("TOPOCQ_MINI_GOLD must point to five reviewed real ETC-CQ .txt / reference .step pairs")
    directory = Path(source or default_directory).resolve(strict=True)
    responses = sorted(directory.glob("*.txt"))
    assert len(responses) == 5, f"expected five reviewed ETC-CQ responses in {directory}"
    for response in responses:
        reference = response.with_suffix(".step")
        result = sandbox_client.run_trial(response.read_text(encoding="utf-8-sig"), reference)
        assert result["ok"] and result["elapsed_seconds"] < 5, (response.name, result)
        assert result["geometry"]["is_watertight"] and result["R_cut_health"] == 1
        assert result["iou"] is not None and result["2D-Reproj-IoU"] is not None


def test_voxel_fallback_on_native_ocp_mesh():
    command = """
import json
import sys
from src.topocq.eg_pir_schema import parse_etc_cq
from src.topocq.sandbox_worker_ocp import execute
from src.topocq.solid_iou_ocp import _brep_iou, _voxel_compare
pairs = json.load(sys.stdin)
shapes = []
for sample in pairs:
    pir, code = parse_etc_cq(sample)
    shape, info = execute(code, pir)
    if not info["ok"]:
        raise RuntimeError(info)
    shapes.append(shape)
first, second = shapes
print(json.dumps({"fuzzy": _brep_iou(first, second, 40, 4e-4),
                  "micro": _brep_iou(first.translate((0.0004, -0.000284, 0.000172)), second, 40),
                  "voxel": _voxel_compare(first, second, 40),
                  "shifted": _voxel_compare(first, second.translate((5, 0, 0)), 40)}))
"""
    result = subprocess.run([str(sandbox_client.CAD_PYTHON), "-c", command],
                            input=json.dumps([make_case("box_culvert")] * 2),
                            encoding="utf-8", capture_output=True, check=True, timeout=5,
                            cwd=sandbox_client.REPOSITORY, env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    metrics = json.loads(result.stdout)
    assert metrics["fuzzy"] == pytest.approx(1)
    assert 0.999 < metrics["micro"] < 1
    assert metrics["voxel"]["voxel_iou"] == {"64": 1.0, "128": 1.0}
    assert metrics["voxel"]["Chamfer3D"] == pytest.approx(0.0)
    assert 0 < metrics["shifted"]["voxel_iou"]["128"] < 1
    assert metrics["shifted"]["Chamfer3D"] > 0



def test_local_overshoot_at_taper_and_step():
    pir, source = parse_etc_cq(make_case("bearing_pad_hole"))
    data = pir.model_dump()
    pose = data["hollow_features"][0]["spatial_pose"]
    pose.update(base_plane="YZ", extrude_axis="X", plane_offset=-23, center_2d=[-10, 30])
    prefix = source.split("solid = ")[0]
    cutter = ('tool = cq.Workplane("YZ").workplane(offset=-13-CUT_MARGIN).center(-10, 30)'
              '.circle(2).extrude(26+2*CUT_MARGIN, combine=False)\n'
              'solid = solid.cut(tool)\nresult = solid')
    envelopes = [
        'solid = cq.Workplane("XY").center(0, -10).rect(40,20).workplane(offset=40).rect(20,20).loft(ruled=True)',
        ('solid = cq.Workplane("XZ").polyline([(-20,0),(20,0),(20,20),(10,20),'
         '(10,40),(-10,40),(-10,20),(-20,20)]).close().extrude(20, combine=False)'),
    ]
    with sandbox_client.SandboxPool() as pool:
        for envelope in envelopes:
            report = pool.run_trial(pack(data, prefix + envelope + "\n" + cutter), project=False)
            assert report["ok"], report
            assert min(report["cut_steps"][0]["axial_overshoot"]) > 10


def test_edge_finishing_runs_only_after_cuts():
    pir, source = parse_etc_cq(make_case("bearing_pad_hole"))
    original = "result = solid"
    with sandbox_client.SandboxPool() as pool:
        for method in ("chamfer", "fillet"):
            finish = f'solid = solid.edges(">Z").{method}(0.5)'
            candidate = source.replace(original, finish + "\n" + original)
            report = pool.run_trial(pack(pir.model_dump(), candidate), project=False)
            assert report["ok"] and report["geometry"]["is_watertight"], report
            premature = source.replace("solid = solid.cut(tool0)", finish + "\nsolid = solid.cut(tool0)")
            assert pool.run_trial(pack(pir.model_dump(), premature))["status"] == "static_rejected"


def test_persistent_worker_replacement_and_timing(monkeypatch):
    from time import monotonic

    original_popen = subprocess.Popen
    stalled = True

    def start(command, **kwargs):
        nonlocal stalled
        if stalled:
            stalled = False
            return original_popen([command[0], "-c", "import time; time.sleep(60)"], **kwargs)
        return original_popen(command, **kwargs)

    monkeypatch.setattr(sandbox_client.subprocess, "Popen", start)
    sample = make_case("bearing_pad_hole")
    with sandbox_client.SandboxPool() as pool:
        started = monotonic()
        timeout = pool.run_trial(sample, project=False)
        assert timeout["status"] == "timeout" and timeout["timeout_seconds"] == 5
        assert monotonic() - started < 6
        ready = pool.warm()
        assert ready["ok"] and ready["worker_pid"] != timeout["worker_pid"], ready
        elapsed = []
        for _ in range(20):
            report = pool.run_trial(sample, project=False)
            assert report["ok"] and report["worker_pid"] == ready["worker_pid"], report
            elapsed.append(report["elapsed_seconds"])
        following = pool.run_trial(sample, project=False)
        assert following["ok"] and following["worker_pid"] != ready["worker_pid"], following
        print(f"persistent CAD worker: warm={ready['elapsed_seconds']:.3f}s "
              f"min={min(elapsed):.3f}s max={max(elapsed):.3f}s over {len(elapsed)} jobs")


def test_five_dimensional_evaluation():
    from src.topocq.eval_5d import METRICS, evaluate

    sample = make_case("bearing_pad_hole")
    with sandbox_client.SandboxPool() as pool:
        report = evaluate([{"prediction": sample, "reference": sample},
                           {"prediction": "invalid", "reference": sample}], sandbox=pool,
                          max_edit_parameters=1)
    good, bad = report["samples"]
    assert tuple(report["metrics"]) == METRICS
    assert good["metrics"]["IoU_best"] == pytest.approx(1)
    assert good["metrics"]["Param-EM@1%-hollow"] == 1
    assert report["metrics"]["VSR"] == 0.5
    assert bad["metrics"]["Hollow-Cut-F1"] == 0
    assert report["metric_counts"]["Edit-Pass@±10%"] == 1
    assert good["metrics"]["Edit-Pass@±10%"] == 1
