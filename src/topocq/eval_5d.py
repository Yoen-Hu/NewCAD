"""Aggregate TopoCQ evaluation without importing the CAD kernel in ML_PYTHON."""

import ast
import json
import math
from statistics import mean

from .eg_pir_schema import _numeric, parse_etc_cq
from .sandbox_client import run_trial

LENGTH_UNITS = {"mm": 1, "cm": 10, "m": 1000}
METRICS = ("VSR", "WR-Single", "IoU_best", "2D-Reproj-IoU", "Vis-CD2D", "Hid-CD2D",
           "R_cut_health", "Param-EM@1%-outer", "Param-EM@1%-hollow", "Hollow-Cut-F1",
           "UnderCon-F1", "Edit-Pass@±10%")


def _parsed(response):
    if isinstance(response, str):
        try:
            return parse_etc_cq(response)
        except (ValueError, OverflowError, RecursionError):
            pass
    return None, None


def _numeric_fields(value, prefix=()):
    if type(value) in (int, float):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _numeric_fields(child, (*prefix, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _numeric_fields(child, (*prefix, index))


def _outer(pir):
    if pir is None:
        return {}
    result = {}
    for name, dim in pir.outer_solid_params.items():
        unit = dim.unit.lower().strip()
        result[name] = ("mm", dim.value * LENGTH_UNITS[unit]) if unit in LENGTH_UNITS else (unit, dim.value)
    return result


def _hollow(pir):
    if pir is None:
        return {}
    fields = {}
    for feature in pir.hollow_features:
        for section in (feature.spatial_pose, feature.profile_shape):
            for key, value in _numeric_fields(section.model_dump()):
                fields[(feature.hollow_id, type(section).__name__, *key)] = ("mm", value)
    return fields


def _parameter_em(predicted, target):
    keys = predicted.keys() | target.keys()
    if not keys:
        return None
    matched = 0
    for key in predicted.keys() & target.keys():
        unit, value = predicted[key]
        target_unit, target_value = target[key]
        if unit == target_unit and abs(value - target_value) <= max(1e-9, 0.01 * abs(target_value)):
            matched += 1
    return matched / len(keys)


def _f1(predicted, target):
    size = len(predicted) + len(target)
    return 2 * len(predicted & target) / size if size else 1.0


def _edits(pir, source, max_parameters):
    tree = ast.parse(source)
    declarations, geometry = {}, []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            if name != "result" and not isinstance(stmt.value, ast.Call):
                declarations[name] = stmt.value
            else:
                geometry.append(stmt.value)

    def dependencies(expressions):
        return {node.id for expr in expressions for node in ast.walk(expr)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}

    active = dependencies(geometry)
    for name, expr in reversed(list(declarations.items())):
        if name in active:
            active.update(dependencies([expr]))
    bounds = {param.name: param.range for param in pir.underconstrained_params}
    base = pir.model_dump()
    lines = source.splitlines(keepends=True)
    selected = 0
    for name, expr in declarations.items():
        if name == "CUT_MARGIN" or name not in active or expr.lineno != expr.end_lineno:
            continue
        if not isinstance(expr, (ast.Constant, ast.UnaryOp)) or dependencies([expr]):
            continue
        value = _numeric(expr, {})
        if value == 0:
            continue
        pair = (value * 0.9, value * 1.1)
        if name in bounds and not all(bounds[name][0] <= v <= bounds[name][1] for v in pair):
            continue
        if max_parameters is not None and selected >= max_parameters:
            break
        selected += 1
        for delta, modified in zip((-0.1, 0.1), pair):
            data = json.loads(json.dumps(base))
            if name in data["outer_solid_params"]:
                data["outer_solid_params"][name]["value"] = modified
            for record in data["underconstrained_params"]:
                if record["name"] == name:
                    record["default"] = modified
            changed = lines.copy()
            line = changed[expr.lineno - 1].encode("utf-8")
            changed[expr.lineno - 1] = (line[:expr.col_offset] + repr(modified).encode("ascii")
                                        + line[expr.end_col_offset:]).decode("utf-8")
            yield name, delta, f"<eg_pir>{json.dumps(data)}</eg_pir>\n<cadquery>{''.join(changed)}</cadquery>"


def evaluate_sample(prediction, *, reference=None, reference_step=None, sandbox=None, max_edit_parameters=None):
    """Score one prediction. When supplied, reference_step is the geometric authority."""
    runner = sandbox.run_trial if sandbox is not None else run_trial
    pir, source = _parsed(prediction)
    truth, _ = _parsed(reference)
    verdict = runner(prediction, reference=reference_step or reference, project=False)
    geometry = verdict.get("geometry", {})
    solid = geometry.get("num_solids") == 1 and bool(geometry.get("is_watertight"))
    declared = {f.hollow_id: f.cadquery_mode for f in pir.hollow_features} if pir else {}
    healthy = {(step["hollow_id"], declared[step["hollow_id"]]) for step in verdict.get("cut_steps", [])
               if step.get("healthy") and step.get("hollow_id") in declared}
    target = {(f.hollow_id, f.cadquery_mode) for f in truth.hollow_features} if truth else set()
    under = {param.name for param in pir.underconstrained_params} if pir else set()
    under_target = {param.name for param in truth.underconstrained_params} if truth else set()
    metrics = {
        "VSR": float(bool(geometry) and bool(geometry.get("is_valid"))),
        "WR-Single": float(solid), "IoU_best": verdict.get("iou"),
        "2D-Reproj-IoU": verdict.get("2D-Reproj-IoU"),
        "Vis-CD2D": verdict.get("Vis-CD2D"), "Hid-CD2D": verdict.get("Hid-CD2D"),
        "R_cut_health": verdict.get("R_cut_health", 0.0),
        "Param-EM@1%-outer": _parameter_em(_outer(pir), _outer(truth)) if truth else None,
        "Param-EM@1%-hollow": _parameter_em(_hollow(pir), _hollow(truth)) if truth else None,
        "Hollow-Cut-F1": _f1(healthy, target) if truth else None,
        "UnderCon-F1": _f1(under, under_target) if truth else None,
        "Edit-Pass@±10%": None,
    }
    edits = []
    if pir and solid:
        for name, delta, edited in _edits(pir, source, max_edit_parameters):
            replay = runner(edited, reference=prediction, align=False, project=False)
            returned = replay.get("geometry", {})
            passed = (replay["ok"] and returned.get("num_solids") == 1
                      and returned.get("is_watertight") and replay.get("R_cut_health") == 1
                      and replay.get("iou") is not None and replay["iou"] < 1.0 - 1e-8)
            edits.append({"parameter": name, "delta": delta, "status": replay["status"],
                          "iou_to_original": replay.get("iou"), "pass": bool(passed)})
    if edits:
        metrics["Edit-Pass@±10%"] = mean(edit["pass"] for edit in edits)
    return {"metrics": metrics, "sandbox": verdict, "edits": edits}


def evaluate(samples, *, sandbox=None, max_edit_parameters=None):
    """Macro-average observed metrics and report each denominator and sample."""
    entries = [evaluate_sample(**sample, sandbox=sandbox, max_edit_parameters=max_edit_parameters)
               for sample in samples]
    summary, counts = {}, {}
    for metric in METRICS:
        values = [entry["metrics"][metric] for entry in entries
                  if entry["metrics"][metric] is not None and math.isfinite(entry["metrics"][metric])]
        counts[metric] = len(values)
        summary[metric] = mean(values) if values else None
    return {"sample_count": len(entries), "metrics": summary, "metric_counts": counts, "samples": entries}
