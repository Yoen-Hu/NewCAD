"""EG-PIR and the restricted, straight-line ETC-CQ program format."""

import ast
import json
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Dimension(StrictModel):
    value: FiniteFloat
    unit: str = Field(min_length=1)
    evidence: str | list[str]

    @model_validator(mode="after")
    def evidence_present(self):
        if not self.evidence or any(not item.strip() for item in (
            [self.evidence] if isinstance(self.evidence, str) else self.evidence
        )):
            raise ValueError("outer dimension needs drawing evidence")
        return self


class SpatialPose(StrictModel):
    base_plane: Literal["XY", "XZ", "YZ"]
    plane_offset: FiniteFloat
    center_2d: list[FiniteFloat] = Field(min_length=2, max_length=2)
    extrude_axis: str = Field(pattern=r"^[+-]?[xyzXYZ]$|^[xyzXYZ][+-]$")
    through_type: Literal["through_all", "blind_internal", "blind_one_side"]
    overshoot_ends: list[str | FiniteFloat] = Field(max_length=2)

    @model_validator(mode="after")
    def check_axis_and_ends(self):
        if self.extrude_axis.strip("+-").upper() != {"XY": "Z", "XZ": "Y", "YZ": "X"}[self.base_plane]:
            raise ValueError("extrude_axis must be normal to base_plane")
        count = {"through_all": 2, "blind_internal": 0, "blind_one_side": 1}[self.through_type]
        if len(self.overshoot_ends) != count or len(set(self.overshoot_ends)) != count:
            raise ValueError("overshoot_ends must identify each open face once")
        return self


class ProfileShape(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    shape_type: Literal["rect", "circle", "slot2D", "rounded_rect", "polygon", "dual_section_rect_chamfer"]

    @model_validator(mode="before")
    @classmethod
    def check_dimensions(cls, data):
        def valid(value):
            if type(value) in (float, int):
                return abs(value) <= 1e12 and math.isfinite(value)
            return isinstance(value, list) and bool(value) and all(valid(item) for item in value)

        if isinstance(data, dict):
            dims = [value for name, value in data.items() if name != "shape_type"]
            if not dims or not all(valid(value) for value in dims):
                raise ValueError("profile needs finite numeric dimensions or coordinate arrays")
        return data


class HollowFeature(StrictModel):
    hollow_id: str = Field(min_length=1)
    cadquery_mode: Literal["extrude_cut", "loft_cut", "oblique_cut"]
    spatial_pose: SpatialPose
    profile_shape: ProfileShape
    evidence: list[str] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def check_pose_order(cls, data):
        if isinstance(data, dict) and {"spatial_pose", "profile_shape"} <= data.keys():
            keys = list(data)
            if keys.index("spatial_pose") > keys.index("profile_shape"):
                raise ValueError("spatial_pose must precede profile_shape")
        return data

    @model_validator(mode="after")
    def check_evidence(self):
        if any(not item.strip() for item in self.evidence):
            raise ValueError("hollow evidence contains an empty drawing ID")
        return self


class UnderconstrainedParameter(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    default: FiniteFloat
    range: list[FiniteFloat] = Field(min_length=2, max_length=2)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_default(self):
        if not self.range[0] <= self.default <= self.range[1]:
            raise ValueError("underconstrained default must lie within its range")
        return self


class EGPIR(StrictModel):
    part_category: str = Field(min_length=1)
    outer_solid_params: dict[str, Dimension] = Field(min_length=1)
    hollow_features: list[HollowFeature]
    constraints: list[str]
    underconstrained_params: list[UnderconstrainedParameter]

    @model_validator(mode="after")
    def check_unique(self):
        hollow_ids = [item.hollow_id for item in self.hollow_features]
        unknown = [item.name for item in self.underconstrained_params]
        if len(set(hollow_ids)) != len(hollow_ids) or len(set(unknown)) != len(unknown):
            raise ValueError("duplicate hollow ID or underconstrained parameter")
        if set(unknown) & self.outer_solid_params.keys():
            raise ValueError("an underconstrained parameter cannot also be a determined outer dimension")
        return self


class StaticError(ValueError):
    """ETC-CQ program rejected before geometry execution."""


HEADERS = ("# 1. Outer Solid Parameters", "# 2. Hollow Feature Parameters", "# 3. Underconstrained Parameters")
CONSTRUCTORS = {"extrude", "revolve", "loft"}
PROFILES = {
    "rect", "circle", "slot2D", "polygon", "polyline", "line", "lineTo", "hLine", "vLine",
    "moveTo", "threePointArc", "radiusArc", "sagittaArc", "spline", "close", "wire", "ellipse",
}
POSE = {"workplane", "transformed", "center"}
RIGID = {"translate", "rotate"}
EMBEDDED = "# [EMBEDDED_SUBASSEMBLY]"
OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub)


def _chain(expr):
    calls = []
    while isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        calls.append(expr)
        expr = expr.func.value
    return expr, list(reversed(calls))


def _duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StaticError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _numeric(expr, constants):
    """Evaluate only bounded numeric header expressions."""
    if isinstance(expr, ast.Constant) and type(expr.value) in (int, float):
        value = expr.value
    elif isinstance(expr, ast.Name) and expr.id in constants:
        value = constants[expr.id]
    elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
        value = _numeric(expr.operand, constants)
        if isinstance(expr.op, ast.USub):
            value = -value
    elif isinstance(expr, ast.BinOp) and isinstance(expr.op, OPERATORS):
        left = _numeric(expr.left, constants)
        right = _numeric(expr.right, constants)
        if isinstance(expr.op, ast.Add):
            value = left + right
        elif isinstance(expr.op, ast.Sub):
            value = left - right
        elif isinstance(expr.op, ast.Mult):
            value = left * right
        else:
            if right == 0:
                raise StaticError("division by zero in constant")
            value = left / right
    else:
        raise StaticError("expected a numeric expression")
    if abs(value) > 1e12 or not math.isfinite(value):
        raise StaticError("numeric expression out of bounds")
    return value


def _argument(expr, constants):
    if isinstance(expr, (ast.List, ast.Tuple)):
        for item in expr.elts:
            _argument(item, constants)
    elif isinstance(expr, ast.Constant) and type(expr.value) in (str, bool, type(None)):
        if isinstance(expr.value, str) and len(expr.value) > 128:
            raise StaticError("string argument exceeds limit")
    else:
        _numeric(expr, constants)


def _uses_margin(call):
    return any(isinstance(node, ast.Name) and node.id == "CUT_MARGIN" for node in ast.walk(call))


def _validate_constructor(root, calls, feature, inherited=None):
    cq_root = isinstance(root, ast.Name) and root.id == "cq" and calls[0].func.attr == "Workplane"
    initial_offset = False
    if cq_root:
        initial_offset = any(kw.arg == "origin" for kw in calls[0].keywords)
        calls = calls[1:]
    elif inherited is None:
        raise StaticError("geometry must start from cq.Workplane or an explicit sketch")
    centered, profiled, posed, modes = inherited or (False, False, initial_offset, ())
    for call in calls:
        method = call.func.attr
        if method in POSE:
            if modes:
                raise StaticError("workplane and center must precede solid construction")
            if method in {"workplane", "transformed"}:
                posed, centered = True, False
            else:
                centered = True
        elif method in PROFILES:
            if modes:
                raise StaticError("solid-to-sketch stacking is forbidden")
            if feature and not (posed and centered):
                raise StaticError("cut tool must set workplane offset and center before the profile")
            profiled = True
        elif method in CONSTRUCTORS:
            if not profiled or modes:
                raise StaticError("a solid must be formed by exactly one profile constructor")
            if method == "extrude":
                if len(call.args) > 1 or any(
                    kw.arg == "combine" and not (isinstance(kw.value, ast.Constant) and kw.value.value is False)
                    for kw in call.keywords
                ):
                    raise StaticError("combined/stacked extrusion is forbidden")
            modes = (method,)
        elif method in RIGID and modes:
            continue
        else:
            raise StaticError(f"unsupported geometry method: {method}")
    return centered, profiled, posed, modes


def validate_cadquery(source: str, eg_pir: EGPIR | None = None) -> ast.Module:
    """Validate syntax, phase dataflow and cut-tool correspondence without importing CAD."""
    if len(source) > 32_000:
        raise StaticError("script exceeds size limit")
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, RecursionError) as exc:
        raise StaticError(f"invalid Python syntax: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > 2_000:
        raise StaticError("script exceeds AST limit")
    lines = source.splitlines()
    locations = [[i for i, line in enumerate(lines, 1) if line.strip() == header] for header in HEADERS]
    if any(len(items) != 1 for items in locations):
        raise StaticError("three exact ETC-CQ parameter headers are required")
    outer, hollow, unknown = [items[0] for items in locations]
    if not outer < hollow < unknown:
        raise StaticError("parameter headers must appear in canonical order")
    constants, under, geometry = {}, set(), {}
    envelope = None
    cut_count = 0
    imported = False
    finished = False
    building = False
    finishing = False
    for stmt in tree.body:
        if (isinstance(stmt, ast.Import) and [(a.name, a.asname) for a in stmt.names] == [("cadquery", "cq")]
                and not imported and stmt.lineno < outer):
            imported = True
            continue
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            raise StaticError("only single-name assignments and the canonical CadQuery import are allowed")
        name = stmt.targets[0].id
        if name.startswith("_") or name in {"cq", "checked_cut", "checked_union"} or finished:
            raise StaticError("reserved name or statement after result")
        expr = stmt.value
        if stmt.lineno < outer:
            raise StaticError("assignment before parameter headers")
        if not isinstance(expr, ast.Call) and name != "result":
            if building or name in constants or name in geometry:
                raise StaticError("parameters must be declared once, before all geometry")
            constants[name] = _numeric(expr, constants)
            if hollow < stmt.lineno < unknown and name == "CUT_MARGIN" and constants[name] == 10.0:
                continue
            if name == "CUT_MARGIN":
                raise StaticError("CUT_MARGIN = 10.0 belongs in the hollow parameter header")
            if stmt.lineno > unknown:
                if "# [UNDERCONSTRAINED]" not in lines[stmt.end_lineno - 1]:
                    raise StaticError("underconstrained parameter needs its explicit tag")
                under.add(name)
            continue
        if stmt.lineno <= unknown or constants.get("CUT_MARGIN") != 10.0:
            raise StaticError("finish the parameter headers before geometry")
        building = True
        if name == "result":
            if not isinstance(expr, ast.Name) or expr.id != envelope:
                raise StaticError("result must be the final main solid")
            finished = True
            continue
        root, calls = _chain(expr)
        if not calls or not isinstance(root, ast.Name) or name in constants:
            raise StaticError("unsupported geometry expression")
        method = calls[0].func.attr
        if method == "edges":
            if (root.id != envelope or name != envelope or len(calls) != 2
                    or calls[1].func.attr not in {"chamfer", "fillet"}
                    or (eg_pir is not None and cut_count != len(eg_pir.hollow_features))):
                raise StaticError("edge finishing belongs after all cuts on the main solid")
            selector, finish = calls
            if (len(selector.args) > 1 or selector.keywords
                    or any(not isinstance(arg, ast.Constant) or not isinstance(arg.value, str)
                           or len(arg.value) > 128 for arg in selector.args)):
                raise StaticError("edges expects an optional literal selector")
            limit = 2 if finish.func.attr == "chamfer" else 1
            if not 1 <= len(finish.args) <= limit or finish.keywords:
                raise StaticError("edge finishing expects positive positional lengths")
            if any(_numeric(arg, constants) <= 0 for arg in finish.args):
                raise StaticError("edge finishing lengths must be positive")
            finishing = True
            continue
        if finishing:
            raise StaticError("only edge finishing and result are allowed after finishing starts")
        if method in {"cut", "union"}:
            if len(calls) != 1 or name != envelope or root.id != envelope:
                raise StaticError("boolean must reassign the current main solid directly")
            call = calls[0]
            if len(call.args) != 1 or call.keywords or not isinstance(call.args[0], ast.Name):
                raise StaticError("boolean needs one named solid tool")
            tool = call.args[0].id
            if tool == envelope or tool not in geometry or not geometry[tool][0][3]:
                raise StaticError("boolean tool must be an independent constructed solid")
            state, calls_for_tool, consumed = geometry[tool]
            if consumed:
                raise StaticError("a tool must be dedicated to one boolean")
            geometry[tool] = (state, calls_for_tool, True)
            if method == "union":
                if EMBEDDED not in lines[stmt.end_lineno - 1] or cut_count:
                    raise StaticError("only tagged, deeply embedded subassemblies are allowed before cuts")
                continue
            if not state[0]:
                raise StaticError("cut tool must position its center before drawing its profile")
            if eg_pir is not None:
                if cut_count >= len(eg_pir.hollow_features):
                    raise StaticError("more cuts than declared hollow_features")
                feature = eg_pir.hollow_features[cut_count]
                expected = "loft" if feature.cadquery_mode == "loft_cut" else "extrude"
                if state[3] != (expected,):
                    raise StaticError(f"{feature.hollow_id}: wrong cut constructor")
                workplane = calls_for_tool[0]
                plane = workplane.args[0] if workplane.args else next(
                    (kw.value for kw in workplane.keywords if kw.arg == "inPlane"), ast.Constant("XY")
                )
                if not isinstance(plane, ast.Constant) or plane.value != feature.spatial_pose.base_plane:
                    raise StaticError(f"{feature.hollow_id}: tool workplane disagrees with EG-PIR")
                profile = feature.profile_shape.shape_type
                if profile in {"circle", "rect", "slot2D"} and not any(
                    call.func.attr == profile for call in calls_for_tool
                ):
                    raise StaticError(f"{feature.hollow_id}: tool profile disagrees with EG-PIR")
                if feature.spatial_pose.overshoot_ends and not any(
                    _uses_margin(call) for call in calls_for_tool
                    if call.func.attr in {"Workplane", "workplane", "transformed", "loft", "extrude"}
                ):
                    raise StaticError(f"{feature.hollow_id}: open cut missing CUT_MARGIN")
            cut_count += 1
            continue
        if name in geometry:
            raise StaticError("geometry names cannot be overwritten apart from main-solid booleans")
        if root.id == "cq" and method != "Workplane":
            raise StaticError("only cq.Workplane may be constructed")
        if root.id != "cq" and root.id not in geometry:
            raise StaticError("geometry root must be cq.Workplane or a previous sketch")
        previous = geometry.get(root.id)
        if previous and previous[0][3]:
            raise StaticError("cannot build on an existing solid")
        for call in calls:
            for arg in call.args:
                _argument(arg, constants)
            for kw in call.keywords:
                if kw.arg is None or kw.arg.startswith("_"):
                    raise StaticError("keyword expansion and private keywords are forbidden")
                _argument(kw.value, constants)
        state = _validate_constructor(root, calls, envelope is not None, previous[0] if previous else None)
        if previous:
            geometry[root.id] = (previous[0], previous[1], True)
        inherited = previous[1] if previous else []
        geometry[name] = (state, inherited + calls, False)
        if envelope is None and state[3]:
            envelope = name
            geometry[name] = (state, inherited + calls, True)
    if not finished or any(not consumed for _, _, consumed in geometry.values()):
        raise StaticError("program must finish with result and may not leave unused geometry")
    if eg_pir is not None:
        if cut_count != len(eg_pir.hollow_features):
            raise StaticError("each declared hollow must have one explicit cut")
        if under != {item.name for item in eg_pir.underconstrained_params}:
            raise StaticError("underconstrained header disagrees with EG-PIR")
        if any(constants[item.name] != item.default for item in eg_pir.underconstrained_params):
            raise StaticError("underconstrained header default disagrees with EG-PIR")
    return tree


def parse_etc_cq(text: str) -> tuple[EGPIR, str]:
    """Parse one complete response with the two XML blocks in canonical order."""
    match = re.fullmatch(r"\s*<eg_pir>\s*(.*?)\s*</eg_pir>\s*<cadquery>\s*(.*?)\s*</cadquery>\s*", text, re.DOTALL)
    if not match:
        raise StaticError("expected exactly <eg_pir> then <cadquery>")
    json_text, source = match.groups()
    if len(json_text) > 32_000:
        raise StaticError("EG-PIR exceeds size limit")
    eg_pir = EGPIR.model_validate(json.loads(json_text, object_pairs_hook=_duplicates))
    validate_cadquery(source, eg_pir)
    return eg_pir, source
