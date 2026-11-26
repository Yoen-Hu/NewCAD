"""One-shot CadQuery/OCP worker. Entry point is stdin/stdout JSON only."""

import ast
import json
import math
import sys


def _cap_clearances(main, cutter, axis, tolerance):
    """Measure from each planar cutter cap to the first locally intersected material."""
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
    from OCP.TopAbs import TopAbs_OUT
    from OCP.gp import gp_Dir, gp_Lin

    solid = main.Solids()[0].wrapped
    classifier = BRepClass3d_SolidClassifier(solid)
    intersect = IntCurvesFace_ShapeIntersector()
    intersect.Load(solid, tolerance)
    caps = {}
    for face in cutter.Faces():
        if face.geomType() != "PLANE":
            continue
        normal = face.normalAt()
        direction = normal.dot(axis)
        if abs(direction) < 1 - 1e-8:
            continue
        cap = face.Center()
        outward = 1 if direction > 0 else -1
        if outward not in caps or outward * cap.dot(axis) > outward * caps[outward][0].dot(axis):
            caps[outward] = (cap, face)
    clearances = [None, None]
    for index, outward in enumerate((-1, 1)):
        entry = caps.get(outward)
        if entry is None:
            continue
        cap, face = entry
        point = cap.toPnt()
        classifier.Perform(point, tolerance)
        if classifier.State() != TopAbs_OUT:
            clearances[index] = 0.0
            continue
        vector = axis.multiply(-outward).toTuple()
        reach = abs((caps[1][0] - caps[-1][0]).dot(axis)) if len(caps) == 2 else 0.0
        intersect.Perform(gp_Lin(point, gp_Dir(*vector)), 0.0, reach + tolerance)
        if not intersect.IsDone():
            raise RuntimeError("OCCT could not intersect the cutter centerline with the main solid")
        if intersect.NbPnt():
            clearances[index] = min(intersect.WParameter(i) for i in range(1, intersect.NbPnt() + 1))
        else:
            # A wide trim can straddle the material even when its centerline stays
            # in air. The full cap clearance is a conservative check in that case.
            clearances[index] = face.distance(main)
    return clearances


def execute(source, eg_pir=None):
    import cadquery as cq
    from OCP.BRep import BRep_Tool
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_SOLID
    from OCP.TopoDS import TopoDS_Iterator

    from .eg_pir_schema import validate_cadquery
    from .solid_iou_ocp import boolean, scale_for

    class Instrument(ast.NodeTransformer):
        def visit_Call(self, call):
            self.generic_visit(call)
            if isinstance(call.func, ast.Attribute) and call.func.attr in {"cut", "union"}:
                name = "checked_" + call.func.attr
                replacement = ast.Call(ast.Name(name, ast.Load()), [call.func.value, *call.args], [])
                return ast.copy_location(replacement, call)
            return call

    def contains_one_solid(topology):
        while topology.ShapeType() in (TopAbs_COMPOUND, TopAbs_COMPSOLID):
            children = TopoDS_Iterator(topology)
            if not children.More():
                return False
            topology = children.Value()
            children.Next()
            if children.More():
                return False
        return topology.ShapeType() == TopAbs_SOLID

    def health(shape):
        solids = shape.Solids()
        valid = BRepCheck_Analyzer(shape.wrapped).IsValid()
        closed = contains_one_solid(shape.wrapped) and len(solids) == 1 and bool(solids[0].Shells()) and all(
            BRep_Tool.IsClosed_s(shell.wrapped) for shell in solids[0].Shells()
        )
        volume = shape.Volume()
        return {"num_solids": len(solids), "is_watertight": bool(closed and valid),
                "is_valid": bool(valid), "volume": volume}

    cuts = []
    unions = []

    def one_shape(workplane):
        if workplane.size() != 1:
            raise ValueError("workplane must contain exactly one shape")
        return workplane.val()

    def checked_cut(current, tool):
        main, cutter = one_shape(current), one_shape(tool)
        scale = scale_for(main)
        tolerance = 1e-5 * scale
        before, tool_volume = main.Volume(), cutter.Volume()
        feature = eg_pir.hollow_features[len(cuts)] if eg_pir else None
        measurement = {"cut_index": len(cuts) + 1, "hollow_id": feature.hollow_id if feature else None,
                       "volume_before": before, "tool_volume": tool_volume, "fuzzy_tolerance": tolerance}
        cuts.append(measurement)
        measurement["healthy"] = False
        if math.isfinite(tool_volume) and tool_volume > 0:
            outcome = boolean(BRepAlgoAPI_Cut, main, cutter, tolerance, unify=True)
            after = health(outcome)
            measurement.update(after)
            drop = before - after["volume"]
            measurement["volume_removed"] = drop
            overshoot_ok = True
            if feature:
                pose = feature.spatial_pose
                axis = tool.plane.zDir
                cap_tolerance = min(tolerance, 1e-5)
                ends = _cap_clearances(main, cutter, axis, cap_tolerance)
                required = len(pose.overshoot_ends)
                qualified = sum(end is not None and end >= 10.0 - cap_tolerance for end in ends)
                overshoot_ok = qualified >= required
                if pose.through_type == "blind_internal":
                    overshoot_ok = overshoot_ok and abs(tool_volume - drop) <= max(
                        1e-8, tolerance * scale**2
                    )
                measurement["axial_overshoot"] = ends
                measurement["overshoot_ok"] = overshoot_ok
            measurement["healthy"] = (after["is_watertight"] and overshoot_ok and math.isfinite(after["volume"])
                                      and 0 < after["volume"] < before
                                      and drop > max(1e-9, abs(before) * 1e-10))
            if measurement["healthy"]:
                return cq.Workplane(obj=outcome)
        raise ValueError(f"cut {len(cuts)} failed geometry health")

    def checked_union(current, tool):
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common

        main, part = one_shape(current), one_shape(tool)
        tolerance = 1e-5 * scale_for(main)
        overlap = boolean(BRepAlgoAPI_Common, main, part, tolerance)
        if overlap.Volume() <= 0 or len(overlap.Solids()) != 1:
            raise ValueError("embedded subassembly has no solid overlap")
        box = overlap.BoundingBox()
        if max(box.xlen, box.ylen, box.zlen) < 10.0 - tolerance:
            raise ValueError("embedded subassembly needs at least 10 mm penetration")
        # A translated attachment must still overlap along its dominant separation axis.
        separation = part.Center() - main.Center()
        axis = max(range(3), key=lambda i: abs(separation.toTuple()[i]))
        if (box.xlen, box.ylen, box.zlen)[axis] < 10.0 - tolerance:
            raise ValueError("embedded subassembly is too shallow along the attachment axis")
        fused = boolean(BRepAlgoAPI_Fuse, main, part, tolerance, unify=True)
        if not health(fused)["is_watertight"] or fused.Volume() <= main.Volume():
            raise ValueError("embedded subassembly must extend a single watertight main solid")
        unions.append({"volume_added": fused.Volume() - main.Volume()})
        return cq.Workplane(obj=fused)

    def only_cadquery(name, *args):
        if name != "cadquery":
            raise ImportError(name)
        return cq

    def checked_envelope(workplane):
        shape = one_shape(workplane)
        envelope = health(shape)
        if not envelope["is_watertight"] or not math.isfinite(envelope["volume"]) or envelope["volume"] <= 0:
            raise ValueError("Stage I must produce one watertight positive-volume envelope")
        if any(len(face.Wires()) > 1 for face in shape.Faces()):
            raise ValueError("Stage I contains a hollow profile; form the opening with a Stage II cut")
        return workplane

    tree = Instrument().visit(validate_cadquery(source, eg_pir))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"extrude", "revolve", "loft"} for node in ast.walk(statement.value)
        ):
            expression = statement.value
            statement.value = ast.copy_location(
                ast.Call(ast.Name("__checked_envelope", ast.Load()), [expression], []), expression
            )
            break
    scope = {"cq": cq, "checked_cut": checked_cut, "checked_union": checked_union,
             "__checked_envelope": checked_envelope, "__builtins__": {"__import__": only_cadquery}}
    try:
        exec(compile(ast.fix_missing_locations(tree), "<etc-cq>", "exec"), scope)
        result = one_shape(scope["result"])
        geometry = health(result)
        if not geometry["is_watertight"] or not math.isfinite(geometry["volume"]) or geometry["volume"] <= 0:
            raise ValueError("result must be one valid water-tight positive-volume solid")
    except Exception as exc:
        return None, {"ok": False, "status": "invalid_geometry", "detail": f"{type(exc).__name__}: {exc}",
                      "cut_steps": cuts, "R_cut_health": 0.0}
    return result, {"ok": True, "status": "ok", "geometry": geometry, "cut_steps": cuts,
                    "subassembly_unions": unions, "R_cut_health": 1.0}


def evaluate(payload):
    from .eg_pir_schema import parse_etc_cq
    from .hlr_projector_ocp import compare_views, project_orthographic
    from .solid_iou_ocp import compare_solids, scale_for

    if "source" in payload:
        prediction, report = execute(payload["source"])
    else:
        eg_pir, source = parse_etc_cq(payload["prediction"])
        prediction, report = execute(source, eg_pir)
    if prediction is None:
        return report
    scale = scale_for(prediction)
    reference = None
    raw_reference = payload.get("reference")
    if raw_reference and payload.get("reference_step"):
        import cadquery as cq
        imported = cq.importers.importStep(raw_reference).solids().vals()
        if len(imported) != 1 or not imported[0].isValid() or imported[0].Volume() <= 0:
            raise ValueError("reference STEP must contain a single valid positive-volume solid")
        reference = imported[0]
    elif raw_reference:
        eg_pir, source = parse_etc_cq(raw_reference)
        reference, reference_report = execute(source, eg_pir)
        if reference is None:
            return {**report, "ok": False, "status": "invalid_reference", "reference_report": reference_report}

    if reference is not None:
        scale = max(scale, scale_for(reference))
    deflection = scale / 2000
    if payload.get("project", True) or reference is not None:
        prediction_views = project_orthographic(prediction.wrapped, deflection)
        if payload.get("project", True):
            report["views"] = prediction_views
    if reference is not None:
        reference_views = project_orthographic(reference.wrapped, deflection)
        report.update(compare_views(prediction_views, reference_views, scale))
        report.update(compare_solids(prediction, reference, scale, payload.get("align", True),
                                     reference_views, report["2D-Reproj-IoU"]))
    return report


def _respond(payload):
    try:
        if payload == {"ping": True}:
            import cadquery
            from . import solid_iou_ocp
            return {"ok": True, "status": "ready"}
        return evaluate(payload)
    except Exception as exc:
        return {"ok": False, "status": "evaluation_error", "detail": f"{type(exc).__name__}: {exc}"}


def main():
    if sys.argv[1:] == ["--persistent"]:
        for line in sys.stdin:
            try:
                response = _respond(json.loads(line))
                print(json.dumps(response, allow_nan=False), flush=True)
            except (ValueError, TypeError) as exc:
                print(json.dumps({"ok": False, "status": "invalid_ipc", "detail": str(exc)}), flush=True)
    else:
        print(json.dumps(_respond(json.load(sys.stdin)), allow_nan=False))


if __name__ == "__main__":
    main()
