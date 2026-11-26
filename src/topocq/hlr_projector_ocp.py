"""Orthographic visible and hidden line projection inside CAD_PYTHON."""

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GCPnts import GCPnts_QuasiUniformDeflection
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from shapely import LineString, union_all


VIEWS = {
    "front": ((0, -1, 0), (1, 0, 0)),
    "top": ((0, 0, 1), (1, 0, 0)),
    "side": ((1, 0, 0), (0, 1, 0)),
}


def _polylines(projected, deflection):
    if projected.IsNull():
        return []
    explorer = TopExp_Explorer(projected, TopAbs_EDGE)
    wires = []
    while explorer.More():
        curve = BRepAdaptor_Curve(TopoDS.Edge_s(explorer.Current()))
        samples = GCPnts_QuasiUniformDeflection(curve, deflection)
        if not samples.IsDone() or samples.NbPoints() < 2:
            raise RuntimeError("OCCT could not discretize an HLR edge")
        coords = [(samples.Value(i).X(), samples.Value(i).Y()) for i in range(1, samples.NbPoints() + 1)]
        if len(set(coords)) > 1:
            wires.append(coords)
        explorer.Next()
    return wires


def project_orthographic(shape, deflection):
    views = {}
    for view, (normal, horizontal) in VIEWS.items():
        algo = HLRBRep_Algo()
        algo.Add(shape)
        algo.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(), gp_Dir(*normal), gp_Dir(*horizontal))))
        algo.Update()
        algo.Hide()
        edges = HLRBRep_HLRToShape(algo)
        views[view] = {
            "visible": _polylines(edges.VCompound(), deflection) + _polylines(edges.OutLineVCompound(), deflection),
            "hidden": _polylines(edges.HCompound(), deflection) + _polylines(edges.OutLineHCompound(), deflection),
        }
    return views


def chamfer_distance(left, right):
    """Mean of the two directed mean nearest-neighbor distances."""
    def directed(a, b):
        return sum(np.sqrt(((batch[:, None] - b) ** 2).sum(axis=2).min(axis=1)).sum()
                   for batch in np.array_split(a, max(1, (len(a) + 127) // 128))) / len(a)
    return float((directed(left, right) + directed(right, left)) / 2)


def _sample(lines, pitch):
    if not lines:
        return np.empty((0, 2))
    return np.concatenate([
        np.linspace(start, end, max(2, int(np.ceil(np.linalg.norm(end - start) / pitch)) + 1))
        for line in lines for start, end in zip(np.asarray(line)[:-1], np.asarray(line)[1:])
    ])


def compare_views(prediction, reference, scale):
    """Equal-weight views, per-line-type buffered IoU, and Chamfer in model units."""
    stroke_radius = scale / 512
    pitch = scale / 150
    by_view = {}
    for view in VIEWS:
        channels = {}
        for channel in ("visible", "hidden"):
            left = prediction[view][channel]
            right = reference[view][channel]
            areas = [union_all([LineString(line).buffer(stroke_radius, quad_segs=4) for line in lines])
                     for lines in (left, right)]
            union = areas[0].union(areas[1]).area
            iou = areas[0].intersection(areas[1]).area / union if union else None
            distance = chamfer_distance(_sample(left, pitch), _sample(right, pitch)) if left and right else None
            channels[channel] = {"iou": iou, "chamfer": distance, "line_type_mismatch": bool(left) != bool(right)}
        by_view[view] = channels
    view_ious = [np.mean([item["iou"] for item in channels.values() if item["iou"] is not None])
                 for channels in by_view.values() if any(item["iou"] is not None for item in channels.values())]
    metrics = {"2D-Reproj-IoU": float(np.mean(view_ious)) if view_ious else None,
               "stroke_radius": stroke_radius, "by_view": by_view}
    for channel, label in (("visible", "Vis-CD2D"), ("hidden", "Hid-CD2D")):
        distances = [item[channel]["chamfer"] for item in by_view.values() if item[channel]["chamfer"] is not None]
        metrics[label] = float(np.mean(distances)) if distances else None
    return metrics
