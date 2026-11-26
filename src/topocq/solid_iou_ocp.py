"""OCP BRep IoU, with observable recovery paths for coplanar failures."""

from itertools import product

import cadquery as cq
import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.gp import gp_Trsf

from .hlr_projector_ocp import chamfer_distance, compare_views, project_orthographic


def scale_for(shape):
    box = shape.BoundingBox()
    return max(box.xlen, box.ylen, box.zlen, 1e-3)


def boolean(algorithm, first, second, tolerance=0.0, *, unify=False):
    operation = algorithm()
    from OCP.TopTools import TopTools_ListOfShape
    args, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
    args.Append(first.wrapped)
    tools.Append(second.wrapped)
    operation.SetArguments(args)
    operation.SetTools(tools)
    operation.SetNonDestructive(True)
    if tolerance:
        operation.SetFuzzyValue(tolerance)
    operation.Build()
    if not operation.IsDone():
        raise RuntimeError(f"{algorithm.__name__} failed")
    shape = operation.Shape()
    if shape.IsNull():
        raise RuntimeError(f"{algorithm.__name__} returned a null shape")
    if unify:
        cleaner = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
        cleaner.SetLinearTolerance(tolerance)
        cleaner.Build()
        shape = cleaner.Shape()
        if shape.IsNull():
            raise RuntimeError("SameDomain returned a null shape")
    return cq.Shape.cast(shape)


def _principal_poses(prediction, reference):
    def axes(solid):
        _, base = np.linalg.eigh(np.asarray(cq.Shape.matrixOfInertia(solid)))
        if np.linalg.det(base) < 0:
            base[:, -1] *= -1
        return base, np.asarray(solid.Center().toTuple())

    left, left_center = axes(prediction)
    right, right_center = axes(reference)
    for signs in product((1, -1), repeat=3):
        if np.prod(signs) != 1:
            continue
        rotation = (right * signs) @ left.T
        offset = right_center - rotation @ left_center
        transform = gp_Trsf()
        transform.SetValues(*np.column_stack((rotation, offset)).ravel())
        yield prediction.moved(cq.Location(transform))


def _brep_iou(prediction, reference, scale, tolerance=0.0):
    overlap = boolean(BRepAlgoAPI_Common, prediction, reference, tolerance)
    shared = sum(solid.Volume() for solid in overlap.Solids())
    a, b = prediction.Volume(), reference.Volume()
    error = max(a, b) * 1e-7 + tolerance * scale**2
    if not np.isfinite(shared) or shared < -error or shared > min(a, b) + error:
        raise RuntimeError(f"boolean intersection volume is inconsistent: {shared}")
    if shared > 0 and not overlap.isValid():
        raise RuntimeError("boolean intersection returned an invalid solid")
    shared = min(max(shared, 0.0), a, b)
    return shared / (a + b - shared)


def _mesh(shape, scale):
    vertices, triangles = shape.tessellate(scale / 1500)
    if not triangles:
        raise RuntimeError("OCCT tessellation returned no triangles")
    return np.asarray([point.toTuple() for point in vertices]), np.asarray(triangles)


def _voxelize(mesh, lower, pitch, resolution):
    """Parity along X from tessellated triangles on a shared cell-center grid."""
    vertices, indices = mesh
    # Small fixed sub-cell offsets avoid rays on shared triangle edges.
    offsets = np.array([0.5, 0.500113, 0.500227])
    triangles = (vertices[indices] - lower) / pitch - offsets
    crossings = np.zeros((resolution + 1, resolution, resolution), dtype=bool)
    for a, b, c in triangles:
        dy, dz = b[1:] - a[1:]
        ey, ez = c[1:] - a[1:]
        det = dy * ez - dz * ey
        if abs(det) < 1e-12:
            continue
        start = np.maximum(0, np.ceil(np.min([a[1:], b[1:], c[1:]], axis=0)).astype(int))
        stop = np.minimum(resolution - 1, np.floor(np.max([a[1:], b[1:], c[1:]], axis=0)).astype(int))
        if np.any(stop < start):
            continue
        y, z = np.meshgrid(np.arange(start[0], stop[0] + 1), np.arange(start[1], stop[1] + 1), indexing="ij")
        u = ((y - a[1]) * ez - (z - a[2]) * ey) / det
        v = (dy * (z - a[2]) - dz * (y - a[1])) / det
        inside = (u >= 0) & (v >= 0) & (u + v < 1)
        x = a[0] + u[inside] * (b[0] - a[0]) + v[inside] * (c[0] - a[0])
        crossings[np.clip(np.ceil(x).astype(int), 0, resolution), y[inside], z[inside]] ^= True
    return np.logical_xor.accumulate(crossings[:-1], axis=0)


def _surface_points(mesh):
    vertices, indices = mesh
    a, b, c = vertices[indices].transpose(1, 0, 2)
    weights = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(34)
    chosen = rng.choice(len(weights), 1536, p=weights / weights.sum())
    u, v = rng.random((2, len(chosen)))
    u, v = np.where(u + v > 1, 1 - u, u), np.where(u + v > 1, 1 - v, v)
    return a[chosen] + u[:, None] * (b - a)[chosen] + v[:, None] * (c - a)[chosen]


def _voxel_compare(prediction, reference, scale):
    meshes = [_mesh(shape, scale) for shape in (prediction, reference)]
    lower = np.minimum(*(vertices.min(axis=0) for vertices, _ in meshes))
    upper = np.maximum(*(vertices.max(axis=0) for vertices, _ in meshes))
    length = max(upper - lower)
    scores = {}
    for n in (64, 128):
        pitch = length / (n - 2)
        first, second = [_voxelize(mesh, lower - pitch, pitch, n) for mesh in meshes]
        shared = np.count_nonzero(first & second)
        total = np.count_nonzero(first | second)
        scores[str(n)] = shared / total if total else None
    if scores["128"] is None:
        raise RuntimeError("voxel grid cannot resolve either solid")
    return {"iou": scores["128"], "voxel_iou": scores,
            "Chamfer3D": chamfer_distance(*[_surface_points(mesh) for mesh in meshes])}


def compare_solids(prediction, reference, scale, align, reference_views, unaligned_view_iou):
    candidates = _principal_poses(prediction, reference) if align else (prediction,)
    results = []
    attempts = []

    def try_iou(pose, method, shape, tolerance=0.0):
        try:
            value = _brep_iou(shape, reference, scale, tolerance)
        except Exception as exc:
            attempts.append({"pose": pose, "method": method, "error": f"{type(exc).__name__}: {exc}"})
            return None
        attempts.append({"pose": pose, "method": method, "iou": value})
        return value

    for pose, shape in enumerate(candidates):
        score = try_iou(pose, "brep", shape)
        if score == 0:
            view_iou = unaligned_view_iou
            if align:
                views = project_orthographic(shape.wrapped, scale / 2000)
                view_iou = compare_views(views, reference_views, scale)["2D-Reproj-IoU"]
            if view_iou is not None and view_iou > 0.90:
                attempts[-1]["coplanar_signature"] = True
                score = None
        if score is not None:
            results.append({"iou": score, "iou_method": "brep", "pose": pose})
            continue
        score = try_iou(pose, "fuzzy", shape, 1e-5 * scale)
        if score:
            results.append({"iou": score, "iou_method": "fuzzy", "pose": pose})
            continue
        shift = cq.Vector(1, -0.71, 0.43).multiply(scale * 1e-5)
        perturbations = [try_iou(pose, "micro_perturbation", shape.translate(vector))
                         for vector in (shift, shift.multiply(-1))]
        score = max((value for value in perturbations if value is not None), default=0.0)
        if score:
            results.append({"iou": score, "iou_method": "micro_perturbation", "pose": pose})
            continue
        try:
            results.append({**_voxel_compare(shape, reference, scale), "iou_method": "voxel", "pose": pose})
        except Exception as exc:
            attempts.append({"pose": pose, "method": "voxel", "error": f"{type(exc).__name__}: {exc}"})
    if not results:
        return {"ok": False, "status": "iou_error", "iou": None, "iou_attempts": attempts}
    return {**max(results, key=lambda result: result["iou"]), "iou_attempts": attempts,
            "aligned": align, "poses_scored": len(results)}
