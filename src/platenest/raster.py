"""A part's top-down footprint as a bitmap, one per allowed rotation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from platenest.types import Options, Part
from platenest.zone import _EPS, _SHIFT, to_cv

_MIN_AREA = 1e-6  # mm²: below this the projection is a line or a point


@dataclass(frozen=True, eq=False)
class Shape:
    angle: float
    raw: np.ndarray  # bool (h, w): the footprint, tight bounding box
    grown: np.ndarray  # bool (h + 2g, w + 2g): raw dilated by g pixels
    grow: int  # g
    origin: tuple[float, float]  # mm: where the part's own (0, 0) lies relative to raw's pixel (0, 0) corner


def grow_pixels(spacing: float, resolution: float) -> int:
    """Dilation radius in pixels. The extra pixel keeps the real gap from falling below ``spacing``."""
    if spacing <= 0:
        return 0
    return math.ceil(spacing / resolution - _EPS) + 1


def _rotated_xy(vertices: np.ndarray, angle: float) -> np.ndarray:
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    # exact values for the right angles, so that equal footprints compare equal
    c, s = (round(c), round(s)) if abs(abs(c) - round(abs(c))) < 1e-12 else (c, s)
    xy = vertices[:, :2]
    return np.column_stack((c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]))


def _footprint(xy: np.ndarray, triangles: np.ndarray, resolution: float) -> tuple[np.ndarray, tuple[float, float]]:
    low = xy.min(axis=0)
    size = xy.max(axis=0) - low
    width = max(1, math.ceil(size[0] / resolution - _EPS))
    height = max(1, math.ceil(size[1] / resolution - _EPS))
    points = to_cv(xy - low, resolution)
    canvas = np.zeros((height, width), dtype=np.uint8)
    # One triangle at a time: a single even-odd fill of all of them lets overlapping faces cancel.
    for tri in points[triangles]:
        cv2.fillConvexPoly(canvas, tri, 1, shift=_SHIFT)
        cv2.polylines(canvas, [tri.reshape(-1, 1, 2)], True, 1, thickness=1, shift=_SHIFT)
    return canvas.astype(bool), (float(-low[0]), float(-low[1]))


def footprints(part: Part, options: Options) -> tuple[Shape, ...]:
    """One :class:`Shape` per distinct rotation; ``()`` when the part has no top-down area."""
    vertices = np.asarray(part.vertices, dtype=np.float64)
    triangles = np.asarray(part.triangles, dtype=np.int64)

    corners = vertices[:, :2][triangles]  # (M, 3, 2)
    edge1, edge2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    areas = np.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]) / 2
    if areas.sum() <= _MIN_AREA:
        return ()
    flat = triangles[areas > 0]  # vertical faces add nothing to the projection

    grow = grow_pixels(options.spacing, options.resolution)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)) if grow else None
    shapes: list[Shape] = []
    for angle in options.rotations:
        raw, origin = _footprint(_rotated_xy(vertices, angle), flat, options.resolution)
        if not raw.any():
            continue
        if any(raw.shape == other.raw.shape and np.array_equal(raw, other.raw) for other in shapes):
            continue
        if grow:
            padded = np.pad(raw.astype(np.uint8), grow)
            grown = cv2.dilate(padded, kernel).astype(bool)
        else:
            grown = raw
        shapes.append(Shape(float(angle), raw, grown, grow, origin))
    return tuple(shapes)
