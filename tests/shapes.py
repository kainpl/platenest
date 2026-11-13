"""Synthetic meshes and measuring helpers shared by the tests.

Every builder returns ``(vertices, triangles)``: float (N, 3) millimetres and int (M, 3) indices.
The meshes are unions of extruded pieces — good enough for a top-down footprint, not watertight.
"""

from __future__ import annotations

import math

import numpy as np

_BOX_TRIANGLES = np.array(
    [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ],
    dtype=np.int64,
)  # fmt: skip


def _prism(corners: list[tuple[float, float]], h: float) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a convex quad given by four XY corners."""
    bottom = [(x, y, 0.0) for x, y in corners]
    top = [(x, y, h) for x, y in corners]
    return np.array(bottom + top, dtype=np.float64), _BOX_TRIANGLES.copy()


def _merge(pieces: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    vertices, triangles, offset = [], [], 0
    for v, t in pieces:
        vertices.append(v)
        triangles.append(t + offset)
        offset += len(v)
    return np.vstack(vertices), np.vstack(triangles)


def _rect(x0: float, y0: float, x1: float, y1: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    return _prism([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], h)


def box(w: float, d: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    return _rect(0, 0, w, d, h)


def frame(w: float, d: float, wall: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """A rectangular ring: outer w x d, walls ``wall`` thick."""
    return _merge(
        [
            _rect(0, 0, w, wall, h),
            _rect(0, d - wall, w, d, h),
            _rect(0, wall, wall, d - wall, h),
            _rect(w - wall, wall, w, d - wall, h),
        ]
    )


def u_shape(w: float, d: float, wall: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """A U open towards +Y."""
    return _merge([_rect(0, 0, w, wall, h), _rect(0, wall, wall, d, h), _rect(w - wall, wall, w, d, h)])


def ring(outer_d: float, inner_d: float, h: float, segments: int = 64) -> tuple[np.ndarray, np.ndarray]:
    ro, ri, c = outer_d / 2, inner_d / 2, outer_d / 2
    pieces = []
    for k in range(segments):
        a0, a1 = 2 * math.pi * k / segments, 2 * math.pi * (k + 1) / segments
        pieces.append(
            _prism(
                [
                    (c + ri * math.cos(a0), c + ri * math.sin(a0)),
                    (c + ro * math.cos(a0), c + ro * math.sin(a0)),
                    (c + ro * math.cos(a1), c + ro * math.sin(a1)),
                    (c + ri * math.cos(a1), c + ri * math.sin(a1)),
                ],
                h,
            )
        )
    return _merge(pieces)


def sliver(h: float) -> tuple[np.ndarray, np.ndarray]:
    """A vertical triangle: its top-down footprint has zero area."""
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, h], [0.0, 0.0, h / 2]]), np.array([[0, 1, 2]], dtype=np.int64)


# --- measuring helpers (exact, via shapely; test-only) ---------------------------------------


def transformed(vertices: np.ndarray, placement) -> np.ndarray:
    """Apply the placement contract: ``p' = Rz(angle) · p + (x, y, 0)``."""
    a = math.radians(placement.angle)
    c, s = math.cos(a), math.sin(a)
    out = np.array(vertices, dtype=np.float64, copy=True)
    x, y = vertices[:, 0], vertices[:, 1]
    out[:, 0] = c * x - s * y + placement.x
    out[:, 1] = s * x + c * y + placement.y
    return out


def _outline(vertices: np.ndarray, triangles: np.ndarray):
    """The top-down footprint as an exact polygon."""
    from shapely import Polygon, union_all

    xy = vertices[:, :2][triangles]
    polygons = [Polygon(t) for t in xy]
    return union_all([p for p in polygons if p.area > 0])


def placed_outlines(result, parts) -> dict[int, list]:
    """Plate index -> footprints of every copy on it, in plate coordinates."""
    from shapely import affinity

    base = {}
    plates: dict[int, list] = {}
    for p in result.placements:
        if p.part not in base:
            base[p.part] = _outline(np.asarray(parts[p.part].vertices), np.asarray(parts[p.part].triangles))
        shape = affinity.translate(affinity.rotate(base[p.part], p.angle, origin=(0, 0)), p.x, p.y)
        plates.setdefault(p.plate, []).append(shape)
    return plates


def min_gap(result, parts) -> float:
    """Smallest distance between two different copies on one plate; ``inf`` when no plate holds two."""
    best = math.inf
    for shapes in placed_outlines(result, parts).values():
        for i, a in enumerate(shapes):
            for b in shapes[i + 1 :]:
                best = min(best, a.distance(b))
    return best


def overlap_area(result, parts) -> float:
    total = 0.0
    for shapes in placed_outlines(result, parts).values():
        for i, a in enumerate(shapes):
            for b in shapes[i + 1 :]:
                total += a.intersection(b).area
    return round(total, 6)


def transformed_bounds(result, parts, index: int = 0) -> tuple[float, float, float, float]:
    """``(x0, y0, x1, y1)`` of placement ``index``."""
    p = result.placements[index]
    v = transformed(np.asarray(parts[p.part].vertices), p)
    return float(v[:, 0].min()), float(v[:, 1].min()), float(v[:, 0].max()), float(v[:, 1].max())
