"""Inputs and outputs of the packer, and the validation of inputs.

Placement contract, relied on everywhere: a placed copy is the part's own vertices transformed by
``p' = Rz(angle) · p + (x, y, 0)`` in plate coordinates — origin at the plate's front-left corner,
X to the right, Y to the back, millimetres, angle in degrees counter-clockwise seen from above.
Z is never changed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Polygon = Sequence[tuple[float, float]]

MAX_GRID_CELLS = 50_000_000  # plate pixels; every plate holds a few arrays of this size


@dataclass(frozen=True, eq=False)
class Part:
    vertices: np.ndarray  # (N, 3) float, mm
    triangles: np.ndarray  # (M, 3) int, indices into vertices
    quantity: int = 1  # ignored when fill is True
    fill: bool = False  # own plate, as many copies as fit
    name: str = ""


@dataclass(frozen=True)
class Zone:
    width: float  # plate size along X
    depth: float  # plate size along Y
    allowed: Sequence[Polygon] = ()  # union = where parts may be; empty = the whole plate
    excluded: Sequence[Polygon] = ()  # subtracted from the allowed area
    max_height: float | None = None


@dataclass(frozen=True)
class Options:
    spacing: float = 2.0
    rotations: Sequence[float] = (0.0, 90.0, 180.0, 270.0)
    centre: bool = True
    resolution: float = 0.5  # mm per pixel
    fill_limit: int = 1000


@dataclass(frozen=True)
class Placement:
    part: int  # index into the parts argument
    plate: int  # 0-based
    x: float
    y: float
    angle: float  # one of Options.rotations


@dataclass(frozen=True)
class Rejected:
    part: int
    reason: Literal["too_large", "too_tall", "empty_mesh"]


@dataclass(frozen=True)
class PackResult:
    plate_count: int
    placements: tuple[Placement, ...]
    rejected: tuple[Rejected, ...] = ()
    fill_counts: Mapping[int, int] = field(default_factory=dict)  # part index -> copies placed (fill parts)
    fill_limited: frozenset[int] = frozenset()  # fill parts that stopped at Options.fill_limit


def _finite_positive(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _check_polygons(polygons: Sequence[Polygon], what: str) -> None:
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError(f"zone: every {what} polygon needs at least three finite (x, y) points")


def validate(parts: Sequence[Part], zone: Zone, options: Options) -> None:
    """Raise ``ValueError`` for input the packer cannot work with. Per-part problems name the part index."""
    if not _finite_positive(zone.width) or not _finite_positive(zone.depth):
        raise ValueError("zone: width and depth must be positive finite numbers")
    if zone.max_height is not None and not _finite_positive(zone.max_height):
        raise ValueError("zone: max_height must be a positive finite number or None")
    _check_polygons(zone.allowed, "allowed")
    _check_polygons(zone.excluded, "excluded")

    if not isinstance(options.spacing, (int, float)) or not math.isfinite(options.spacing) or options.spacing < 0:
        raise ValueError("options: spacing must be a finite number >= 0")
    if not _finite_positive(options.resolution):
        raise ValueError("options: resolution must be a positive finite number")
    cells = math.ceil(zone.width / options.resolution) * math.ceil(zone.depth / options.resolution)
    if cells > MAX_GRID_CELLS:
        raise ValueError(
            f"options: resolution {options.resolution} is too fine for a {zone.width} x {zone.depth} zone "
            f"({cells} cells, the limit is {MAX_GRID_CELLS})"
        )
    if len(options.rotations) == 0 or not all(math.isfinite(a) for a in options.rotations):
        raise ValueError("options: rotations must hold at least one finite angle")
    if not isinstance(options.fill_limit, int) or options.fill_limit < 1:
        raise ValueError("options: fill_limit must be an integer >= 1")

    for index, part in enumerate(parts):
        vertices, triangles = np.asarray(part.vertices), np.asarray(part.triangles)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError(f"part {index}: vertices must be a non-empty (N, 3) array")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
            raise ValueError(f"part {index}: triangles must be a non-empty (M, 3) array")
        if not np.issubdtype(vertices.dtype, np.number) or not np.isfinite(vertices.astype(np.float64)).all():
            raise ValueError(f"part {index}: vertices contain a non-finite coordinate")
        if not np.issubdtype(triangles.dtype, np.integer):
            raise ValueError(f"part {index}: triangle indices must be integers")
        if triangles.min() < 0 or triangles.max() >= len(vertices):
            raise ValueError(f"part {index}: a triangle index is outside the vertex array")
        if not part.fill and (not isinstance(part.quantity, int) or part.quantity < 1):
            raise ValueError(f"part {index}: quantity must be an integer >= 1")
