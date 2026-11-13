"""A top-view picture of one packed plate."""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from platenest.types import Options, PackResult, Part, Zone
from platenest.zone import _SHIFT, allowed_mask, to_cv

_FREE = (236, 236, 236)
_BLOCKED = (72, 72, 72)
_COLOURS = (  # BGR
    (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148), (75, 86, 140),
    (194, 119, 227), (34, 189, 188), (207, 190, 23), (127, 127, 127), (120, 187, 255), (138, 223, 152),
)  # fmt: skip


def render_plate(
    result: PackResult,
    parts: Sequence[Part],
    zone: Zone,
    options: Options,
    plate: int,
    *,
    scale: int = 2,
) -> np.ndarray:
    """``uint8 (h·scale, w·scale, 3)`` BGR image of ``plate``; the plate front is at the bottom.

    Areas where parts may not be are dark, free area is light, each part index has its own colour
    (twelve colours, then they repeat). Drawn from the placements, by the placement contract.
    """
    if not 0 <= plate < result.plate_count:
        raise IndexError(f"plate {plate} is not one of the {result.plate_count} packed plates")
    allowed = allowed_mask(zone, options)
    height, width = allowed.shape
    big = cv2.resize(allowed.astype(np.uint8), (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
    image = np.where(big[:, :, None] > 0, np.array(_FREE, np.uint8), np.array(_BLOCKED, np.uint8))
    image = np.ascontiguousarray(image, dtype=np.uint8)

    step = options.resolution / scale
    for placement in result.placements:
        if placement.plate != plate:
            continue
        part = parts[placement.part]
        xy = np.asarray(part.vertices, dtype=np.float64)[:, :2]
        a = math.radians(placement.angle)
        c, s = math.cos(a), math.sin(a)
        placed = np.column_stack((c * xy[:, 0] - s * xy[:, 1] + placement.x, s * xy[:, 0] + c * xy[:, 1] + placement.y))
        points = to_cv(placed, step)
        colour = _COLOURS[placement.part % len(_COLOURS)]
        for tri in points[np.asarray(part.triangles, dtype=np.int64)]:
            cv2.fillConvexPoly(image, tri, colour, shift=_SHIFT)
    return image[::-1].copy()
