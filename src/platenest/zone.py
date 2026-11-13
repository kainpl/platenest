"""The work zone as a pixel mask.

Pixel ``(row, col)`` covers ``x in [col·res, (col+1)·res)``, ``y in [row·res, (row+1)·res)``;
row 0 is the plate front. OpenCV puts integer coordinates at pixel centres, so a millimetre
coordinate ``m`` becomes ``m / res - 0.5`` before drawing.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from platenest.types import Options, Polygon, Zone

_SHIFT = 4  # sub-pixel bits for OpenCV drawing
_EPS = 1e-6


def grid_size(zone: Zone, resolution: float) -> tuple[int, int]:
    """``(width_px, height_px)`` of the plate grid."""
    return math.ceil(zone.width / resolution - _EPS), math.ceil(zone.depth / resolution - _EPS)


def to_cv(points_mm: np.ndarray, resolution: float) -> np.ndarray:
    """Millimetre XY -> OpenCV fixed-point coordinates (pixel centres at integers)."""
    return np.round((points_mm / resolution - 0.5) * (1 << _SHIFT)).astype(np.int32)


def _fill(mask: np.ndarray, polygon: Polygon, resolution: float, *, touch: bool) -> None:
    points = to_cv(np.asarray(polygon, dtype=np.float64), resolution).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [points], 1, shift=_SHIFT)
    if touch:  # conservative: every pixel the outline passes through counts
        cv2.polylines(mask, [points], True, 1, thickness=1, shift=_SHIFT)


def allowed_mask(zone: Zone, options: Options) -> np.ndarray:
    """Bool ``(h, w)``: True where a part's pixel may be.

    The allowed area (union of ``zone.allowed``, or the whole plate) minus ``zone.excluded``,
    shrunk by half the spacing so that brims stay inside; the plate border counts as not allowed.
    """
    width, height = grid_size(zone, options.resolution)
    if zone.allowed:
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in zone.allowed:
            _fill(mask, polygon, options.resolution, touch=False)
    else:
        mask = np.ones((height, width), dtype=np.uint8)
    if zone.excluded:
        excluded = np.zeros((height, width), dtype=np.uint8)
        for polygon in zone.excluded:
            _fill(excluded, polygon, options.resolution, touch=True)
        mask[excluded > 0] = 0

    margin = math.ceil(options.spacing / 2 / options.resolution - _EPS)
    if margin > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        mask = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask.astype(bool)
