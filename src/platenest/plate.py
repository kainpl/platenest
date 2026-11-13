"""One plate: what is occupied, and where a shape still fits.

A position is free for a shape when its GROWN mask touches no pixel of a part already placed
and its RAW mask touches no blocked pixel. Only the incoming shape is dilated, so two parts end
up one full spacing apart, not two. Both conditions are evaluated for every position at once by
correlating the mask with the plate (``cv2.matchTemplate``), then the chosen position is checked
exactly — the correlation is floating-point and must not be the last word on an overlap.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from platenest.raster import Shape

Gravity = Literal["y", "x", "xy"]
GRAVITIES: tuple[Gravity, ...] = ("y", "x", "xy")


class Plate:
    def __init__(self, allowed: np.ndarray) -> None:
        self._height, self._width = allowed.shape
        self._blocked = ~allowed
        self._blocked_f = self._blocked.astype(np.float32)
        self._occupied = np.zeros(allowed.shape, dtype=bool)
        self._bounds: tuple[int, int, int, int] | None = None

    @property
    def empty(self) -> bool:
        return self._bounds is None

    @property
    def size(self) -> tuple[int, int]:
        """``(width_px, height_px)``."""
        return self._width, self._height

    def bounds(self) -> tuple[int, int, int, int]:
        """``(col0, row0, col1, row1)`` of everything placed, end exclusive; zeros for an empty plate."""
        return self._bounds or (0, 0, 0, 0)

    def _free(self, shape: Shape) -> np.ndarray | None:
        height, width = shape.raw.shape
        if height > self._height or width > self._width:
            return None
        clear_of_blocked = cv2.matchTemplate(self._blocked_f, shape.raw.astype(np.float32), cv2.TM_CCORR) < 0.5
        if self.empty:
            return clear_of_blocked
        occupied = np.pad(self._occupied, shape.grow).astype(np.float32)
        clear_of_parts = cv2.matchTemplate(occupied, shape.grown.astype(np.float32), cv2.TM_CCORR) < 0.5
        return clear_of_blocked & clear_of_parts

    def fits(self, shape: Shape, col: int, row: int) -> bool:
        """Exact check of one position."""
        height, width = shape.raw.shape
        if col < 0 or row < 0 or row + height > self._height or col + width > self._width:
            return False
        if (self._blocked[row : row + height, col : col + width] & shape.raw).any():
            return False
        g = shape.grow
        r0, c0 = max(0, row - g), max(0, col - g)
        r1, c1 = min(self._height, row + height + g), min(self._width, col + width + g)
        grown = shape.grown[r0 - (row - g) : r1 - (row - g), c0 - (col - g) : c1 - (col - g)]
        return not (self._occupied[r0:r1, c0:c1] & grown).any()

    def best(self, shape: Shape, gravity: Gravity) -> tuple[int, int] | None:
        """``(col, row)`` for the shape's raw corner with the smallest gravity score, or None."""
        free = self._free(shape)
        while free is not None and free.any():
            if gravity == "y":  # row·W + col: the first free cell in row-major order
                row, col = np.unravel_index(int(free.argmax()), free.shape)
            elif gravity == "x":  # col·H + row: the first free cell in column-major order
                col, row = np.unravel_index(int(free.T.argmax()), free.T.shape)
            else:  # row + col, then row
                rows, cols = np.nonzero(free)
                k = int(np.lexsort((rows, rows + cols))[0])
                row, col = rows[k], cols[k]
            col, row = int(col), int(row)
            if self.fits(shape, col, row):
                return col, row
            free[row, col] = False  # a rounding artefact of the correlation; look further
        return None

    def put(self, shape: Shape, col: int, row: int) -> None:
        height, width = shape.raw.shape
        self._occupied[row : row + height, col : col + width] |= shape.raw
        if self._bounds is None:
            self._bounds = (col, row, col + width, row + height)
        else:
            c0, r0, c1, r1 = self._bounds
            self._bounds = (min(c0, col), min(r0, row), max(c1, col + width), max(r1, row + height))

    def score(self, col: int, row: int, gravity: Gravity) -> int:
        """The gravity score ``best`` minimises — lets a caller compare positions of different shapes."""
        if gravity == "y":
            return row * self._width + col
        if gravity == "x":
            return col * self._height + row
        return (row + col) * (self._height + self._width) + row
