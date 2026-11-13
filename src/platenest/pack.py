"""``pack()``: put parts on as few plates as possible.

A run is one (order, gravity) pair: parts sorted by an order are placed one copy at a time, each
on the first plate that has a free position, at the position — over all its rotations — with the
smallest gravity score. Several runs are tried and the best one is kept.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from platenest.plate import GRAVITIES, Gravity, Plate
from platenest.raster import Shape, footprints
from platenest.types import Options, PackResult, Part, Placement, Rejected, Zone, validate
from platenest.zone import allowed_mask

_ORDERS = ("area", "longest")
_HEIGHT_EPS = 1e-6


class Cancelled(Exception):
    """Raised by :func:`pack` when the ``cancel`` callback returns True."""


@dataclass(frozen=True)
class Progress:
    run: int  # 1-based
    runs: int
    placed: int  # copies placed in this run so far
    total: int  # copies this run will place
    plate: int  # 0-based plate the last copy went to


@dataclass(frozen=True, eq=False)
class _Item:
    part: int
    plate: int
    shape: Shape
    col: int
    row: int


def _reject(parts: Sequence[Part], shapes: list[tuple[Shape, ...]], zone: Zone, empty: Plate) -> dict[int, str]:
    rejected: dict[int, str] = {}
    for index, part in enumerate(parts):
        z = np.asarray(part.vertices, dtype=np.float64)[:, 2]
        if not shapes[index]:
            rejected[index] = "empty_mesh"
        elif zone.max_height is not None and float(z.max() - z.min()) > zone.max_height + _HEIGHT_EPS:
            rejected[index] = "too_tall"
        elif all(empty.best(shape, "y") is None for shape in shapes[index]):
            rejected[index] = "too_large"
    return rejected


def _place(plate: Plate, shapes: tuple[Shape, ...], gravity: Gravity) -> tuple[Shape, int, int] | None:
    """The best position on one plate over all rotations (earlier rotation wins a tie)."""
    best: tuple[int, Shape, int, int] | None = None
    for shape in shapes:
        position = plate.best(shape, gravity)
        if position is None:
            continue
        score = plate.score(*position, gravity)
        if best is None or score < best[0]:
            best = (score, shape, *position)
    return None if best is None else best[1:]


def _order(copies: list[tuple[int, int]], shapes: list[tuple[Shape, ...]], order: str) -> list[tuple[int, int]]:
    def size(part: int) -> int:
        raw = shapes[part][0].raw
        return int(raw.sum()) if order == "area" else max(raw.shape)

    return sorted(copies, key=lambda c: (-size(c[0]), c[0], c[1]))


def _run(
    copies: list[tuple[int, int]],
    shapes: list[tuple[Shape, ...]],
    gravity: Gravity,
    new_plate: Callable[[], Plate],
    step: Callable[[int, int], None],
) -> tuple[list[Plate], list[_Item]]:
    plates: list[Plate] = []
    items: list[_Item] = []
    full: dict[int, int] = {}  # part -> first plate still worth trying (plates only ever get fuller)
    for placed, (part, _copy) in enumerate(copies):
        step(placed, len(plates) - 1)
        index = full.get(part, 0)
        while True:
            if index == len(plates):
                plates.append(new_plate())
            found = _place(plates[index], shapes[part], gravity)
            if found is not None:
                break
            index += 1
            full[part] = index
        shape, col, row = found
        plates[index].put(shape, col, row)
        items.append(_Item(part, index, shape, col, row))
    return plates, items


def _fill(
    shapes: tuple[Shape, ...],
    part: int,
    limit: int,
    new_plate: Callable[[], Plate],
    check_cancel: Callable[[], None],
) -> tuple[Plate, list[_Item], bool]:
    """One plate holding as many copies as fit; the gravity that fits the most wins (earliest on a tie)."""
    best: tuple[Plate, list[tuple[Shape, int, int]], bool] | None = None
    for gravity in GRAVITIES:
        plate, placed, limited = new_plate(), [], False
        while True:
            check_cancel()
            found = _place(plate, shapes, gravity)
            if found is None:
                break
            if len(placed) == limit:
                limited = True
                break
            plate.put(*found)
            placed.append(found)
        if best is None or len(placed) > len(best[1]):
            best = (plate, placed, limited)
    plate, placed, limited = best
    return plate, [_Item(part, -1, shape, col, row) for shape, col, row in placed], limited


def _shifted_ok(items: list[_Item], allowed: np.ndarray, dx: int, dy: int) -> bool:
    height, width = allowed.shape
    for item in items:
        h, w = item.shape.raw.shape
        col, row = item.col + dx, item.row + dy
        if col < 0 or row < 0 or col + w > width or row + h > height:
            return False
        if (item.shape.raw & ~allowed[row : row + h, col : col + w]).any():
            return False
    return True


def _centre(items: list[_Item], allowed: np.ndarray) -> list[_Item]:
    """Move one plate's group towards the centre of the allowed area, as far as it stays allowed.

    Every copy moves by the same whole-pixel vector, so the gaps between them do not change.
    """
    if not items or not allowed.any():
        return items
    rows, cols = np.nonzero(allowed.any(axis=1))[0], np.nonzero(allowed.any(axis=0))[0]
    zone_cols, zone_rows = cols[0] + cols[-1] + 1, rows[0] + rows[-1] + 1  # twice the centre
    c0 = min(i.col for i in items)
    c1 = max(i.col + i.shape.raw.shape[1] for i in items)
    r0 = min(i.row for i in items)
    r1 = max(i.row + i.shape.raw.shape[0] for i in items)
    want_x, want_y = (zone_cols - (c0 + c1)) // 2, (zone_rows - (r0 + r1)) // 2
    for k in range(16, 0, -1):
        dx, dy = round(want_x * k / 16), round(want_y * k / 16)
        if (dx or dy) and _shifted_ok(items, allowed, dx, dy):
            return [_Item(i.part, i.plate, i.shape, i.col + dx, i.row + dy) for i in items]
    return items


def pack(
    parts: Sequence[Part],
    zone: Zone,
    options: Options = Options(),
    *,
    progress: Callable[[Progress], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> PackResult:
    """Pack ``parts`` into ``zone``. See :mod:`platenest.types` for the placement contract.

    Parts that cannot be placed at all come back in ``rejected``; the rest are still packed.
    Raises ``ValueError`` for malformed input and :class:`Cancelled` when ``cancel()`` returns True.
    """
    validate(parts, zone, options)
    allowed = allowed_mask(zone, options)
    cache: dict[int, np.ndarray] = {}

    def new_plate() -> Plate:
        return Plate(allowed, cache)

    shapes = [footprints(part, options) for part in parts]
    rejected = _reject(parts, shapes, zone, new_plate())

    copies = [
        (index, copy)
        for index, part in enumerate(parts)
        if index not in rejected and not part.fill
        for copy in range(part.quantity)
    ]

    runs = [(order, gravity) for order in _ORDERS for gravity in GRAVITIES]
    best: tuple[tuple[int, int, int], list[Plate], list[_Item]] | None = None
    for number, (order, gravity) in enumerate(runs, start=1):
        if not copies:
            break

        def step(placed: int, plate: int, number: int = number) -> None:
            if cancel is not None and cancel():
                raise Cancelled
            if progress is not None and placed:
                progress(Progress(number, len(runs), placed, len(copies), plate))

        plates, items = _run(_order(copies, shapes, order), shapes, gravity, new_plate, step)
        if progress is not None:
            progress(Progress(number, len(runs), len(copies), len(copies), len(plates) - 1))
        c0, r0, c1, r1 = plates[-1].bounds()
        key = (len(plates), (c1 - c0) * (r1 - r0), number)
        if best is None or key < best[0]:
            best = (key, plates, items)

    plates, items = (best[1], best[2]) if best is not None else ([], [])

    def check_cancel() -> None:
        if cancel is not None and cancel():
            raise Cancelled

    fill_counts: dict[int, int] = {}
    fill_limited: set[int] = set()
    for index, part in enumerate(parts):
        if not part.fill or index in rejected:
            continue
        plate, filled, limited = _fill(shapes[index], index, options.fill_limit, new_plate, check_cancel)
        items = items + [_Item(index, len(plates), i.shape, i.col, i.row) for i in filled]
        plates = plates + [plate]
        fill_counts[index] = len(filled)
        if limited:
            fill_limited.add(index)
    if options.centre:
        items = [
            moved for plate in range(len(plates)) for moved in _centre([i for i in items if i.plate == plate], allowed)
        ]

    resolution = options.resolution
    placements = tuple(
        Placement(
            part=item.part,
            plate=item.plate,
            x=item.col * resolution + item.shape.origin[0],
            y=item.row * resolution + item.shape.origin[1],
            angle=item.shape.angle,
        )
        for item in sorted(items, key=lambda i: (i.plate, i.part, i.row, i.col))
    )
    return PackResult(
        plate_count=len(plates),
        placements=placements,
        rejected=tuple(Rejected(index, reason) for index, reason in sorted(rejected.items())),
        fill_counts=fill_counts,
        fill_limited=frozenset(fill_limited),
    )
