"""Optional, bounded search using the existing bitmap packer and Sparrow.

``pack`` keeps its deterministic contract and light dependencies. Install the
``optimize`` extra for this entry point. Every accepted candidate is checked on
the original projected triangles; only translation and Z rotation are permitted.
Native solver calls are short, synchronous attempts. Stop is cooperative between
attempts; callers needing a hard deadline should retain their process supervisor.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from platenest.packing import Cancelled, pack
from platenest.types import Options, PackResult, Part, Placement, Zone

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchProgress:
    engine: str
    elapsed: float
    best: PackResult
    attempt: int
    diagnostic: str | None = None


def _geometry(parts, zone, spacing):
    from shapely import Polygon, union_all
    from shapely.geometry import box

    outlines = []
    for part in parts:
        vertices = np.asarray(part.vertices, dtype=float)
        triangles = np.asarray(part.triangles)
        polygons = [Polygon(vertices[t, :2]) for t in triangles]
        outline = union_all([p for p in polygons if p.area > 1e-10])
        if outline.is_empty or not outline.is_valid:
            raise ValueError("part has an invalid projected outline")
        outlines.append(outline)
    region = box(0, 0, zone.width, zone.depth)
    if zone.allowed:
        region = region.intersection(union_all([Polygon(p) for p in zone.allowed]))
    if zone.excluded:
        region = region.difference(union_all([Polygon(p) for p in zone.excluded]))
    return outlines, region.buffer(-spacing / 2) if spacing else region


def _transformed(outline, placement):
    from shapely.affinity import affine_transform

    angle = math.radians(placement.angle)
    c, s = math.cos(angle), math.sin(angle)
    return affine_transform(outline, [c, -s, s, c, placement.x, placement.y])


def _valid(result, parts, outlines, region, options, free):
    from shapely import STRtree

    if result.rejected or result.plate_count < 1:
        return False
    counts = Counter(p.part for p in result.placements)
    if set(counts) != set(range(len(parts))):
        return False
    for i, part in enumerate(parts):
        if counts[i] != part.quantity and not part.fill:
            return False
        if part.fill and not 1 <= counts[i] <= options.fill_limit:
            return False
    groups = {}
    for p in result.placements:
        if not all(math.isfinite(v) for v in (p.x, p.y, p.angle)):
            return False
        if not free and not any(abs((p.angle - a + 180) % 360 - 180) < 1e-5 for a in options.rotations):
            return False
        if not 0 <= p.plate < result.plate_count:
            return False
        polygon = _transformed(outlines[p.part], p)
        if not region.buffer(1e-7).covers(polygon):
            return False
        groups.setdefault(p.plate, []).append((p.part, polygon))
    if set(groups) != set(range(result.plate_count)):
        return False
    fill_plates = {}
    for plate, group in groups.items():
        indices = {i for i, _ in group}
        fills = {i for i in indices if parts[i].fill}
        if fills:
            if len(indices) != 1:
                return False
            index = next(iter(fills))
            if index in fill_plates:
                return False
            fill_plates[index] = plate
        polygons = [p for _, p in group]
        tree = STRtree(polygons)
        for i, polygon in enumerate(polygons):
            for j in tree.query(polygon, predicate="dwithin", distance=max(options.spacing, 1e-6)):
                if j <= i:
                    continue
                other = polygons[int(j)]
                if polygon.intersection(other).area > 1e-8 or polygon.distance(other) < options.spacing - 1e-5:
                    return False
    return all(result.fill_counts.get(i) == counts[i] for i in fill_plates)


def _score(result, parts):
    counted = {p.plate for p in result.placements if not parts[p.part].fill}
    return len(counted), -sum(result.fill_counts.values())


def _normalize(placements, outlines, bounds, free, centre=True):
    """Change the complete group's coordinate frame with a proper Z rotation only."""
    from shapely.geometry import MultiPoint

    points = []
    for p in placements:
        points.extend(_transformed(outlines[p.part].convex_hull, p).exterior.coords)
    hull = np.asarray(MultiPoint(points).convex_hull.exterior.coords)
    angles = np.linspace(0, math.pi / 2, 181) if free else np.array([0.0])
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    x = c * hull[:, 0] - s * hull[:, 1]
    y = s * hull[:, 0] + c * hull[:, 1]
    widths, heights = np.ptp(x, axis=1), np.ptp(y, axis=1)
    w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
    k = int(np.argmin(np.maximum(widths / w, heights / h)))
    dx = (bounds[0] + bounds[2] - x[k].min() - x[k].max()) / 2
    dy = (bounds[1] + bounds[3] - y[k].min() - y[k].max()) / 2
    if not centre:
        dx, dy = bounds[0] - x[k].min(), bounds[1] - y[k].min()
    co, si = float(c[k, 0]), float(s[k, 0])
    angle = math.degrees(float(angles[k]))
    return tuple(
        replace(p, x=co * p.x - si * p.y + dx, y=si * p.x + co * p.y + dy, angle=(p.angle + angle) % 360)
        for p in placements
    )


def _candidate(best, indices, placements, parts, options):
    """Replace just one fill plate, or the complete counted demand with one plate."""
    removed = {p.plate for p in best.placements if p.part in indices}
    retained = [p for p in best.placements if p.plate not in removed]
    plate_ids = {old: new for new, old in enumerate(sorted({p.plate for p in retained}))}
    target = len(plate_ids)
    combined = tuple(replace(p, plate=plate_ids[p.plate]) for p in retained) + tuple(
        replace(p, plate=target + p.plate) for p in placements
    )
    counts = dict(best.fill_counts)
    for index in indices:
        if parts[index].fill:
            counts[index] = sum(p.part == index for p in placements)
    return PackResult(
        target + max(p.plate for p in placements) + 1,
        combined,
        fill_counts=counts,
        fill_limited=frozenset(i for i, n in counts.items() if n >= options.fill_limit),
    )


def _plates(placements, outlines, bounds, free, centre):
    """Adapt a strip result to finite beds, preserving every requested copy.

    Only complete parts enter a bed. Parts crossing a tentative cut remain for
    a later bed. Each group keeps its relative layout and is checked afterwards.
    This is a conservative adapter, not a replacement bin-packing engine.
    """
    pending = [(p, _transformed(outlines[p.part], p).bounds) for p in placements]
    result = []
    width = bounds[2] - bounds[0]
    plate = 0
    while pending:
        left = min(b[0] for _, b in pending)
        selected = [(p, b) for p, b in pending if b[2] - left <= width + 1e-7]
        if not selected:
            return None
        normalized = _normalize(tuple(p for p, _ in selected), outlines, bounds, free, centre)
        result.extend(replace(p, plate=plate) for p in normalized)
        chosen = {id(p) for p, _ in selected}
        pending = [(p, b) for p, b in pending if id(p) not in chosen]
        plate += 1
    return tuple(result)


def optimize(
    parts: Sequence[Part],
    zone: Zone,
    options: Options = Options(),
    *,
    seconds: float = 300,
    free_rotation: bool = False,
    seed: int = 1729,
    progress: Callable[[SearchProgress], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    finish: Callable[[], bool] | None = None,
    attempt_seconds: float = 15,
) -> PackResult:
    """Keep the best feasible layout while trying bitmap restarts and Sparrow.

    ``seconds`` is a soft total budget, including geometry preparation. A native
    attempt can overrun it slightly. ``finish`` returns the best known result;
    ``cancel`` raises ``Cancelled``. Both are checked between native attempts.
    No filename, slicer configuration, mirror, scale or X/Y tilt is involved.
    Fixed demands remain complete; fill parts keep independent, unmixed plates.
    Continuous Z angles are used only when ``free_rotation`` is explicitly true.
    Time-limited optimization is not a byte-reproducibility guarantee.
    """
    if isinstance(seconds, bool) or not math.isfinite(seconds) or not 0 <= seconds <= 3600:
        raise ValueError("seconds must be between 0 and 3600")
    if not math.isfinite(attempt_seconds) or not 0 < attempt_seconds <= 60:
        raise ValueError("attempt_seconds must be between 0 and 60")
    import spyrrow

    started = time.monotonic()
    deadline = started + seconds
    # A positive raster spacing avoids the documented sub-pixel overlaps of pack(spacing=0).
    baseline_options = replace(options, spacing=max(options.spacing, 0.001))
    best = pack(parts, zone, baseline_options, cancel=cancel)
    if best.rejected and (not free_rotation or any(r.reason != "too_large" for r in best.rejected)):
        return best
    outlines, region = _geometry(parts, zone, options.spacing)
    soft_stop_requested = False

    def stopped():
        nonlocal soft_stop_requested
        if cancel is not None and cancel():
            raise Cancelled
        soft_stop_requested = time.monotonic() >= deadline or (finish is not None and finish())
        return soft_stop_requested

    if best.rejected:
        # Raster seed angles are not a proof that free rotation cannot fit.
        # Reuse the existing rigid-group normalization to suggest feasible
        # angles, then let the ordinary packer and original contours judge.
        if stopped() or region.is_empty or any(outlines[r.part].area > region.area for r in best.rejected):
            return best
        angles = list(options.rotations)
        for rejected in best.rejected:
            placed = _normalize((Placement(rejected.part, 0, 0, 0, 0),), outlines, region.bounds, True, False)
            angles.extend((placed[0].angle + quarter) % 360 for quarter in (0, 90, 180, 270))
        try:
            candidate = pack(
                parts, zone, replace(baseline_options, rotations=tuple(dict.fromkeys(angles))), cancel=stopped
            )
            if not candidate.rejected:
                best = candidate
            # Native free angles also get a bounded opportunity if the rigid
            # seed still fails (for example, around a bed exclusion).
            for rejected in candidate.rejected:
                remaining = deadline - time.monotonic()
                if stopped() or remaining < 1:
                    break
                try:
                    item = spyrrow.Item(str(rejected.part), list(outlines[rejected.part].exterior.coords), 1, None)
                    solution = spyrrow.StripPackingInstance(
                        "platenest-initial", region.bounds[3] - region.bounds[1], [item]
                    ).solve(
                        spyrrow.StripPackingConfig(
                            early_termination=False,
                            min_items_separation=options.spacing + 0.001,
                            total_computation_time=max(1, int(min(3, remaining))),
                            num_workers=2,
                            seed=seed,
                        )
                    )
                except ValueError as error:
                    logger.warning("Sparrow initial trial failed: %s", error)
                    continue
                for placed_item in solution.placed_items:
                    placement = Placement(rejected.part, 0, *placed_item.translation, placed_item.rotation)
                    normalized = _normalize((placement,), outlines, region.bounds, True, False)
                    angles.extend((normalized[0].angle + quarter) % 360 for quarter in (0, 90, 180, 270))
                candidate = pack(
                    parts, zone, replace(baseline_options, rotations=tuple(dict.fromkeys(angles))), cancel=stopped
                )
                if not candidate.rejected:
                    best = candidate
                    break
        except Cancelled:
            # Only our observed finish/budget signal is a successful stop.
            # An operator callback may report cancellation just once.
            if not soft_stop_requested:
                raise
            return best
        if best.rejected:
            logger.info("Free-angle initialization exhausted its bounded attempts without a feasible layout")
            return best
    if not _valid(best, parts, outlines, region, options, free_rotation):
        raise ValueError("initial packing did not satisfy the original geometry")
    attempt = 0

    def report(engine, diagnostic=None):
        if progress is not None:
            progress(SearchProgress(engine, time.monotonic() - started, best, attempt, diagnostic))

    def accept(candidate):
        nonlocal best
        if any(candidate.fill_counts.get(i, 0) < n for i, n in best.fill_counts.items()):
            return
        if _score(candidate, parts) < _score(best, parts) and _valid(
            candidate, parts, outlines, region, options, free_rotation
        ):
            best = candidate

    report("bitmap")
    rng = random.Random(seed)
    for _ in range(6):
        if stopped():
            return best
        attempt += 1
        rotations = list(options.rotations)
        rng.shuffle(rotations)
        try:
            candidate = pack(parts, zone, replace(baseline_options, rotations=rotations), cancel=stopped)
        except Cancelled:
            if not soft_stop_requested:
                raise
            return best
        accept(candidate)
        report("bitmap")

    groups = [(i,) for i, part in enumerate(parts) if part.fill and outlines[i].geom_type == "Polygon"]
    fixed = tuple(i for i, part in enumerate(parts) if not part.fill)
    if fixed and all(outlines[i].geom_type == "Polygon" for i in fixed):
        groups.append(fixed)
    if not groups or region.is_empty:
        return best
    bounds = region.bounds
    group_attempts = Counter()
    for turn in range(10000):
        if stopped():
            break
        indices = groups[turn % len(groups)]
        filling = len(indices) == 1 and parts[indices[0]].fill
        if filling:
            index = indices[0]
            demand = best.fill_counts[index] + 1
            if demand > options.fill_limit:
                groups.remove(indices)
                if not groups:
                    break
                continue
            demands = {index: demand}
        else:
            if len({p.plate for p in best.placements if p.part in indices}) <= 1:
                groups.remove(indices)
                if not groups:
                    break
                continue
            demands = {i: int(parts[i].quantity) for i in indices}
        if filling and sum(outlines[i].area * n for i, n in demands.items()) > region.area + 1e-6:
            groups.remove(indices)
            if not groups:
                break
            continue
        remaining = deadline - time.monotonic()
        if remaining < 1:
            break
        # Restricted trials are members of the unrestricted portfolio too.
        continuous = free_rotation and group_attempts[indices] % 2 == 1
        group_attempts[indices] += 1
        rotations = None if continuous else [0, 90, 180, 270] if free_rotation else list(options.rotations)
        items = [spyrrow.Item(str(i), list(outlines[i].exterior.coords), n, rotations) for i, n in demands.items()]
        instance = spyrrow.StripPackingInstance("platenest", bounds[3] - bounds[1], items)
        config = spyrrow.StripPackingConfig(
            early_termination=False,
            min_items_separation=options.spacing + 0.001,
            total_computation_time=max(1, int(min(attempt_seconds, remaining))),
            num_workers=2,
            seed=rng.randrange(2**31),
        )
        attempt += 1
        try:
            solution = instance.solve(config)
        except ValueError as error:
            # A failed optional trial cannot invalidate a checked incumbent.
            # Cancellation, MemoryError and process limits remain distinct.
            diagnostic = f"{type(error).__name__}: {str(error)[:300]}"
            logger.warning("Sparrow trial failed; keeping the validated layout: %s", diagnostic)
            report("sparrow", diagnostic)
            continue
        placed = tuple(
            Placement(int(p.id), 0, p.translation[0], p.translation[1], p.rotation) for p in solution.placed_items
        )
        if len(placed) == sum(demands.values()):
            if filling:
                placed = _normalize(placed, outlines, bounds, free_rotation, options.centre)
            else:
                placed = _plates(placed, outlines, bounds, free_rotation, options.centre)
            if placed:
                accept(_candidate(best, indices, placed, parts, options))
        report("sparrow")
    return best
