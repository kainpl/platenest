import sys
from types import SimpleNamespace

import pytest
from shapes import box, frame

from platenest import Cancelled, Options, PackResult, Part, Placement, Zone, optimize, pack
from platenest.optimization import _geometry, _normalize, _valid


def test_zero_budget_preserves_a_hole_nest_and_core_pack():
    parts = [Part(*frame(120, 120, 6, 5)), Part(*box(90, 90, 5))]
    zone = Zone(180, 180)
    expected = pack(parts, zone)
    assert expected.plate_count == 1
    assert optimize(parts, zone, seconds=0) == expected


def test_finish_returns_a_valid_incumbent_but_cancel_is_not_success():
    parts = [Part(*box(20, 20, 5), fill=True)]
    events = []
    result = optimize(parts, Zone(80, 80), finish=lambda: True, progress=events.append)
    assert result == events[0].best and result.fill_counts[0] > 0
    with pytest.raises(Cancelled):
        optimize(parts, Zone(80, 80), cancel=lambda: True)


@pytest.mark.parametrize("seconds", [-1, True, float("nan"), float("inf"), 3601])
def test_budget_validation(seconds):
    with pytest.raises(ValueError):
        optimize([Part(*box(20, 20, 5))], Zone(80, 80), seconds=seconds)


def test_original_contours_gap_zone_and_angles_are_all_checked():
    parts = [Part(*box(20, 20, 5), quantity=2)]
    options = Options(rotations=(0,))
    zone = Zone(80, 80, excluded=[[(0, 0), (30, 0), (30, 30), (0, 30)]])
    outlines, region = _geometry(parts, zone, 2)
    good = PackResult(1, (Placement(0, 0, 35, 5, 0), Placement(0, 0, 35, 30, 0)))
    assert _valid(good, parts, outlines, region, options, False)
    for positions in [
        (Placement(0, 0, 5, 5, 0), Placement(0, 0, 35, 30, 0)),
        (Placement(0, 0, 35, 5, 0), Placement(0, 0, 35, 26, 0)),
        (Placement(0, 0, 35, 5, 0),),
        (Placement(0, 0, 35, 5, 45), Placement(0, 0, 35, 30, 0)),
    ]:
        assert not _valid(PackResult(1, positions), parts, outlines, region, options, False)


def test_normalization_recovers_solver_coordinate_frame_without_deforming_parts():
    parts = [Part(*box(20, 10, 5), quantity=2)]
    outlines, region = _geometry(parts, Zone(80, 80), 2)
    source = (Placement(0, 0, 110, 3, 13), Placement(0, 0, 140, 3, 13))
    result = _normalize(source, outlines, region.bounds, True)
    assert _valid(PackResult(1, result), parts, outlines, region, Options(), True)
    assert result[1].angle - result[0].angle == pytest.approx(0)
    assert ((result[1].x - result[0].x) ** 2 + (result[1].y - result[0].y) ** 2) ** 0.5 == pytest.approx(30)


def _fake_solver(monkeypatch, positions, baseline=None):
    import platenest.optimization as module

    baseline = baseline or PackResult(1, (Placement(0, 0, 1, 1, 0), Placement(0, 0, 23, 1, 0)), fill_counts={0: 2})
    monkeypatch.setattr(module, "pack", lambda *a, **k: baseline)
    solver = SimpleNamespace(placed_items=[SimpleNamespace(id="0", translation=p, rotation=0) for p in positions])
    monkeypatch.setitem(
        sys.modules,
        "spyrrow",
        SimpleNamespace(
            Item=lambda *a, **k: None,
            StripPackingConfig=lambda **k: None,
            StripPackingInstance=lambda *a, **k: SimpleNamespace(solve=lambda config: solver),
        ),
    )
    return baseline


def test_ready_made_solver_improvement_is_retained_and_reported(monkeypatch):
    _fake_solver(monkeypatch, [(1, 1), (23, 1), (45, 1)])
    events = []
    result = optimize(
        [Part(*box(20, 20, 5), fill=True)],
        Zone(80, 30),
        progress=events.append,
        finish=lambda: bool(events and events[-1].best.fill_counts[0] == 3),
    )
    assert result.fill_counts == {0: 3}
    assert events[-1].engine == "sparrow"


def test_solver_collision_never_replaces_incumbent(monkeypatch):
    baseline = _fake_solver(monkeypatch, [(1, 1), (2, 1), (3, 1)])
    events = []
    result = optimize(
        [Part(*box(20, 20, 5), fill=True)],
        Zone(80, 30),
        progress=events.append,
        finish=lambda: bool(events and events[-1].engine == "sparrow"),
    )
    assert result == baseline


def test_a_strip_can_reduce_multiple_counted_plates_without_losing_copies(monkeypatch):
    baseline = PackResult(3, tuple(Placement(0, plate, x, 1, 0) for plate in range(3) for x in (1, 23)))
    _fake_solver(monkeypatch, [(x, y) for x in (1, 23, 45) for y in (1, 23)], baseline)
    events = []
    result = optimize(
        [Part(*box(20, 20, 5), quantity=6)],
        Zone(60, 60),
        progress=events.append,
        finish=lambda: bool(events and events[-1].best.plate_count == 2),
    )
    assert result.plate_count == 2
    assert len(result.placements) == 6
    assert {p.plate for p in result.placements} == {0, 1}
