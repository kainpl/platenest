import math
from types import SimpleNamespace

import numpy as np
import pytest
import spyrrow
from shapes import box, u_shape

import platenest.optimization as search
from platenest import Cancelled, Options, Part, Zone, optimize, pack
from platenest.optimization import _geometry, _valid


@pytest.mark.parametrize("recovery", [False, True], ids=["bitmap-restart", "free-angle-recovery"])
@pytest.mark.parametrize("signal", ["cancel", "cancel-exception", "finish", "deadline", "resource-error"])
def test_raster_stop_observation_is_not_polled_again(monkeypatch, recovery, signal):
    vertices, triangles = box(100, 3, 2) if recovery else box(20, 20, 2)
    angle = math.radians(7.5) if recovery else 0
    rotation = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    parts = [Part(np.asarray(vertices) @ rotation, triangles)]
    zone = Zone(77, 77)
    options = Options(spacing=2, rotations=tuple(range(0, 360, 15)))
    expected = pack(parts, zone, options)
    assert bool(expected.rejected) is recovery
    real_pack = search.pack
    calls = 0
    armed = False
    observations = 0

    def traced_pack(*args, **kwargs):
        nonlocal calls, armed
        calls += 1
        armed = calls > 1
        return real_pack(*args, **kwargs)

    def cancel():
        nonlocal observations
        if armed and signal in {"cancel", "cancel-exception", "resource-error"}:
            observations += 1
            if observations == 1:
                if signal == "cancel-exception":
                    raise Cancelled
                if signal == "resource-error":
                    raise MemoryError
                return True
        return False

    def finish():
        nonlocal observations
        if armed and signal == "finish":
            observations += 1
            return observations == 1
        return False

    def monotonic():
        nonlocal observations
        if armed and signal == "deadline":
            observations += 1
            return 20.0
        return 0.0

    monkeypatch.setattr(search, "pack", traced_pack)
    monkeypatch.setattr(search.time, "monotonic", monotonic)
    kwargs = dict(free_rotation=recovery, seconds=10, cancel=cancel, finish=finish)
    if signal in {"cancel", "cancel-exception", "resource-error"}:
        with pytest.raises(MemoryError if signal == "resource-error" else Cancelled):
            optimize(parts, zone, options, **kwargs)
    else:
        assert optimize(parts, zone, options, **kwargs) == expected
    assert calls == 2, "exercise a raster stop inside the optimizer's exception handler"
    assert observations == 1, "use the outcome already observed by the packer"


def test_free_rotation_recovers_between_seed_angles():
    vertices, triangles = box(100, 3, 2)
    angle = math.radians(7.5)
    rotation = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    parts = [Part(np.asarray(vertices) @ rotation, triangles)]
    zone = Zone(77, 77)
    options = Options(spacing=2, rotations=tuple(range(0, 360, 15)))
    assert pack(parts, zone, options).rejected
    result = optimize(parts, zone, options, free_rotation=True, seconds=2)
    outlines, region = _geometry(parts, zone, options.spacing)
    assert _valid(result, parts, outlines, region, options, True)
    assert len(result.placements) == 1
    assert optimize(parts, zone, options, free_rotation=False, seconds=2).rejected


@pytest.mark.parametrize("height,zone", [(100, Zone(20, 20, max_height=10)), (2, Zone(20, 20))])
def test_rotation_independent_rejections_stay_fast(monkeypatch, height, zone):
    def forbidden(*args, **kwargs):
        pytest.fail("a proven impossible part entered native search")

    monkeypatch.setattr(spyrrow, "StripPackingInstance", forbidden)
    result = optimize([Part(*box(100, 100, height))], zone, free_rotation=True, seconds=2)
    assert result.rejected


def test_recoverable_native_failure_keeps_valid_layout_and_diagnostic(monkeypatch, caplog):
    def fail(config):
        raise ValueError("recoverable injected failure")

    monkeypatch.setattr(spyrrow, "StripPackingInstance", lambda *a: SimpleNamespace(solve=fail))
    parts = [Part(*u_shape(24, 25, 4, 3), fill=True), Part(*u_shape(22, 24, 5, 3), fill=True)]
    zone, options = Zone(60, 60), Options(spacing=2, rotations=tuple(range(0, 360, 15)))
    events = []
    result = optimize(
        parts,
        zone,
        options,
        free_rotation=True,
        progress=events.append,
        finish=lambda: any(e.diagnostic for e in events),
    )
    outlines, region = _geometry(parts, zone, options.spacing)
    assert _valid(result, parts, outlines, region, options, True)
    assert "recoverable injected failure" in events[-1].diagnostic
    assert "keeping the validated layout" in caplog.text


@pytest.mark.parametrize("error", [Cancelled, MemoryError])
def test_native_cancel_and_resource_failure_are_not_swallowed(monkeypatch, error):
    def fail(config):
        raise error()

    monkeypatch.setattr(spyrrow, "StripPackingInstance", lambda *a: SimpleNamespace(solve=fail))
    with pytest.raises(error):
        optimize([Part(*box(20, 20, 3), fill=True)], Zone(80, 80), seconds=3)


@pytest.mark.parametrize("reverse", [False, True])
def test_each_active_group_gets_restricted_and_continuous_trials(monkeypatch, reverse):
    calls = []
    item = spyrrow.Item

    def traced(name, polygon, count, rotations):
        calls.append((name, rotations is None))
        return item(name, polygon, count, rotations)

    monkeypatch.setattr(spyrrow, "Item", traced)
    monkeypatch.setattr(
        spyrrow, "StripPackingInstance", lambda *a: SimpleNamespace(solve=lambda c: SimpleNamespace(placed_items=[]))
    )
    parts = [Part(*u_shape(24, 25, 4, 3), fill=True), Part(*u_shape(22, 24, 5, 3), fill=True)]
    if reverse:
        parts.reverse()
    result = optimize(parts, Zone(60, 60), free_rotation=True, finish=lambda: len(calls) >= 6)
    assert not result.rejected
    assert len(calls) == 6
    for name in ("0", "1"):
        assert {free for group, free in calls if group == name} == {False, True}


def test_real_sparrow_smoke_keeps_an_original_contour_valid_result():
    parts = [Part(*u_shape(24, 25, 4, 3), fill=True), Part(*u_shape(22, 24, 5, 3), fill=True)]
    zone, options = Zone(60, 60), Options(spacing=2)
    events = []
    result = optimize(
        parts,
        zone,
        options,
        free_rotation=True,
        seconds=10,
        attempt_seconds=1,
        progress=events.append,
        finish=lambda: any(e.engine == "sparrow" for e in events),
    )
    outlines, region = _geometry(parts, zone, options.spacing)
    assert any(e.engine == "sparrow" for e in events)
    assert _valid(result, parts, outlines, region, options, True)
