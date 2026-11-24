import math

import numpy as np
import pytest
from shapes import box

import platenest.optimization as search
from platenest import Cancelled, Options, Part, Zone, optimize, pack


def test_cancellation_observed_during_recovery_must_propagate(monkeypatch):
    vertices, triangles = box(100, 3, 2)
    angle = math.radians(7.5)
    rotation = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    parts = [Part(np.asarray(vertices) @ rotation, triangles)]
    zone = Zone(77, 77)
    options = Options(spacing=2, rotations=tuple(range(0, 360, 15)))
    assert pack(parts, zone, options).rejected
    original = search._normalize
    armed = False
    observed = False

    def normalize(*args, **kwargs):
        nonlocal armed
        result = original(*args, **kwargs)
        armed = True
        return result

    def cancel():
        nonlocal observed
        if armed and not observed:
            observed = True
            return True
        return False

    monkeypatch.setattr(search, "_normalize", normalize)
    try:
        with pytest.raises(Cancelled):
            optimize(parts, zone, options, free_rotation=True, seconds=10, cancel=cancel)
    finally:
        assert observed, "the recovery packer must see the cancellation"
