import numpy as np
import pytest
from shapes import box

from platenest import Options, Part, Zone, validate


def test_validate_accepts_a_box():
    validate([Part(*box(10, 10, 5))], Zone(100, 100), Options())


@pytest.mark.parametrize("bad", ["nan", "index", "empty", "shape"])
def test_validate_names_the_bad_part(bad):
    v, t = box(10, 10, 5)
    if bad == "nan":
        v = v.copy()
        v[0, 0] = np.nan
    if bad == "index":
        t = t.copy()
        t[0, 0] = len(v)
    if bad == "empty":
        v, t = np.zeros((0, 3)), np.zeros((0, 3), int)
    if bad == "shape":
        v = v[:, :2]
    with pytest.raises(ValueError, match="part 1"):
        validate([Part(*box(1, 1, 1)), Part(v, t)], Zone(100, 100), Options())


@pytest.mark.parametrize("kw", [{"spacing": -1}, {"resolution": 0}, {"rotations": ()}, {"fill_limit": 0}])
def test_validate_rejects_bad_options(kw):
    with pytest.raises(ValueError):
        validate([Part(*box(1, 1, 1))], Zone(100, 100), Options(**kw))


def test_validate_rejects_bad_quantity_and_zone():
    with pytest.raises(ValueError, match="part 0"):
        validate([Part(*box(1, 1, 1), quantity=0)], Zone(100, 100), Options())
    with pytest.raises(ValueError):
        validate([Part(*box(1, 1, 1))], Zone(0, 100), Options())
