from shapes import box, frame

from platenest import Options, Part, Zone
from platenest.plate import Plate
from platenest.raster import footprints
from platenest.zone import allowed_mask


def _plate(w=100, h=100, spacing=2):
    return Plate(allowed_mask(Zone(w, h), Options(spacing=spacing, resolution=1)))


def _shape(mesh, spacing=2):
    return footprints(Part(*mesh), Options(rotations=(0,), spacing=spacing, resolution=1))[0]


def test_a_new_plate_is_empty():
    p = _plate()
    assert p.empty
    s = _shape(box(20, 10, 5))
    p.put(s, *p.best(s, "y"))
    assert not p.empty and p.bounds() == (1, 1, 21, 11)


def test_first_part_goes_to_the_front_left_inside_the_margin():
    assert _plate().best(_shape(box(20, 10, 5)), "y") == (1, 1)


def test_second_part_keeps_the_spacing():
    p, s = _plate(), _shape(box(20, 10, 5))
    p.put(s, *p.best(s, "y"))
    col, row = p.best(s, "y")
    assert (col, row) == (1 + 20 + 3, 1)  # grow = ceil(2/1)+1 = 3 px


def test_small_part_lands_inside_a_frame():
    p = _plate(62, 62)  # no room beside the frame: the hole is the only free place
    f = _shape(frame(60, 60, 5, 4))
    p.put(f, *p.best(f, "y"))
    s = _shape(box(20, 20, 5))
    col, row = p.best(s, "y")
    assert 1 + 5 < col < 1 + 55 - 20 and 1 + 5 < row < 1 + 55 - 20


def test_no_room_returns_none():
    p, s = _plate(30, 30), _shape(box(20, 20, 5))
    p.put(s, *p.best(s, "y"))
    assert p.best(s, "y") is None


def test_a_shape_larger_than_the_plate_returns_none():
    assert _plate(30, 30).best(_shape(box(40, 10, 5)), "y") is None


def test_blocked_area_is_respected():
    z = Zone(100, 100, excluded=[[(0, 0), (50, 0), (50, 100), (0, 100)]])
    p = Plate(allowed_mask(z, Options(spacing=0, resolution=1)))
    col, _ = p.best(_shape(box(10, 10, 5), spacing=0), "x")
    assert col >= 50


def test_gravity_changes_the_second_position():
    a, b = _plate(), _plate()
    s = _shape(box(20, 10, 5))
    for p in (a, b):
        p.put(s, *p.best(s, "y"))
    assert a.best(s, "y") != b.best(s, "x")
