import numpy as np
from shapes import box, frame, sliver

from platenest import Options, Part
from platenest.raster import footprints, grow_pixels


def test_grow_pixels_adds_one_for_rounding_and_is_zero_without_spacing():
    assert grow_pixels(2.0, 0.5) == 5
    assert grow_pixels(2.0, 1.0) == 3
    assert grow_pixels(0.0, 0.5) == 0


def test_box_footprint_size_and_origin():
    (s,) = footprints(Part(*box(20, 10, 5)), Options(rotations=(0,), spacing=2, resolution=0.5))
    assert s.raw.shape in {(20, 40), (21, 41)} and s.raw.all()
    assert s.grow == 5 and s.grown.shape == (s.raw.shape[0] + 10, s.raw.shape[1] + 10)


def test_frame_keeps_its_hole():
    (s,) = footprints(Part(*frame(60, 60, 5, 4)), Options(rotations=(0,), resolution=0.5))
    assert not s.raw[60, 60]  # centre is empty
    assert s.raw[2, 60]  # wall is filled


def test_overlapping_faces_do_not_cancel():
    v, t = box(20, 20, 5)  # top and bottom faces overlap in projection
    (s,) = footprints(Part(v, t), Options(rotations=(0,), resolution=0.5))
    assert s.raw.mean() > 0.99


def test_square_collapses_equal_rotations():
    assert len(footprints(Part(*box(10, 10, 5)), Options())) == 1
    assert len(footprints(Part(*box(20, 10, 5)), Options())) == 2


def test_sliver_has_no_footprint():
    assert footprints(Part(*sliver(10)), Options()) == ()


def test_grown_is_raw_dilated_round():
    (s,) = footprints(Part(*box(10, 10, 5)), Options(rotations=(0,), spacing=2, resolution=1))
    g = s.grow
    assert s.grown[g:-g, g:-g].all()  # raw is inside
    assert s.grown[g + 5, 0] and s.grown[0, g + 5]  # straight out by g pixels: reached
    assert not s.grown[0, 0]  # the corner is rounded off


def test_origin_round_trip():
    v, t = box(20, 10, 5)
    v = v + [7.0, 3.0, 0.0]
    (s,) = footprints(Part(v, t), Options(rotations=(90,), resolution=0.5))
    # placing raw's corner at plate (0, 0) puts the part origin at s.origin; rotated vertices + origin stay inside raw
    r = np.array([[0, -1], [1, 0]]) @ v[:, :2].T
    pts = r.T + s.origin
    assert pts.min() >= -0.5 and (pts.max(axis=0) <= np.array(s.raw.shape[::-1]) * 0.5 + 0.5).all()
