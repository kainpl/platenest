import numpy as np
from shapes import box, frame, min_gap, overlap_area, ring, sliver, transformed, u_shape

from platenest import Options, Part, Rejected, Zone, pack

OPT = Options(resolution=0.5, centre=False)


def test_nests_a_box_inside_a_frame():
    r = pack([Part(*frame(130, 110, 6, 6)), Part(*box(40, 40, 6))], Zone(150, 130), OPT)
    assert r.plate_count == 1 and len(r.placements) == 2 and r.rejected == ()


def test_quantities_spill_onto_more_plates():
    r = pack([Part(*box(60, 60, 5), quantity=5)], Zone(140, 140), OPT)
    assert r.plate_count == 2 and len(r.placements) == 5
    assert sorted({p.plate for p in r.placements}) == [0, 1]


def test_is_deterministic():
    parts = [
        Part(*frame(130, 110, 6, 6)),
        Part(*u_shape(100, 80, 8, 6), quantity=2),
        Part(*box(15, 15, 6), quantity=12),
    ]
    assert pack(parts, Zone(256, 256), OPT) == pack(parts, Zone(256, 256), OPT)


def test_rejections_do_not_stop_the_rest():
    parts = [Part(*box(300, 10, 5)), Part(*box(10, 10, 50)), Part(*sliver(5)), Part(*box(10, 10, 5))]
    r = pack(parts, Zone(100, 100, max_height=40), Options(rotations=(0,), centre=False))
    assert {(x.part, x.reason) for x in r.rejected} == {(0, "too_large"), (1, "too_tall"), (2, "empty_mesh")}
    assert [p.part for p in r.placements] == [3]


def test_rotation_lets_a_long_part_fit():
    r = pack([Part(*box(10, 90, 5))], Zone(100, 50), Options(centre=False))
    assert r.rejected == () and r.placements[0].angle in (90.0, 270.0)


def test_placements_stay_inside_the_plate():
    parts = [Part(*box(30, 12, 5), quantity=9)]
    zone = Zone(100, 100)
    r = pack(parts, zone, Options(centre=False))
    for p in r.placements:
        v = transformed(np.asarray(parts[0].vertices), p)
        assert v[:, 0].min() >= 0 and v[:, 1].min() >= 0 and v[:, 0].max() <= 100 and v[:, 1].max() <= 100


def test_zero_spacing_touches_but_never_overlaps():  # Review Focus 3
    parts = [Part(*box(50, 50, 5), quantity=4)]
    r = pack(parts, Zone(100, 100), Options(spacing=0, rotations=(0,), centre=False))
    assert r.plate_count == 1 and overlap_area(r, parts) == 0


def test_fully_excluded_zone_rejects_everything():  # Review Focus 4
    z = Zone(50, 50, excluded=[[(0, 0), (50, 0), (50, 50), (0, 50)]])
    r = pack([Part(*box(5, 5, 5), quantity=3)], z, OPT)
    assert r.plate_count == 0 and r.placements == () and r.rejected == (Rejected(0, "too_large"),)


def test_part_the_size_of_the_zone_is_too_large_with_spacing():  # Review Focus 4
    r = pack([Part(*box(100, 100, 5))], Zone(100, 100), OPT)
    assert r.rejected == (Rejected(0, "too_large"),)


def test_real_gap_is_never_below_the_request():
    parts = [Part(*ring(110, 80, 6)), Part(*u_shape(100, 80, 8, 6)), Part(*box(20, 12, 6), quantity=10)]
    r = pack(parts, Zone(256, 256), Options(spacing=2.0, resolution=0.5, centre=False))
    assert r.rejected == () and min_gap(r, parts) >= 2.0
