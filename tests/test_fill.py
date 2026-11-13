from shapes import box

from platenest import Options, Part, Rejected, Zone, pack


def test_fill_gives_one_plate_and_a_count():
    r = pack([Part(*box(40, 40, 5), fill=True)], Zone(100, 100), Options(rotations=(0,), centre=False))
    assert r.plate_count == 1 and r.fill_counts == {0: 4} and len(r.placements) == 4


def test_fill_plates_follow_counted_plates_and_do_not_mix():
    parts = [
        Part(*box(30, 30, 5), fill=True),
        Part(*box(60, 60, 5), quantity=2),
        Part(*box(45, 45, 5), fill=True),
    ]
    r = pack(parts, Zone(140, 140), Options(centre=False))
    by_plate = {pl: {p.part for p in r.placements if p.plate == pl} for pl in range(r.plate_count)}
    assert by_plate == {0: {1}, 1: {0}, 2: {2}}
    assert r.fill_counts == {0: 16, 2: 4}


def test_fill_ignores_quantity():
    a = pack([Part(*box(40, 40, 5), fill=True, quantity=1)], Zone(100, 100), Options(centre=False))
    b = pack([Part(*box(40, 40, 5), fill=True, quantity=99)], Zone(100, 100), Options(centre=False))
    assert a.fill_counts == b.fill_counts == {0: 4}


def test_fill_stops_at_the_limit_and_says_so():  # Review Focus 2
    r = pack([Part(*box(2, 2, 2), fill=True)], Zone(200, 200), Options(spacing=1, fill_limit=50, centre=False))
    assert r.fill_counts == {0: 50} and r.fill_limited == frozenset({0})


def test_fill_that_runs_out_of_room_is_not_marked_limited():
    r = pack([Part(*box(40, 40, 5), fill=True)], Zone(100, 100), Options(rotations=(0,), centre=False))
    assert r.fill_limited == frozenset()


def test_rejected_fill_part_has_no_plate():
    r = pack([Part(*box(500, 5, 5), fill=True)], Zone(100, 100), Options(rotations=(0,), centre=False))
    assert r.plate_count == 0 and r.fill_counts == {} and r.rejected == (Rejected(0, "too_large"),)
