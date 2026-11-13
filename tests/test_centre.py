from shapes import box, transformed_bounds

from platenest import Options, Part, Zone, pack


def test_single_box_ends_in_the_middle():
    parts = [Part(*box(20, 20, 5))]
    r = pack(parts, Zone(100, 100), Options(rotations=(0,), resolution=0.5))
    v = transformed_bounds(r, parts)  # (x0, y0, x1, y1)
    assert abs((v[0] + v[2]) / 2 - 50) <= 0.5 and abs((v[1] + v[3]) / 2 - 50) <= 0.5


def test_centres_on_the_work_zone_not_the_plate():
    z = Zone(330, 320, allowed=[[(25, 0), (330, 0), (330, 320), (25, 320)]])
    parts = [Part(*box(20, 20, 5))]
    r = pack(parts, z, Options(rotations=(0,), resolution=0.5))
    v = transformed_bounds(r, parts)
    assert abs((v[0] + v[2]) / 2 - 177.5) <= 1.0


def test_centring_preserves_relative_positions():
    parts = [Part(*box(30, 20, 5), quantity=3)]
    a = pack(parts, Zone(200, 200), Options(centre=False))
    b = pack(parts, Zone(200, 200), Options())
    da = [(p.x - a.placements[0].x, p.y - a.placements[0].y) for p in a.placements]
    db = [(p.x - b.placements[0].x, p.y - b.placements[0].y) for p in b.placements]
    assert da == db and a.placements != b.placements


def test_partial_shift_never_enters_an_excluded_area():
    z = Zone(200, 100, excluded=[[(90, 0), (110, 0), (110, 100), (90, 100)]])  # a wall across the middle
    parts = [Part(*box(40, 40, 5))]
    r = pack(parts, z, Options(spacing=0, rotations=(0,), resolution=1))
    v = transformed_bounds(r, parts)
    assert v[2] <= 90 or v[0] >= 110
    assert v[0] > 0  # it did move towards the middle as far as it could


def test_fill_plate_is_centred_too():
    parts = [Part(*box(40, 40, 5), fill=True)]
    r = pack(parts, Zone(100, 100), Options(rotations=(0,)))
    xs = [transformed_bounds(r, parts, i) for i in range(len(r.placements))]
    assert abs(min(b[0] for b in xs) - (100 - max(b[2] for b in xs))) <= 1.0
