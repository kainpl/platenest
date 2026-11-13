from platenest import Options, Zone
from platenest.zone import allowed_mask, grid_size


def test_grid_size_rounds_up():
    assert grid_size(Zone(100, 80), 0.5) == (200, 160)
    assert grid_size(Zone(10.2, 10), 0.5) == (21, 20)


def test_plain_zone_keeps_half_spacing_from_the_border():
    m = allowed_mask(Zone(100, 80), Options(spacing=2, resolution=0.5))
    assert m.shape == (160, 200)
    assert not m[:2].any() and not m[-2:].any() and not m[:, :2].any() and not m[:, -2:].any()
    assert m[2:-2, 2:-2].all()


def test_excluded_corner_and_allowed_strip():
    z = Zone(
        100,
        100,
        allowed=[[(0, 0), (90, 0), (90, 100), (0, 100)]],
        excluded=[[(0, 0), (18, 0), (18, 28), (0, 28)]],
    )
    m = allowed_mask(z, Options(spacing=0, resolution=1))
    assert not m[10, 10] and m[40, 10] and not m[50, 95] and m[50, 85]


def test_fully_excluded_zone_is_all_false():
    z = Zone(50, 50, excluded=[[(0, 0), (50, 0), (50, 50), (0, 50)]])
    assert not allowed_mask(z, Options()).any()
