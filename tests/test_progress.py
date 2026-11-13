import numpy as np
import pytest
from shapes import box

from platenest import Cancelled, Options, Part, Zone, pack, render_plate


def test_progress_reports_every_copy_of_every_run():
    seen = []
    pack([Part(*box(10, 10, 5), quantity=3)], Zone(100, 100), Options(), progress=seen.append)
    assert seen[0].runs == 6 and seen[-1].run == 6 and max(p.placed for p in seen) == 3 and seen[0].total == 3
    assert [p.placed for p in seen if p.run == 1] == [1, 2, 3]


def test_cancel_raises_before_finishing():
    calls = iter([False, False, True])
    with pytest.raises(Cancelled):
        pack([Part(*box(10, 10, 5), quantity=50)], Zone(100, 100), Options(), cancel=lambda: next(calls, True))


def test_cancel_stops_a_fill_too():
    calls = iter([False] * 5)
    with pytest.raises(Cancelled):
        pack([Part(*box(5, 5, 5), fill=True)], Zone(100, 100), Options(), cancel=lambda: next(calls, True))


def test_preview_shape_and_distinct_colours():
    parts = [Part(*box(30, 30, 5)), Part(*box(20, 20, 5))]
    opt = Options(resolution=1, rotations=(0,))
    r = pack(parts, Zone(100, 80), opt)
    img = render_plate(r, parts, Zone(100, 80), opt, 0, scale=2)
    assert img.shape == (160, 200, 3) and img.dtype == np.uint8
    assert len({tuple(c) for c in img.reshape(-1, 3)}) >= 4


def test_preview_is_flipped_so_the_plate_front_is_at_the_bottom():
    parts = [Part(*box(30, 10, 5))]
    opt = Options(resolution=1, rotations=(0,), centre=False, spacing=0)
    r = pack(parts, Zone(100, 80), opt)  # the box sits at the front-left corner
    img = render_plate(r, parts, Zone(100, 80), opt, 0, scale=1)
    background = tuple(img[5, 90])
    assert tuple(img[-5, 10]) != background and tuple(img[5, 10]) == background


def test_preview_rejects_a_missing_plate():
    parts = [Part(*box(10, 10, 5))]
    r = pack(parts, Zone(100, 100), Options())
    with pytest.raises(IndexError):
        render_plate(r, parts, Zone(100, 100), Options(), 3)
