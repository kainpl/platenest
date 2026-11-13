"""Regressions for the findings of the whole-branch review."""

import io
import json
import time
import zipfile

import pytest
from shapes import box, frame

from platenest import Options, Part, Rejected, Zone, pack, validate
from platenest.bambu3mf import Instance, MeshObject, read_project, write_project

# --- F1: a part far too big must be refused before anything is allocated for it ---------------


@pytest.mark.parametrize("side", [50_000, 5_000_000])
def test_a_part_in_the_wrong_units_is_too_large_at_once(side):
    started = time.perf_counter()
    r = pack([Part(*box(side, side, 5)), Part(*box(10, 10, 5))], Zone(256, 256), Options(centre=False))
    assert time.perf_counter() - started < 2
    assert r.rejected == (Rejected(0, "too_large"),) and [p.part for p in r.placements] == [1]


def test_an_oversized_fill_part_is_too_large_not_empty():
    r = pack([Part(*box(50_000, 50_000, 5), fill=True)], Zone(256, 256), Options())
    assert r.rejected == (Rejected(0, "too_large"),) and r.plate_count == 0


def test_a_grid_too_fine_to_hold_is_refused():
    with pytest.raises(ValueError, match="resolution"):
        validate([Part(*box(1, 1, 1))], Zone(256, 256), Options(resolution=0.002))
    validate([Part(*box(1, 1, 1))], Zone(350, 320), Options(resolution=0.1))  # a fine but sane grid is fine


# --- F2: read_project answers ValueError, whatever it is handed -------------------------------


def _members() -> dict[str, bytes]:
    objs = [MeshObject("frame", *frame(60, 60, 5, 4)), MeshObject("lid", *box(20, 20, 3))]
    inst = [Instance(0, 0, 10, 10, 0), Instance(1, 0, 25, 25, 90)]
    z = zipfile.ZipFile(io.BytesIO(write_project(objs, inst, bed_size=(256, 256), project_settings=b"{}")))
    return {n: z.read(n) for n in z.namelist()}


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buffer.getvalue()


def _record(**changes) -> bytes:
    record = json.loads(_members()["Metadata/platenest.json"])
    record.update(changes)
    return json.dumps(record).encode()


@pytest.mark.parametrize(
    "member, content",
    [
        ("3D/Objects/part_0.model", b'<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
                                    b'<vertex y="1" z="1"/></model>'),
        ("3D/Objects/part_0.model", b'<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
                                    b'<triangle v2="1" v3="2"/></model>'),
        ("3D/Objects/part_0.model", b'<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
                                    b'<vertex x="a" y="1" z="1"/></model>'),
        ("Metadata/platenest.json", b"[" * 200_000),
        ("Metadata/platenest.json", b"[1, 2]"),
        ("Metadata/platenest.json", _record(instances=[{"object": 99, "plate": 0, "x": 0, "y": 0, "angle": 0}])),
        ("Metadata/platenest.json", _record(instances=[{"object": 0, "plate": -5, "x": 0, "y": 0, "angle": 0}])),
        ("Metadata/platenest.json", _record(instances=[{"object": 0, "plate": 0, "x": "a", "y": 0, "angle": 0}])),
        ("Metadata/platenest.json", _record(bed_size="ab")),
        ("Metadata/platenest.json", _record(bed_size=[0, 10])),
        ("Metadata/platenest.json", _record(objects="frame")),
    ],
)  # fmt: skip
def test_a_damaged_member_is_a_value_error(member, content):
    members = _members()
    members[member] = content
    with pytest.raises(ValueError, match="not a platenest project"):
        read_project(_zip(members))


@pytest.mark.parametrize("member", ["3D/Objects/part_0.model", "Metadata/platenest.json"])
def test_corrupted_bytes_are_a_value_error(member):
    data = bytearray(_zip(_members()))
    info = next(i for i in zipfile.ZipFile(io.BytesIO(bytes(data))).infolist() if i.filename == member)
    start = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    for k in range(start + 2, start + min(info.compress_size, 40)):
        data[k] ^= 0xFF
    with pytest.raises(ValueError, match="not a platenest project"):
        read_project(bytes(data))


def test_an_archive_that_unpacks_too_large_is_refused_before_reading():
    members = _members()
    members["Metadata/big.bin"] = b"\0" * 5_000_000
    data = _zip(members)
    assert len(data) < 200_000
    with pytest.raises(ValueError, match="too large"):
        read_project(data, max_bytes=1_000_000)
    assert len(read_project(data).extra_files["Metadata/big.bin"]) == 5_000_000


# --- F5: a long fill must not look like a hang ------------------------------------------------


def test_fill_reports_progress_for_its_part():
    seen = []
    parts = [Part(*box(60, 60, 5)), Part(*box(20, 20, 5), fill=True)]
    r = pack(parts, Zone(100, 100), Options(rotations=(0,)), progress=seen.append)
    fill = [p for p in seen if p.fill_part == 1]
    assert max(p.placed for p in fill) == r.fill_counts[1] and all(p.plate == 1 for p in fill)
    assert all(p.fill_part is None for p in seen if p not in fill) and len(seen) > len(fill) > 0
