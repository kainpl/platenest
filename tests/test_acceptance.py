"""End-to-end checks: nesting really happens, it is fast enough, and the result is a readable project.

The real part set of the original spike is a third-party model and is not in this repository.
Point ``PLATENEST_ACCEPTANCE_3MF`` at a Bambu project 3MF to run the same check against real meshes:
one ``Part`` per distinct mesh, ``quantity`` = how many build items use it.
"""

import os
import time
import zipfile
from collections import Counter
from xml.etree import ElementTree

import numpy as np
import pytest
from shapes import box, frame, min_gap, ring, u_shape

from platenest import Options, Part, Zone, pack
from platenest.bambu3mf import Instance, MeshObject, read_project, write_project

_CORE = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
_PATH = "{http://schemas.microsoft.com/3dmanufacturing/production/2015/06}path"

# One nozzle's reach on a 330 x 320 bed, with the strip the printer keeps clear at the back.
BIG_BED = Zone(
    330,
    320,
    allowed=[[(0, 0), (325, 0), (325, 320), (0, 320)]],
    excluded=[[(145, 310), (251, 310), (251, 326), (145, 326)]],
    max_height=320,
)


def load_parts(path: str) -> list[Part]:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("3D/3dmodel.model"))
        mesh_of = {}
        for obj in root.iter(f"{_CORE}object"):
            component = obj.find(f"{_CORE}components/{_CORE}component")
            if component is not None:
                mesh_of[obj.get("id")] = (component.get(_PATH).lstrip("/"), component.get("objectid"))
        counts = Counter(mesh_of[item.get("objectid")] for item in root.iter(f"{_CORE}item"))
        parts = []
        for (member, object_id), quantity in sorted(counts.items()):
            model = ElementTree.fromstring(archive.read(member))
            mesh = next(o for o in model.iter(f"{_CORE}object") if o.get("id") == object_id)
            vertices = np.array(
                [[float(v.get(a)) for a in "xyz"] for v in mesh.iter(f"{_CORE}vertex")], dtype=np.float64
            )
            triangles = np.array(
                [[int(t.get(a)) for a in ("v1", "v2", "v3")] for t in mesh.iter(f"{_CORE}triangle")], dtype=np.int64
            )
            parts.append(Part(vertices, triangles, quantity=quantity, name=member))
    return parts


def _synthetic() -> list[Part]:
    return [
        Part(*frame(130, 110, 6, 6), quantity=2, name="frame"),
        Part(*ring(110, 80, 6), quantity=2, name="ring"),
        Part(*u_shape(100, 80, 8, 6), quantity=2, name="u"),
        Part(*box(40, 30, 6), quantity=6, name="block"),
        Part(*box(12, 12, 6), quantity=40, name="peg"),
    ]


def test_synthetic_set_nests_and_is_fast():
    parts = _synthetic()
    started = time.perf_counter()
    r = pack(parts, Zone(256, 256), Options())
    elapsed = time.perf_counter() - started
    assert r.rejected == () and len(r.placements) == 52
    assert r.plate_count <= 2 and elapsed < 30
    assert min_gap(r, parts) >= 2.0
    assert r == pack(parts, Zone(256, 256), Options())


@pytest.mark.acceptance
def test_real_set_three_plates_under_a_minute():
    path = os.environ.get("PLATENEST_ACCEPTANCE_3MF")
    if not path:
        pytest.skip("PLATENEST_ACCEPTANCE_3MF is not set")
    parts = load_parts(path)
    assert sum(p.quantity for p in parts) == 90
    started = time.perf_counter()
    r = pack(parts, BIG_BED, Options(spacing=2.0))
    elapsed = time.perf_counter() - started
    print(f"\n{r.plate_count} plates, {elapsed:.1f} s")
    assert r.rejected == () and len(r.placements) == 90
    assert r.plate_count <= 3 and elapsed <= 60
    assert min_gap(r, parts) >= 2.0


def test_pack_result_writes_a_readable_project():
    parts = [Part(*frame(60, 60, 5, 4), name="frame"), Part(*box(20, 20, 3), quantity=3, name="lid")]
    r = pack(parts, Zone(256, 256), Options())
    data = write_project(
        [MeshObject(p.name, p.vertices, p.triangles) for p in parts],
        [Instance(p.part, p.plate, p.x, p.y, p.angle) for p in r.placements],
        bed_size=(256, 256),
        project_settings=b"{}",
    )
    assert len(read_project(data).instances) == 4
