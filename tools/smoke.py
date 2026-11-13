"""Run against an installed wheel, outside the checkout: the package imports and packs something."""

import sys

import platenest.bambu3mf as bambu3mf

assert "numpy" not in sys.modules and "cv2" not in sys.modules, "the 3MF module must stay stdlib-only"

import numpy as np  # noqa: E402

from platenest import Options, Part, Zone, pack  # noqa: E402

vertices = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=float)
triangles = np.array([[0, 1, 2], [0, 2, 3]])
result = pack([Part(vertices, triangles, quantity=3, name="tile")], Zone(100, 100), Options())
assert result.plate_count == 1 and len(result.placements) == 3, result
data = bambu3mf.write_project(
    [bambu3mf.MeshObject("tile", vertices, triangles)],
    [bambu3mf.Instance(p.part, p.plate, p.x, p.y, p.angle) for p in result.placements],
    bed_size=(100, 100),
    project_settings=b"{}",
)
assert len(bambu3mf.read_project(data).instances) == 3
print("ok:", result.plate_count, "plate,", len(data), "bytes")
