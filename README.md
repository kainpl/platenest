# platenest

Dense nesting of 3D-print parts onto build plates. Parts are placed by their real outline, not their
bounding box, so a small part ends up inside the hole of a frame, in the mouth of a U, between the arms of
a bracket — the places a slicer's own auto-arrange leaves empty.

Two independent modules:

- **`platenest`** — the packer. Meshes, quantities, a work zone and a spacing go in; plate index, offset
  and rotation for every copy come out. Needs `numpy` and `opencv-python-headless`.
- **`platenest.bambu3mf`** — writes meshes + placements + your slicer settings as a Bambu Studio /
  OrcaSlicer project 3MF, and reads such a file back. Standard library only.

No printer or slicer data is bundled. You pass your own work zone (bed size, areas to keep clear, maximum
height) and your own settings; the library has no opinion about which machine you own.

## Install

```
pip install platenest
```

Python 3.12 or newer.

## Example

```python
import trimesh  # any mesh loader will do; platenest reads no files itself
from platenest import Options, Part, Zone, pack
from platenest.bambu3mf import Instance, MeshObject, write_project

meshes = {name: trimesh.load(f"{name}.stl") for name in ("jar", "lid")}
parts = [
    Part(meshes["jar"].vertices, meshes["jar"].faces, quantity=6, name="jar"),
    Part(meshes["lid"].vertices, meshes["lid"].faces, fill=True, name="lid"),  # its own plate, as many as fit
]
zone = Zone(256, 256, excluded=[[(0, 0), (18, 0), (18, 28), (0, 28)]], max_height=250)

result = pack(parts, zone, Options(spacing=2.0, rotations=(0, 90, 180, 270)))
print(result.plate_count, "plates;", result.fill_counts, "copies on fill plates;", result.rejected)

project = write_project(
    [MeshObject(p.name, p.vertices, p.triangles) for p in parts],
    [Instance(p.part, p.plate, p.x, p.y, p.angle) for p in result.placements],
    bed_size=(256, 256),
    project_settings=open("project_settings.config", "rb").read(),  # yours
)
open("plates.3mf", "wb").write(project)
```

## What the packer does

1. Projects each part top-down into a bitmap (0.5 mm per pixel by default), once per allowed rotation.
2. Grows the bitmap by the spacing, plus one pixel so that rounding never shrinks the real gap below what
   you asked for.
3. Finds every free position of a part on a plate in one correlation of that bitmap with the plate's
   occupancy, and takes the one closest to the front / left corner.
4. Places parts greedily, largest first, opening a new plate when nothing fits; repeats this for six
   combinations of ordering and direction and keeps the result with the fewest plates.
5. Optionally moves each plate's group to the centre of the work zone.

Behaviour worth knowing:

- **Placement contract.** A placed copy is the part's own vertices transformed by
  `p' = Rz(angle) · p + (x, y, 0)` — plate origin at the front-left corner, X to the right, Y to the back,
  millimetres, degrees counter-clockwise seen from above. Z is never changed: put your mesh on the bed
  (`z_min = 0`) before packing.
- **Spacing** is kept between parts in full and to the edge of the work zone and to excluded areas in
  half, so a brim stays inside.
- **`fill=True`** gives the part a plate of its own with as many copies as fit (capped by
  `Options.fill_limit`; `PackResult.fill_limited` tells you when the cap was what stopped it). Fill plates
  come after the counted ones and are never mixed with other parts.
- **Rejections** do not stop the run: a part with no top-down area (`empty_mesh`), taller than
  `Zone.max_height` (`too_tall`) or too big for an empty plate in every rotation (`too_large`) is reported
  in `PackResult.rejected` and the rest is packed.
- **Deterministic.** The same input gives the same placements and a byte-identical 3MF.
- `progress=` receives a `Progress` after every placed copy; `cancel=` is polled before every copy and
  makes `pack` raise `Cancelled`.
- `render_plate(...)` returns a top-view image of one plate as a numpy array.

Speed, as a reference point: 90 copies of 15 real parts on three 325 x 320 mm plates take about 15 s on
a desktop CPU.

## The 3MF module

`write_project` produces a multi-plate project the slicer opens and slices without re-arranging anything.
`project_settings` is written verbatim as `Metadata/project_settings.config`; a complete settings file is
your responsibility — the slicer's command line does not fill in missing keys. `extra_files` lets you
carry your own records inside the archive. `read_project` reads back what `write_project` wrote (it relies
on a small `Metadata/platenest.json` record) and refuses anything else.

## Development

```
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

All test fixtures are generated in code. To run the acceptance check against real meshes, point
`PLATENEST_ACCEPTANCE_3MF` at a Bambu project 3MF and run `pytest -m acceptance`.

## Licence and origin

AGPL-3.0-or-later. The implementation is original: it was written from a description of the
bitmap-nesting idea and is derived from no other packer's source code.
