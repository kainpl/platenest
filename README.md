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
- **`spacing=0`** lets parts touch. Axis-aligned edges meet exactly; slanted edges are judged on the pixel
  grid and may overlap by up to about a tenth of a millimetre. Ask for a small spacing if that matters.
- **Deterministic.** The same input gives the same placements and a byte-identical 3MF.
- `progress=` receives a `Progress` after every placed copy; `cancel=` is polled before every copy and
  makes `pack` raise `Cancelled`.
- `render_plate(...)` returns a top-view image of one plate as a numpy array.

Speed, as a reference point: 90 copies of 15 real parts on three 325 x 320 mm plates take about 15 s on
a desktop CPU.

## Optional longer search

`pip install "platenest[optimize]"` adds Shapely and the ready-made Sparrow
optimizer through `spyrrow`. The ordinary `pack()` call and the stdlib-only
3MF module keep their existing dependencies and deterministic behavior.

Free-angle search can recover an initial finite-angle rejection using bounded additional seeds; a rejected
search is not a proof that no continuous-angle solution exists. Proven height/area impossibilities remain
fast rejections. Recoverable native `ValueError` trials keep the last original-contour-validated layout and
report `SearchProgress.diagnostic`; cancellation and resource errors remain failures, not successful results.
Restricted and continuous native trials alternate independently for each active group.

```python
from platenest import optimize

operator_requested_best = False  # the caller may set this from its UI
best = optimize(
    parts,
    zone,
    Options(spacing=2),
    seconds=300,
    free_rotation=True,
    progress=lambda event: print(len(event.best.placements)),
    finish=lambda: operator_requested_best,
)
```

The search starts with a checked bitmap layout, tries different rotation
preferences, then keeps improvements from Sparrow. `free_rotation=True`
permits any rotation around Z; otherwise `Options.rotations` remains binding.
Models are never mirrored, resized or tilted. Work-zone exclusions and actual
projected mesh contours are rechecked independently before accepting a layout.
Frame holes remain available to the bitmap packer; Sparrow's Python wrapper
conservatively treats an item's holes as solid. An unsupported or unsuccessful
candidate never replaces the existing result.

Fixed demands stay complete and aim for fewer plates. The strip result is
conservatively divided into finite plates without cutting or dropping any part;
every resulting plate is checked against the original work zone. Each fill part retains
its own plate and cannot lose copies to improve another fill part. The group
returned by the strip optimizer is rigidly rotated/repositioned to the work
zone before checking it; a reported strip width alone is not a fit decision.

`finish` cooperatively returns the best known result. `cancel` raises
`Cancelled`, as it does for `pack`. Native solver attempts last up to 15 seconds
by default (`attempt_seconds`), so stopping is checked between attempts and
native cleanup can add a few seconds. `seconds` is a soft total budget, including
geometry preparation; use an existing process supervisor for hard termination.
The default 5-minute limit is a search budget, not an optimality proof. A seed
is recorded by the caller; wall-clock-limited search does not promise identical
results on different hardware. Preserve the returned placements to reproduce
the chosen project.

The optional module reuses Sparrow and Shapely instead of introducing a new
nesting engine, geometry parser, scheduler or persistence layer. SVGnest and
libnest2d were considered: SVGnest's JS/NFP integration was more expensive in
the exploratory trials, while libnest2d's documented default geometry support
does not fully cover holes/concavities. Neither is substituted for the existing
bitmap path.

## The 3MF module

`write_project` produces a multi-plate project the slicer opens and slices without re-arranging anything.
`project_settings` is written verbatim as `Metadata/project_settings.config`; a complete settings file is
your responsibility — the slicer's command line does not fill in missing keys. `extra_files` lets you
carry your own records inside the archive. `read_project` reads back what `write_project` wrote (it relies
on a small `Metadata/platenest.json` record) and answers `ValueError` for anything else — a foreign 3MF, a
damaged archive, or one that unpacks to more than `max_bytes`.

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
