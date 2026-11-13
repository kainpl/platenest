"""The four small findings the first review left open."""

import io
import zipfile
from xml.etree import ElementTree

import numpy as np
import pytest
from shapes import box

from platenest import Options, Part, Rejected, Zone, pack, validate
from platenest.bambu3mf import Instance, MeshObject, read_project, write_project
from platenest.zone import allowed_mask

TOUCH = Options(spacing=0, rotations=(0,), resolution=0.5, centre=False)

# --- a part never hangs over a zone edge, even one that is off the pixel grid ------------------


def test_the_partial_last_column_of_the_plate_is_not_usable():
    m = allowed_mask(Zone(10.1, 10), TOUCH)
    assert m.shape == (20, 21) and m[:, :20].all() and not m[:, 20].any()


def test_a_part_wider_than_an_off_grid_zone_is_too_large():
    r = pack([Part(*box(10.45, 5, 1))], Zone(10.1, 10), TOUCH)
    assert r.rejected == (Rejected(0, "too_large"),)
    assert pack([Part(*box(10, 5, 1))], Zone(10.1, 10), TOUCH).rejected == ()


def test_a_part_wider_than_an_off_grid_allowed_strip_is_too_large():
    strip = Zone(50, 50, allowed=[[(10, 0), (19.6, 0), (19.6, 50), (10, 50)]])
    assert pack([Part(*box(10, 5, 1))], strip, TOUCH).rejected == (Rejected(0, "too_large"),)
    assert pack([Part(*box(9.5, 5, 1))], strip, TOUCH).rejected == ()


def test_an_allowed_area_reaching_past_the_plate_is_cut_at_the_plate():
    m = allowed_mask(Zone(10, 10, allowed=[[(-5, -5), (30, -5), (30, 30), (-5, 30)]]), TOUCH)
    assert m.shape == (20, 20) and m.all()


def test_grid_aligned_edges_stay_exact():
    z = Zone(330, 320, allowed=[[(0, 0), (325, 0), (325, 320), (0, 320)]])
    m = allowed_mask(z, TOUCH)
    assert m[:, :650].all() and not m[:, 650:].any()


# --- numpy numbers are numbers ---------------------------------------------------------------


def test_numpy_integers_are_accepted_and_booleans_are_not():
    validate([Part(*box(1, 1, 1), quantity=np.int64(3))], Zone(10, 10), Options(fill_limit=np.int32(5)))
    assert len(pack([Part(*box(1, 1, 1), quantity=np.int64(3))], Zone(10, 10), Options()).placements) == 3
    with pytest.raises(ValueError, match="part 0"):
        validate([Part(*box(1, 1, 1), quantity=True)], Zone(10, 10), Options())
    with pytest.raises(ValueError):
        validate([Part(*box(1, 1, 1))], Zone(10, 10), Options(fill_limit=True))


def test_write_project_takes_numpy_numbers_in_instances():
    objs = [MeshObject("a", *box(5, 5, 5))]
    inst = [Instance(np.int64(0), np.int32(1), np.float64(1.5), np.float32(2.0), np.float64(90.0))]
    p = read_project(write_project(objs, inst, bed_size=(np.float64(100), 100), project_settings=b"{}"))
    assert p.instances == (Instance(0, 1, 1.5, 2.0, 90.0),) and p.bed_size == (100.0, 100.0)
    assert type(p.instances[0].object) is int and type(p.instances[0].x) is float


# --- names cannot break the XML --------------------------------------------------------------


def test_control_characters_in_names_do_not_break_the_file():
    objs = [MeshObject("a\x01b\tc\x00", *box(5, 5, 5))]
    data = write_project(objs, [Instance(0, 0, 0, 0, 0)], bed_size=(10, 10), project_settings=b"{}", title="t\x02\x0b")
    z = zipfile.ZipFile(io.BytesIO(data))
    ElementTree.fromstring(z.read("Metadata/model_settings.config"))
    ElementTree.fromstring(z.read("3D/3dmodel.model"))
    assert read_project(data).objects[0].name == "ab c"


# --- extra file names --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["metadata/platenest.json", "METADATA/Project_Settings.config", "3D/objects/part_0.model", "3d/3DMODEL.model",
     "C:/x", "Metadata/a\x00b", "Metadata/a\nb", "[content_types].xml"],
)  # fmt: skip
def test_extra_file_names_that_would_shadow_or_escape_are_refused(name):
    objs = [MeshObject("a", *box(5, 5, 5))]
    with pytest.raises(ValueError):
        write_project(objs, [Instance(0, 0, 0, 0, 0)], bed_size=(1, 1), project_settings=b"{}", extra_files={name: b""})
