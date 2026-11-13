import io
import re
import zipfile

import pytest
from shapes import box, frame

from platenest.bambu3mf import Instance, MeshObject, read_project, write_project


def _project(n_plates=3):
    objs = [MeshObject("frame", *frame(60, 60, 5, 4)), MeshObject('lid "A" & <b>', *box(20, 20, 3))]
    inst = [
        Instance(0, 0, 10, 10, 0),
        Instance(1, 0, 25, 25, 90),
        Instance(1, 1, 5, 5, 0),
        Instance(1, n_plates - 1, 5, 5, 180),
    ]
    return objs, inst


def _zip(data: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(data))


def _plain_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buffer.getvalue()


def test_members_and_settings_pass_through():
    objs, inst = _project()
    z = _zip(write_project(objs, inst, bed_size=(256, 256), project_settings=b'{"a":1}'))
    assert {
        "[Content_Types].xml",
        "_rels/.rels",
        "3D/3dmodel.model",
        "3D/_rels/3dmodel.model.rels",
        "3D/Objects/part_0.model",
        "3D/Objects/part_1.model",
        "Metadata/model_settings.config",
        "Metadata/project_settings.config",
        "Metadata/platenest.json",
    } <= set(z.namelist())
    assert z.read("Metadata/project_settings.config") == b'{"a":1}'
    assert z.testzip() is None


def test_mesh_ids_are_unique_and_copies_share_a_mesh():
    objs, inst = _project()
    z = _zip(write_project(objs, inst, bed_size=(256, 256), project_settings=b"{}"))
    model = z.read("3D/3dmodel.model").decode()
    components = re.findall(r'objectid="(\d+)" p:UUID="[^"]+" transform="1 0 0', model)
    assert components == ["100001", "100002", "100002", "100002"]
    assert re.findall(r'<object id="(\d+)"', model) == ["2", "4", "6", "8"]
    assert 'id="100001"' in z.read("3D/Objects/part_0.model").decode()
    assert 'id="100002"' in z.read("3D/Objects/part_1.model").decode()
    rels = z.read("3D/_rels/3dmodel.model.rels").decode()
    assert rels.count("<Relationship ") == 2


def test_item_transform_and_plate_grid():
    objs, inst = _project()
    model = _zip(write_project(objs, inst, bed_size=(256, 200), project_settings=b"{}")).read("3D/3dmodel.model")
    t = re.findall(r'<item objectid="\d+" p:UUID="[^"]+" transform="([^"]+)"', model.decode())
    assert t[0] == "1.000000000 0.000000000 0 0.000000000 1.000000000 0 0 0 1 10.00000 10.00000 0"
    assert t[1] == "0.000000000 1.000000000 0 -1.000000000 0.000000000 0 0 0 1 25.00000 25.00000 0"
    assert t[2].endswith(f" {256 * 1.2 + 5:.5f} 5.00000 0")  # plate 1: second column
    assert t[3].endswith(f" 5.00000 {-200 * 1.2 + 5:.5f} 0")  # plate 2: second row (cols = 2)
    assert 'printable="1"' in model.decode()


def test_model_settings_plates():
    objs, inst = _project()
    z = _zip(write_project(objs, inst, bed_size=(256, 256), project_settings=b"{}"))
    cfg = z.read("Metadata/model_settings.config").decode()
    assert re.findall(r'key="plater_id" value="(\d+)"', cfg) == ["1", "2", "3"]
    assert cfg.count("<model_instance>") == 4 and 'value="lid &quot;A&quot; &amp; &lt;b&gt;"' in cfg
    assert re.findall(r'key="object_id" value="(\d+)"', cfg) == ["2", "4", "6", "8"]
    assert re.findall(r'key="identify_id" value="(\d+)"', cfg) == ["1002", "1004", "1006", "1008"]


def test_a_plate_without_instances_still_exists():
    objs, _ = _project()
    data = write_project(objs, [Instance(0, 2, 1, 1, 0)], bed_size=(256, 256), project_settings=b"{}")
    cfg = _zip(data).read("Metadata/model_settings.config").decode()
    assert re.findall(r'key="plater_id" value="(\d+)"', cfg) == ["1", "2", "3"]


def test_output_is_byte_identical():
    objs, inst = _project()
    kw = dict(bed_size=(256, 256), project_settings=b"{}")
    assert write_project(objs, inst, **kw) == write_project(objs, inst, **kw)


def test_round_trip():
    objs, inst = _project()
    data = write_project(
        objs, inst, bed_size=(256, 200), project_settings=b'{"k":2}', extra_files={"Metadata/inputs.json": b"{}"}
    )
    p = read_project(data)
    assert p.instances == tuple(inst) and p.bed_size == (256, 200) and p.project_settings == b'{"k":2}'
    assert [o.name for o in p.objects] == [o.name for o in objs]
    assert len(p.objects[0].triangles) == len(objs[0].triangles)
    assert len(p.objects[0].vertices) == len(objs[0].vertices)
    assert [round(c, 5) for c in p.objects[1].vertices[6]] == [20.0, 20.0, 3.0]
    assert p.extra_files == {"Metadata/inputs.json": b"{}"}


def test_unused_object_is_not_written_but_indices_survive():
    objs, _ = _project()
    data = write_project(objs, [Instance(1, 0, 0, 0, 0)], bed_size=(100, 100), project_settings=b"{}")
    assert "3D/Objects/part_0.model" not in _zip(data).namelist()
    p = read_project(data)
    assert p.instances == (Instance(1, 0, 0, 0, 0),) and p.objects[1].name == objs[1].name
    assert p.objects[0].name == "frame" and len(p.objects[0].vertices) == 0


@pytest.mark.parametrize("name", ["3D/3dmodel.model", "Metadata/platenest.json", "../evil", "/abs", "a/../b", ""])
def test_bad_extra_file_names(name):
    objs, inst = _project()
    with pytest.raises(ValueError):
        write_project(objs, inst, bed_size=(1, 1), project_settings=b"{}", extra_files={name: b""})


def test_read_rejects_foreign_files():
    with pytest.raises(ValueError, match="not a platenest project"):
        read_project(_plain_zip({"3D/3dmodel.model": b"<model/>"}))
    with pytest.raises(ValueError, match="not a platenest project"):
        read_project(b"this is not a zip")


@pytest.mark.parametrize("instance", [Instance(7, 0, 0, 0, 0), Instance(0, -1, 0, 0, 0), Instance(-1, 0, 0, 0, 0)])
def test_bad_instance(instance):
    objs, _ = _project()
    with pytest.raises(ValueError):
        write_project(objs, [instance], bed_size=(1, 1), project_settings=b"{}")


def test_read_refuses_a_document_type_declaration():
    objs, inst = _project()
    members = {
        n: _zip(write_project(objs, inst, bed_size=(1, 1), project_settings=b"{}")).read(n)
        for n in _zip(write_project(objs, inst, bed_size=(1, 1), project_settings=b"{}")).namelist()
    }
    bomb = b'<?xml version="1.0"?><!DOCTYPE m [<!ENTITY a "aaaa">]><model>&a;</model>'
    members["3D/Objects/part_0.model"] = bomb
    with pytest.raises(ValueError, match="not a platenest project"):
        read_project(_plain_zip(members))
