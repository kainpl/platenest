"""Write a Bambu-format project 3MF from meshes, placements and caller-supplied settings — and read it back.

Standard library only: this module imports neither numpy nor OpenCV, and the packer does not import it.
Mesh arguments are any sequences of 3-item sequences, so numpy arrays work as they are.

No printer or slicer data lives here. ``project_settings`` is written verbatim as
``Metadata/project_settings.config``; producing it is the caller's business.

An :class:`Instance` follows the packer's placement contract: the object's own vertices transformed by
``p' = Rz(angle) · p + (x, y, 0)`` in the coordinates of its plate (origin front-left, millimetres,
degrees counter-clockwise seen from above).

Output is deterministic: the same arguments give the same bytes.
"""

from __future__ import annotations

import io
import json
import math
import uuid
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from xml.etree import ElementTree
from xml.sax.saxutils import escape

_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_PRODUCTION_NS = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
_BAMBU_NS = "http://schemas.bambulab.com/package/2021"
_MODEL_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/kainpl/platenest")
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)
_MESH_ID_BASE = 100001  # mesh object ids must be unique across the whole archive, not per sub-model
_PLATE_STRIDE = 1.2  # plates lie on a grid, this many bed sizes apart

_CONTENT_TYPES = "[Content_Types].xml"
_ROOT_RELS = "_rels/.rels"
_MODEL = "3D/3dmodel.model"
_MODEL_RELS = "3D/_rels/3dmodel.model.rels"
_MODEL_SETTINGS = "Metadata/model_settings.config"
_PROJECT_SETTINGS = "Metadata/project_settings.config"
_RECORD = "Metadata/platenest.json"
_PART_PREFIX = "3D/Objects/part_"
DEFAULT_MAX_BYTES = 256 * 1024 * 1024  # unpacked size read_project accepts unless told otherwise
_FIXED = frozenset({_CONTENT_TYPES, _ROOT_RELS, _MODEL, _MODEL_RELS, _MODEL_SETTINGS, _PROJECT_SETTINGS, _RECORD})
_XML = '<?xml version="1.0" encoding="UTF-8"?>\n'


@dataclass(frozen=True)
class MeshObject:
    name: str
    vertices: Sequence[Sequence[float]]
    triangles: Sequence[Sequence[int]]


@dataclass(frozen=True)
class Instance:
    object: int  # index into objects
    plate: int  # 0-based
    x: float
    y: float
    angle: float


@dataclass(frozen=True)
class Project:
    objects: tuple[MeshObject, ...]
    instances: tuple[Instance, ...]
    bed_size: tuple[float, float]
    project_settings: bytes
    extra_files: Mapping[str, bytes]


def _uuid(kind: str, index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{kind}:{index}"))


def _attr(text: str) -> str:
    return '"' + escape(text, {'"': "&quot;"}) + '"'


def _part_path(index: int) -> str:
    return f"{_PART_PREFIX}{index}.model"


def _unit(value: float) -> str:
    text = f"{value:.9f}"
    return "0.000000000" if text == "-0.000000000" else text


def _mesh_model(mesh: MeshObject, index: int) -> str:
    vertices = "".join(f'<vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>' for x, y, z in mesh.vertices)
    triangles = "".join(f'<triangle v1="{int(a)}" v2="{int(b)}" v3="{int(c)}"/>' for a, b, c in mesh.triangles)
    return (
        f'{_XML}<model unit="millimeter" xml:lang="en-US" xmlns="{_CORE_NS}" xmlns:p="{_PRODUCTION_NS}" '
        f'requiredextensions="p"><resources><object id="{_MESH_ID_BASE + index}" p:UUID="{_uuid("mesh", index)}" '
        f'type="model"><mesh><vertices>{vertices}</vertices><triangles>{triangles}</triangles></mesh></object>'
        "</resources></model>"
    )


def _check_extra(extra_files: Mapping[str, bytes]) -> None:
    for name in extra_files:
        parts = name.split("/")
        if not name or name.startswith("/") or "\\" in name or any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"extra file name {name!r} must be a relative path without '..'")
        if name in _FIXED or name.startswith(_PART_PREFIX):
            raise ValueError(f"extra file name {name!r} collides with a member the writer owns")


def write_project(
    objects: Sequence[MeshObject],
    instances: Sequence[Instance],
    *,
    bed_size: tuple[float, float],
    project_settings: bytes,
    title: str = "",
    application: str = "BambuStudio-02.05.00.66",
    extra_files: Mapping[str, bytes] | None = None,
) -> bytes:
    """Serialise a project. Plates are numbered from 0; the highest plate index used decides how many exist."""
    extra_files = dict(extra_files or {})
    _check_extra(extra_files)
    for number, instance in enumerate(instances):
        if not 0 <= instance.object < len(objects):
            raise ValueError(f"instance {number}: object index {instance.object} is out of range")
        if instance.plate < 0:
            raise ValueError(f"instance {number}: plate index must be >= 0")

    plate_count = max((i.plate for i in instances), default=0) + 1
    columns = math.ceil(math.sqrt(plate_count))
    used = sorted({i.object for i in instances})

    resources, items, settings = [], [], []
    plate_instances: list[list[str]] = [[] for _ in range(plate_count)]
    for number, instance in enumerate(instances):
        object_id = 2 * (number + 1)
        name = objects[instance.object].name
        angle = math.radians(instance.angle)
        c, s = math.cos(angle), math.sin(angle)
        tx = (instance.plate % columns) * bed_size[0] * _PLATE_STRIDE + instance.x
        ty = -(instance.plate // columns) * bed_size[1] * _PLATE_STRIDE + instance.y
        resources.append(
            f'<object id="{object_id}" p:UUID="{_uuid("object", number)}" type="model"><components>'
            f'<component p:path="/{_part_path(instance.object)}" objectid="{_MESH_ID_BASE + instance.object}" '
            f'p:UUID="{_uuid("component", number)}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/></components></object>'
        )
        items.append(  # rows are the images of the basis vectors: Rz = "c s 0 / -s c 0 / 0 0 1"
            f'<item objectid="{object_id}" p:UUID="{_uuid("item", number)}" '
            f'transform="{_unit(c)} {_unit(s)} 0 {_unit(-s)} {_unit(c)} 0 0 0 1 {tx:.5f} {ty:.5f} 0" printable="1"/>'
        )
        settings.append(
            f'<object id="{object_id}"><metadata key="name" value={_attr(name)}/>'
            f'<metadata key="extruder" value="1"/><part id="{object_id - 1}" subtype="normal_part">'
            f'<metadata key="name" value={_attr(name)}/>'
            '<metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/></part></object>'
        )
        plate_instances[instance.plate].append(
            f'<model_instance><metadata key="object_id" value="{object_id}"/>'
            '<metadata key="instance_id" value="0"/>'
            f'<metadata key="identify_id" value="{1000 + object_id}"/></model_instance>'
        )

    plates = "".join(
        f'<plate><metadata key="plater_id" value="{number + 1}"/><metadata key="plater_name" value=""/>'
        f'<metadata key="locked" value="false"/>{"".join(entries)}</plate>'
        for number, entries in enumerate(plate_instances)
    )
    model = (
        f'{_XML}<model unit="millimeter" xml:lang="en-US" xmlns="{_CORE_NS}" xmlns:BambuStudio="{_BAMBU_NS}" '
        f'xmlns:p="{_PRODUCTION_NS}" requiredextensions="p">'
        f'<metadata name="Application">{escape(application)}</metadata>'
        '<metadata name="BambuStudio:3mfVersion">1</metadata>'
        f'<metadata name="Title">{escape(title)}</metadata>'
        f"<resources>{''.join(resources)}</resources>"
        f'<build p:UUID="{_uuid("build", 0)}">{"".join(items)}</build></model>'
    )
    relationships = "".join(
        f'<Relationship Target="/{_part_path(index)}" Id="rel-{index + 1}" Type="{_MODEL_REL}"/>' for index in used
    )
    record = {
        "version": 1,
        "bed_size": list(bed_size),
        "objects": [mesh.name for mesh in objects],
        "instances": [{"object": i.object, "plate": i.plate, "x": i.x, "y": i.y, "angle": i.angle} for i in instances],
    }

    members: list[tuple[str, bytes]] = [
        (
            _CONTENT_TYPES,
            (
                f'{_XML}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
                '<Default Extension="png" ContentType="image/png"/>'
                '<Default Extension="gcode" ContentType="text/x.gcode"/></Types>'
            ).encode(),
        ),
        (
            _ROOT_RELS,
            (
                f'{_XML}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f'<Relationship Target="/{_MODEL}" Id="rel-1" Type="{_MODEL_REL}"/></Relationships>'
            ).encode(),
        ),
        (_MODEL, model.encode()),
        (
            _MODEL_RELS,
            (
                f'{_XML}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f"{relationships}</Relationships>"
            ).encode(),
        ),
    ]
    members += [(_part_path(index), _mesh_model(objects[index], index).encode()) for index in used]
    members += [
        (_MODEL_SETTINGS, f"{_XML}<config>{''.join(settings)}{plates}</config>".encode()),
        (_PROJECT_SETTINGS, bytes(project_settings)),
        (_RECORD, json.dumps(record, sort_keys=True, ensure_ascii=False).encode()),
    ]
    members += sorted((name, bytes(data)) for name, data in extra_files.items())

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members:
            info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3  # the same on every platform, or the bytes would differ
            archive.writestr(info, data)
    return buffer.getvalue()


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _read_mesh(data: bytes) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[int, int, int], ...]]:
    # The file may come from anywhere. A mesh model needs no document type declaration, and refusing
    # one closes entity expansion and external entities without a third-party parser.
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise ValueError("a mesh model carries a document type declaration")
    root = ElementTree.fromstring(data)
    vertices = tuple(
        (float(v.get("x")), float(v.get("y")), float(v.get("z"))) for v in root.iter(f"{{{_CORE_NS}}}vertex")
    )
    triangles = tuple(
        (int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))) for t in root.iter(f"{{{_CORE_NS}}}triangle")
    )
    return vertices, triangles


def _read_record(data: bytes) -> tuple[tuple[Instance, ...], tuple[float, float], list[str]]:
    record = json.loads(data)
    if not isinstance(record, dict) or record.get("version") != 1:
        raise ValueError("the record is not version 1")
    names, bed, entries = record.get("objects"), record.get("bed_size"), record.get("instances")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("the record's object names are not a list of strings")
    if not isinstance(bed, list) or len(bed) != 2 or not all(_number(v) and v > 0 for v in bed):
        raise ValueError("the record's bed size is not two positive numbers")
    if not isinstance(entries, list):
        raise ValueError("the record's instances are not a list")
    instances = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("an instance is not an object")
        obj, plate = entry.get("object"), entry.get("plate")
        if not _index(obj) or obj >= len(names) or not _index(plate):
            raise ValueError("an instance names an object or a plate that does not exist")
        x, y, angle = entry.get("x"), entry.get("y"), entry.get("angle")
        if not (_number(x) and _number(y) and _number(angle)):
            raise ValueError("an instance position is not numeric")
        instances.append(Instance(obj, plate, x, y, angle))
    return tuple(instances), (bed[0], bed[1]), names


def read_project(data: bytes, *, max_bytes: int = DEFAULT_MAX_BYTES) -> Project:
    """Read back a file :func:`write_project` produced.

    Anything else — a foreign 3MF, a damaged or truncated archive, a record that does not add up —
    is a ``ValueError`` starting with "not a platenest project". The archive's declared unpacked
    size is checked against ``max_bytes`` before any member is read.

    An object no instance uses was never written, so it comes back with its name and no geometry.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            names = {info.filename for info in infos}
            if _RECORD not in names:
                raise ValueError("Metadata/platenest.json is missing")
            unpacked = sum(info.file_size for info in infos)
            if unpacked > max_bytes:
                raise ValueError(f"too large: unpacks to {unpacked} bytes, the limit is {max_bytes}")
            instances, bed_size, object_names = _read_record(archive.read(_RECORD))
            objects = []
            for index, name in enumerate(object_names):
                path = _part_path(index)
                vertices, triangles = _read_mesh(archive.read(path)) if path in names else ((), ())
                objects.append(MeshObject(name, vertices, triangles))
            extra = {
                name: archive.read(name)
                for name in sorted(names)
                if name not in _FIXED and not name.startswith(_PART_PREFIX) and not name.endswith("/")
            }
            settings = archive.read(_PROJECT_SETTINGS) if _PROJECT_SETTINGS in names else b""
    except ValueError as error:  # ours, and json.JSONDecodeError / UnicodeDecodeError, which are ValueErrors
        raise ValueError(f"not a platenest project: {error}") from None
    except (zipfile.BadZipFile, zlib.error, ElementTree.ParseError, TypeError, RecursionError, EOFError) as error:
        raise ValueError(f"not a platenest project: damaged archive ({type(error).__name__})") from None
    return Project(tuple(objects), instances, bed_size, settings, extra)
