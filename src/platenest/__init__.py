"""platenest — dense nesting of 3D-print parts onto plates.

The packer is exposed here: ``from platenest import pack, Part, Zone, Options``.
The Bambu project 3MF writer is the separate, stdlib-only ``platenest.bambu3mf`` module.

Names are resolved on first use, so ``import platenest.bambu3mf`` alone pulls in neither numpy nor OpenCV.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from platenest.pack import Cancelled, Progress, pack
    from platenest.preview import render_plate
    from platenest.types import Options, PackResult, Part, Placement, Polygon, Rejected, Zone, validate

_HOME = {
    "Cancelled": "pack",
    "Progress": "pack",
    "pack": "pack",
    "render_plate": "preview",
    "Options": "types",
    "PackResult": "types",
    "Part": "types",
    "Placement": "types",
    "Polygon": "types",
    "Rejected": "types",
    "Zone": "types",
    "validate": "types",
}

__all__ = [
    "Cancelled",
    "Options",
    "PackResult",
    "Part",
    "Placement",
    "Polygon",
    "Progress",
    "Rejected",
    "Zone",
    "pack",
    "render_plate",
    "validate",
]


def __getattr__(name: str):
    module = _HOME.get(name)
    if module is None:
        raise AttributeError(f"module 'platenest' has no attribute {name!r}")
    value = getattr(import_module(f"platenest.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
