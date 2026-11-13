"""platenest — dense nesting of 3D-print parts onto plates.

The packer lives here; the Bambu project 3MF writer is the separate, stdlib-only
``platenest.bambu3mf`` module and is not imported by this package.
"""

from platenest.pack import Cancelled, Progress, pack
from platenest.preview import render_plate
from platenest.types import Options, PackResult, Part, Placement, Polygon, Rejected, Zone, validate

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
