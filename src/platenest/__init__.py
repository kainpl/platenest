"""platenest — dense nesting of 3D-print parts onto plates.

The packer lives here; the Bambu project 3MF writer is the separate, stdlib-only
``platenest.bambu3mf`` module and is not imported by this package.
"""

from platenest.types import Options, PackResult, Part, Placement, Polygon, Rejected, Zone, validate

__all__ = ["Options", "PackResult", "Part", "Placement", "Polygon", "Rejected", "Zone", "validate"]
