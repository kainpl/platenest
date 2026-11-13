# Changelog

## [Unreleased]

## [0.1.0] - 2026-11-13

The first release.

- **The packer** (`from platenest import pack`). Parts are placed by their real top-down outline, so a small part
  goes into the hole of a frame, the mouth of a U or between the arms of a bracket. You give meshes, quantities, a
  work zone (plate size, areas parts may or may not be in, a maximum height) and a spacing; you get a plate index,
  an offset and a rotation for every copy. Six combinations of ordering and direction are tried and the one with the
  fewest plates is kept.
- **The spacing you ask for is the least you get** between any two parts, measured on the real outlines; half of it
  is kept to the edge of the work zone and to excluded areas.
- **Fill mode**: a part marked `fill` gets a plate of its own with as many copies as fit, and the result says how
  many that was.
- **Nothing is dropped silently.** A part with no top-down area, one taller than the zone allows or one too large
  for an empty plate comes back in `rejected` with its reason, and the rest is still packed. A mesh in the wrong
  units is refused at once instead of being rasterised first.
- The group on each plate can be centred in the work zone; a callback reports progress, another cancels a run;
  `render_plate` draws a top view of a plate.
- **`platenest.bambu3mf`**, a separate module that needs only the standard library: `write_project` turns meshes,
  placements and your own project settings into a multi-plate Bambu Studio / OrcaSlicer project 3MF that slices
  without anything being re-arranged, and `read_project` reads such a file back. A file that is not one of ours,
  is damaged, or unpacks larger than a limit you set is a `ValueError`.
- The same input gives the same placements and a byte-identical 3MF.
- No printer or slicer data is bundled: the work zone and the settings are always yours.
