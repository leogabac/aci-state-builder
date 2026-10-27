# ACI State Builder

A desktop editor for creating and inspecting artificial colloidal ice states, using the compact CSV format from `icenumerics` and the historical `stuckgs` simulations.

## Run

This checkout selects the shared `rpy` environment named `aci`.

```bash
PYTHONPATH=src rpy run python -m aci_state_builder.app
```

For development, install the project into the selected environment once:

```bash
rpy pip install -e .
rpy run aci-state-builder
```

## First-version features

- Generate square, honeycomb/kagome-ice, triangular, and height-offset square
  lattices, with periodic or finite closed-spin/closed-vertex boundaries where
  the geometry supports them.
- Click a trap to flip its colloid; Shift/Ctrl-click selects without flipping.
- Rubber-band select traps and press `F` to flip the selection.
- Apply square-lattice polarized, 2-in/2-out, and 4-in/4-out configurations,
  or randomize all trap occupancies, from the context-aware sidebar.
- Undo and redo state changes.
- Import legacy state CSVs using Polars.
- Open multi-frame CSV trajectories without loading the whole file, using a
  reusable external frame index and a user-controlled bounded chunk cache.
- Scrub or play trajectories, choose frame stride and playback speed, prefetch
  upcoming frames, and extract the current frame into an editable state.
- Mark trajectory frames A and B for a lightweight visual comparison of
  particle displacements and flipped traps, with direct jumps back to either
  reference frame.
- Open marked frames in a draggable side-by-side workspace with linked
  navigation, optionally linked per-view overlays, corresponding-trap hover
  highlighting, and a compact qualitative change summary.
- Export either canonical `icenumerics` CSVs or round-trip legacy CSVs.
- Save richer, versioned `.aci.json` project files.
- Display topology-aware charges `q = N_in - N_out` as either nonzero-only or
  all-vertex overlays, including odd coordination on honeycomb ice and oblique
  periodic seams.
- Display live charge counts and validate states.
- Independently toggle trap bodies, particles, IDs, periodic boundaries,
  charges, charge legend, and a zoom-aware physical scale bar.
- Show short particle trails from frames already held in the bounded cache;
  right-click a trap or particle to inspect and copy its coordinates, state,
  displacement, and imported metadata.
- Switch to a canvas-only presentation mode, copy the canvas to the clipboard,
  or export the current view (and either side of a comparison) as scaled PNG or
  SVG with the active visual overlays.
- Export trajectories directly to H.264 MP4 at a chosen FPS, frame stride,
  resolution, quality, and TikZ-inspired paper, light, or dark presentation
  style, with circle or filled-cell vertex coloring, a palette selector,
  optional content-fitted output dimensions, trails, cancellable progress,
  and no temporary frame images.
- Use a persistent right-hand parameter pane for lattice dimensions and spacing,
  trap separation/height/stiffness, particle properties, uniform-field spherical
  coordinates, optional pair cutoff, and live interaction energy.
- Pan naturally with a middle-button drag, zoom around the pointer with the
  wheel, use the toolbar for explicit zoom controls, and press `0` to fit the
  lattice.

Trajectory frame indexes are stored under the user's cache directory rather
than beside the simulation data. The first opening scans a CSV once; later
openings seek directly to the required frame blocks. The reader, cache, and UI
are deliberately separated so a native Parquet backend can be added later
without changing trajectory playback or analysis controls.

Video export requires `ffmpeg` on `PATH`; no Python video package is required.
The TikZ-inspired style uses `latex` and `dvisvgm` for genuine TeX typography
when available, caches the resulting SVG labels, and falls back to bundled
desktop fonts when either command is missing.

## Tests

```bash
PYTHONPATH=src rpy run python -m unittest discover -s tests -v
```

Compatibility tests read the surviving AF4 fixture from the adjacent `stuckgs` checkout when it is present.
