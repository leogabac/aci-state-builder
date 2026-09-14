# ACI State Builder

A desktop editor for creating and inspecting artificial colloidal ice states. The first version focuses on periodic square ice and the compact CSV format used by `icenumerics` and the historical `stuckgs` simulations.

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

- Generate rectangular periodic square lattices.
- Click a trap to flip its colloid; Shift/Ctrl-click selects without flipping.
- Rubber-band select traps and press `F` to flip the selection.
- Apply square-lattice polarized, 2-in/2-out, and 4-in/4-out configurations,
  or randomize all trap occupancies, from the context-aware sidebar.
- Undo and redo state changes.
- Import legacy state CSVs using Polars.
- Export either canonical `icenumerics` CSVs or round-trip legacy CSVs.
- Save richer, versioned `.aci.json` project files.
- Display periodic topological charges `q = N_in - N_out` as either nonzero-only
  or all-vertex overlays, with modulo wrapping across both PBC seams.
- Display live charge counts and validate states.
- Use a persistent right-hand parameter pane for lattice dimensions and spacing,
  trap separation/height/stiffness, particle properties, uniform-field spherical
  coordinates, optional pair cutoff, and live interaction energy.
- Pan with the scrollbars, zoom with the wheel, and press `0` to fit the lattice.

## Tests

```bash
PYTHONPATH=src rpy run python -m unittest discover -s tests -v
```

Compatibility tests read the surviving AF4 fixture from the adjacent `stuckgs` checkout when it is present.
