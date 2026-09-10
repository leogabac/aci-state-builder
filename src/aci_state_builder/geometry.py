from __future__ import annotations

import numpy as np

from .model import IceDocument, TrapState


def periodic_square(nx: int, ny: int, lattice_constant: float, trap_separation: float) -> IceDocument:
    if nx < 1 or ny < 1:
        raise ValueError("lattice dimensions must be positive")
    if lattice_constant <= 0 or trap_separation <= 0:
        raise ValueError("lengths must be positive")
    traps: list[TrapState] = []
    trap_id = 0
    # Keep the historical icenumerics ordering: all horizontal traps, then vertical.
    for y in range(ny):
        for x in range(nx):
            traps.append(TrapState(trap_id, [(x + 0.5) * lattice_constant, y * lattice_constant, 0], [1, 0, 0]))
            trap_id += 1
    for y in range(ny):
        for x in range(nx):
            traps.append(TrapState(trap_id, [x * lattice_constant, (y + 0.5) * lattice_constant, 0], [0, 1, 0]))
            trap_id += 1
    return IceDocument(
        traps=traps, geometry="square", boundary="periodic", nx=nx, ny=ny,
        lattice_constant=float(lattice_constant), trap_separation=float(trap_separation),
        name=f"Square {nx} x {ny}",
    )


def periodic_vertex_charges(document: IceDocument) -> np.ndarray | None:
    """Return q = N_in - N_out at each primary-cell vertex.

    Neighbor indices wrap modulo ``nx`` and ``ny``. A spin pointing along its
    positive axis therefore contributes -1 where it leaves and +1 where it
    enters, including when that edge crosses a periodic seam.
    """
    if (document.geometry != "square" or document.boundary != "periodic"
            or document.nx is None or document.ny is None
            or len(document.traps) != 2 * document.nx * document.ny):
        return None
    nx, ny = document.nx, document.ny
    charge = np.zeros((ny, nx), dtype=np.int8)
    for y in range(ny):
        for x in range(nx):
            horizontal = document.traps[y * nx + x].occupancy
            charge[y, x] -= horizontal
            charge[y, (x + 1) % nx] += horizontal
            vertical = document.traps[nx * ny + y * nx + x].occupancy
            charge[y, x] -= vertical
            charge[(y + 1) % ny, x] += vertical
    return charge


def periodic_vertex_data(document: IceDocument) -> list[tuple[int, int, float, float, int]]:
    """Return ``(ix, iy, x, y, q)`` for vertices in the primary periodic cell."""
    charges = periodic_vertex_charges(document)
    if charges is None or document.lattice_constant is None:
        return []
    return [
        (ix, iy, ix * document.lattice_constant, iy * document.lattice_constant, int(charges[iy, ix]))
        for iy in range(document.ny or 0)
        for ix in range(document.nx or 0)
    ]
