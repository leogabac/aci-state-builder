from __future__ import annotations

import numpy as np

from .model import IceDocument


def _shape(document: IceDocument) -> tuple[int, int]:
    if document.geometry != "square" or document.nx is None or document.ny is None:
        raise ValueError("this preset requires a generated square lattice")
    if len(document.traps) != 2 * document.nx * document.ny:
        raise ValueError("square lattice has an unexpected number of traps")
    return document.nx, document.ny


def _even_shape(document: IceDocument) -> tuple[int, int]:
    nx, ny = _shape(document)
    if nx % 2 or ny % 2:
        raise ValueError("this periodic pattern requires even Nx and Ny")
    return nx, ny


def polarized(document: IceDocument) -> np.ndarray:
    return np.ones(len(document.traps), dtype=np.int8)


def four_in_four_out(document: IceDocument) -> np.ndarray:
    """Return the alternating charged-vertex pattern of square ice."""
    nx, ny = _even_shape(document)
    values = np.ones(len(document.traps), dtype=np.int8)
    offset = nx * ny
    for y in range(ny):
        for x in range(nx):
            if (x % 2 == 1) != (y % 2 == 0):
                values[y * nx + x] = -1
            if x % 2 == 0:
                values[offset + y * nx + x] = -1
    for y in range(1, ny, 2):
        values[offset + y * nx: offset + (y + 1) * nx] *= -1
    return values


def two_in_two_out(document: IceDocument) -> np.ndarray:
    """Return a charge-neutral two-in/two-out pattern of square ice."""
    nx, ny = _even_shape(document)
    values = four_in_four_out(document)
    values[nx * ny:] *= -1
    return values


def randomized(document: IceDocument, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw independent, equiprobable Ising occupancies for every trap."""
    generator = np.random.default_rng() if rng is None else rng
    return generator.choice((-1, 1), size=len(document.traps)).astype(np.int8)


SQUARE_CONFIGURATIONS = {
    "Polarized": polarized,
    "2-in / 2-out": two_in_two_out,
    "4-in / 4-out": four_in_four_out,
    "Randomize": randomized,
}
