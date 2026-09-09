from __future__ import annotations

import numpy as np

from .model import IceDocument


def _shape(document: IceDocument) -> tuple[int, int]:
    if document.geometry != "square" or document.nx is None or document.ny is None:
        raise ValueError("this preset requires a generated square lattice")
    if len(document.traps) != 2 * document.nx * document.ny:
        raise ValueError("square lattice has an unexpected number of traps")
    return document.nx, document.ny


def ferromagnetic(document: IceDocument) -> np.ndarray:
    return np.ones(len(document.traps), dtype=np.int8)


def af2(document: IceDocument) -> np.ndarray:
    nx, ny = _shape(document)
    values = np.ones(len(document.traps), dtype=np.int8)
    offset = nx * ny
    for y in range(ny):
        for x in range(nx):
            # This is equivalent to the three historical flip lists, including duplicates.
            if (x % 2 == 1) != (y % 2 == 0):
                values[y * nx + x] = -1
            if x % 2 == 0:
                values[offset + y * nx + x] = -1
    return values


def af4(document: IceDocument) -> np.ndarray:
    nx, ny = _shape(document)
    values = af2(document)
    offset = nx * ny
    for y in range(1, ny, 2):
        values[offset + y * nx: offset + (y + 1) * nx] *= -1
    return values


def ice(document: IceDocument) -> np.ndarray:
    nx, ny = _shape(document)
    values = af4(document)
    values[nx * ny:] *= -1
    return values


PRESETS = {"Ferromagnetic": ferromagnetic, "AF2": af2, "AF4": af4, "Ice": ice}
