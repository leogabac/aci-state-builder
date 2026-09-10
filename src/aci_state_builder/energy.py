from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

import numpy as np

from .model import IceDocument


MU0_SI = 4e-7 * np.pi
JOULE_PER_PN_NM = 1e-21
METRES_PER_UM = 1e-6


@dataclass(frozen=True)
class EnergyParameters:
    particle_radius_um: float = 1.4
    susceptibility: float = 0.4
    field_mT: float = 10.0
    field_angle_deg: float = 0.0
    cutoff_um: float = 0.0

    def validated(self) -> EnergyParameters:
        values = np.asarray([
            self.particle_radius_um, self.susceptibility, self.field_mT,
            self.field_angle_deg, self.cutoff_um,
        ])
        if not np.all(np.isfinite(values)):
            raise ValueError("energy parameters must be finite")
        if self.particle_radius_um <= 0:
            raise ValueError("particle radius must be positive")
        if self.susceptibility < 0 or self.field_mT < 0 or self.cutoff_um < 0:
            raise ValueError("susceptibility, field, and cutoff cannot be negative")
        return self

    def to_dict(self) -> dict[str, float]:
        return {
            "particle_radius_um": self.particle_radius_um,
            "susceptibility": self.susceptibility,
            "field_mT": self.field_mT,
            "field_angle_deg": self.field_angle_deg,
            "cutoff_um": self.cutoff_um,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, float] | None) -> EnergyParameters:
        if not values:
            return cls()
        known = {name: float(values[name]) for name in cls.__dataclass_fields__ if name in values}
        return cls(**known).validated()


@dataclass(frozen=True)
class EnergyResult:
    total_pn_nm: float
    per_particle_pn_nm: float
    geometric_sum_per_um3: float
    elapsed_seconds: float


def dipole_prefactor_pn_nm_um3(parameters: EnergyParameters) -> float:
    """Return mu0*m^2/(4*pi) in pN nm um^3.

    The induced moment follows the convention used in the historical parameter
    files: m = (4*pi*r^3/3) * chi * B / mu0.
    """
    p = parameters.validated()
    radius_m = p.particle_radius_um * METRES_PER_UM
    field_t = p.field_mT * 1e-3
    volume_m3 = 4 * np.pi * radius_m**3 / 3
    moment_a_m2 = volume_m3 * p.susceptibility * field_t / MU0_SI
    prefactor_j_m3 = MU0_SI * moment_a_m2**2 / (4 * np.pi)
    return float(prefactor_j_m3 / JOULE_PER_PN_NM / METRES_PER_UM**3)


def colloid_positions(document: IceDocument) -> np.ndarray:
    return np.asarray([
        trap.center + trap.displayed_displacement(document.trap_separation)
        for trap in document.traps
    ], dtype=float)


def geometric_dipole_sum_pbc(
    positions_um: np.ndarray,
    box_um: tuple[float, float],
    field_angle_deg: float = 0.0,
    cutoff_um: float = 0.0,
    block_size: int = 256,
) -> float:
    """Compute sum[(1-3(Bhat.rhat)^2)/r^3] using 2D minimum-image PBC."""
    positions = np.asarray(positions_um, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n, 3)")
    if len(positions) < 2:
        return 0.0
    box = np.asarray(box_um, dtype=float)
    if box.shape != (2,) or np.any(~np.isfinite(box)) or np.any(box <= 0):
        raise ValueError("PBC box lengths must be positive and finite")
    angle = np.deg2rad(field_angle_deg)
    field_hat = np.asarray([np.cos(angle), np.sin(angle), 0.0])
    cutoff2 = cutoff_um**2
    total = 0.0

    # Blocks bound peak memory while leaving the pair arithmetic in compiled NumPy loops.
    for start in range(0, len(positions), block_size):
        delta = positions[start:start + block_size, None, :] - positions[None, :, :]
        delta[..., :2] -= box * np.rint(delta[..., :2] / box)
        r2 = np.einsum("ijk,ijk->ij", delta, delta)
        mask = r2 > 0
        if cutoff_um > 0:
            mask &= r2 <= cutoff2
        projection = np.einsum("ijk,k->ij", delta, field_hat)
        terms = np.zeros_like(r2)
        terms[mask] = (
            1 - 3 * projection[mask] ** 2 / r2[mask]
        ) / (r2[mask] * np.sqrt(r2[mask]))
        total += float(terms.sum())
    return total / 2


def calculate_energy(document: IceDocument, parameters: EnergyParameters) -> EnergyResult:
    if (document.geometry != "square" or document.boundary != "periodic"
            or document.nx is None or document.ny is None
            or document.lattice_constant is None):
        raise ValueError("energy currently requires a periodic square document")
    started = perf_counter()
    positions = colloid_positions(document)
    geometric_sum = geometric_dipole_sum_pbc(
        positions,
        (document.nx * document.lattice_constant, document.ny * document.lattice_constant),
        parameters.field_angle_deg,
        parameters.cutoff_um,
    )
    total = dipole_prefactor_pn_nm_um3(parameters) * geometric_sum
    return EnergyResult(
        total_pn_nm=total,
        per_particle_pn_nm=total / len(document.traps) if document.traps else 0.0,
        geometric_sum_per_um3=geometric_sum,
        elapsed_seconds=perf_counter() - started,
    )
