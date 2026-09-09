from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np


def _vec3(value: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(value), dtype=float)
    if array.shape != (3,):
        raise ValueError("expected a three-component vector")
    return array


def canonical_axis(direction: Iterable[float]) -> tuple[np.ndarray, int, float]:
    """Split an oriented vector into an unoriented unit axis and an Ising state."""
    vector = _vec3(direction)
    scale = float(np.linalg.norm(vector))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("trap direction must be finite and non-zero")
    unit = vector / scale
    first = next((value for value in unit if abs(value) > 1e-12), 1.0)
    occupancy = 1 if first > 0 else -1
    return unit * occupancy, occupancy, scale


@dataclass
class TrapState:
    id: int
    center: np.ndarray
    axis: np.ndarray
    occupancy: int = 1
    direction_scale: float = 1.0
    displacement: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.center = _vec3(self.center)
        self.axis = _vec3(self.axis)
        norm = float(np.linalg.norm(self.axis))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("trap axis must be finite and non-zero")
        self.axis = self.axis / norm
        self.occupancy = 1 if self.occupancy >= 0 else -1
        if self.displacement is not None:
            self.displacement = _vec3(self.displacement)

    @property
    def direction(self) -> np.ndarray:
        return self.axis * self.occupancy * self.direction_scale

    def ideal_displacement(self, trap_separation: float) -> np.ndarray:
        return self.axis * self.occupancy * trap_separation / 2

    def displayed_displacement(self, trap_separation: float) -> np.ndarray:
        return self.displacement if self.displacement is not None else self.ideal_displacement(trap_separation)

    def flip(self) -> None:
        self.occupancy *= -1
        if self.displacement is not None:
            self.displacement *= -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "center": self.center.tolist(),
            "axis": self.axis.tolist(),
            "occupancy": self.occupancy,
            "direction_scale": self.direction_scale,
            "displacement": None if self.displacement is None else self.displacement.tolist(),
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrapState:
        return cls(**data)


@dataclass
class IceDocument:
    traps: list[TrapState]
    geometry: str = "custom"
    boundary: str = "unknown"
    nx: int | None = None
    ny: int | None = None
    lattice_constant: float | None = None
    trap_separation: float = 3.0
    units: str = "um"
    name: str = "Untitled"
    source_columns: list[str] = field(default_factory=list)
    version: int = 1

    def flip_indices(self, indices: Iterable[int]) -> None:
        for index in indices:
            self.traps[index].flip()

    def occupancies(self) -> np.ndarray:
        return np.asarray([trap.occupancy for trap in self.traps], dtype=np.int8)

    def state_snapshot(self) -> list[tuple[int, np.ndarray | None]]:
        return [
            (trap.occupancy, None if trap.displacement is None else trap.displacement.copy())
            for trap in self.traps
        ]

    def restore_snapshot(self, snapshot: list[tuple[int, np.ndarray | None]]) -> None:
        if len(snapshot) != len(self.traps):
            raise ValueError("state snapshot has the wrong length")
        for trap, (occupancy, displacement) in zip(self.traps, snapshot, strict=True):
            trap.occupancy = occupancy
            trap.displacement = None if displacement is None else displacement.copy()

    def set_occupancies(self, values: Iterable[int], *, idealize: bool = True) -> None:
        values_array = np.asarray(tuple(values), dtype=np.int8)
        if values_array.shape != (len(self.traps),):
            raise ValueError("occupancy array has the wrong length")
        for trap, value in zip(self.traps, values_array, strict=True):
            trap.occupancy = 1 if value >= 0 else -1
            if idealize:
                trap.displacement = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "aci-state-builder", "version": self.version, "name": self.name,
            "units": self.units, "geometry": self.geometry, "boundary": self.boundary,
            "nx": self.nx, "ny": self.ny, "lattice_constant": self.lattice_constant,
            "trap_separation": self.trap_separation, "source_columns": self.source_columns,
            "traps": [trap.to_dict() for trap in self.traps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IceDocument:
        if data.get("format") != "aci-state-builder":
            raise ValueError("not an ACI State Builder project")
        fields = dict(data)
        fields.pop("format")
        fields["traps"] = [TrapState.from_dict(item) for item in fields["traps"]]
        return cls(**fields)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> IceDocument:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
