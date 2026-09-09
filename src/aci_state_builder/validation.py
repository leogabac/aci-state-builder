from __future__ import annotations

import numpy as np

from .model import IceDocument


def validate(document: IceDocument) -> list[str]:
    issues: list[str] = []
    if not document.traps:
        return ["State contains no traps."]
    ids = [trap.id for trap in document.traps]
    if len(ids) != len(set(ids)):
        issues.append("Trap IDs are not unique.")
    centers: set[tuple[float, float, float]] = set()
    for trap in document.traps:
        if not np.all(np.isfinite(trap.center)):
            issues.append(f"Trap {trap.id} has a non-finite center.")
        key = tuple(np.round(trap.center, 10))
        if key in centers:
            issues.append(f"Trap {trap.id} shares a center with another trap.")
        centers.add(key)
        if trap.displacement is not None:
            if not np.all(np.isfinite(trap.displacement)):
                issues.append(f"Trap {trap.id} has a non-finite colloid displacement.")
            elif np.dot(trap.displacement, trap.direction) < -1e-9:
                issues.append(f"Trap {trap.id} has direction and displacement pointing oppositely.")
    if document.nx is not None and document.ny is not None:
        expected = 2 * document.nx * document.ny
        if document.geometry == "square" and document.boundary == "periodic" and len(ids) != expected:
            issues.append(f"Periodic square metadata expects {expected} traps, found {len(ids)}.")
    return issues
