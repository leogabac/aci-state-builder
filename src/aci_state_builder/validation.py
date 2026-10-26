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
    if document.trap_vertices and len(document.trap_vertices) != len(document.traps):
        issues.append("Topology edge metadata does not match the trap count.")
    if document.vertex_coordination and len(document.vertex_coordination) != len(document.vertex_positions):
        issues.append("Topology coordination metadata does not match the vertex count.")
    if document.boundary == "periodic" and document.geometry != "custom" and len(document.cell_vectors) != 2:
        issues.append("Generated periodic lattice has no valid primitive cell.")
    return issues
