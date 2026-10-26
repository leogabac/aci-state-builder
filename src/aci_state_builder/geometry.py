from __future__ import annotations

import numpy as np

from .model import IceDocument, TrapState, canonical_axis


GEOMETRY_LABELS = {
    "square": "Square",
    "honeycomb": "Honeycomb / kagome ice",
    "triangular": "Triangular",
    "square3d": "Square (height-offset)",
}
BOUNDARY_LABELS = {
    "periodic": "Periodic",
    "closed_spin": "Closed spins",
    "closed_vertex": "Closed vertices",
}


def _unique_rows(values: np.ndarray, tolerance: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    keys: dict[tuple[int, ...], int] = {}
    unique: list[np.ndarray] = []
    inverse: list[int] = []
    for value in values:
        key = tuple(np.rint(value / tolerance).astype(np.int64))
        if key not in keys:
            keys[key] = len(unique)
            unique.append(value)
        inverse.append(keys[key])
    return np.asarray(unique), np.asarray(inverse, dtype=int)


def _connectivity(centers: np.ndarray, directions: np.ndarray, cell: np.ndarray | None):
    endpoints = np.stack((centers - directions / 2, centers + directions / 2), axis=1)
    flat = endpoints.reshape(-1, 3).copy()
    # Vertices are planar even for height-offset square ice: the z displacement
    # separates crossing trap families energetically, not topologically.
    flat[:, 2] = 0.0
    if cell is not None:
        matrix = np.asarray(cell, dtype=float)
        fractional = flat[:, :2] @ np.linalg.inv(matrix)
        fractional -= np.floor(fractional + 1e-10)
        flat[:, :2] = fractional @ matrix
    vertices, inverse = _unique_rows(flat)
    edges = inverse.reshape(-1, 2)
    coordination = np.bincount(edges.ravel(), minlength=len(vertices)).astype(int)
    return vertices.tolist(), edges.tolist(), coordination.tolist()


def _document(geometry, boundary, nx, ny, a, separation, centers, directions, cell, height=0.0):
    traps: list[TrapState] = []
    canonical_directions: list[np.ndarray] = []
    for trap_id, (center, direction) in enumerate(zip(centers, directions, strict=True)):
        axis, occupancy, scale = canonical_axis(direction)
        traps.append(TrapState(trap_id, center, axis, occupancy, scale))
        canonical_directions.append(axis * scale)
    vertices, edges, coordination = _connectivity(
        centers, np.asarray(canonical_directions), cell if boundary == "periodic" else None,
    )
    return IceDocument(
        traps=traps, geometry=geometry, boundary=boundary, nx=nx, ny=ny,
        lattice_constant=float(a), trap_separation=float(separation),
        name=f"{GEOMETRY_LABELS[geometry]} {nx} x {ny}",
        cell_vectors=[] if cell is None else np.asarray(cell, float).tolist(),
        vertex_positions=vertices, trap_vertices=edges,
        vertex_coordination=coordination, height_offset=float(height),
    )


def _origins(nx: int, ny: int) -> np.ndarray:
    return np.asarray([(x + y / 2, np.sqrt(3) * y / 2, 0.0) for y in range(ny) for x in range(nx)])


def _deduplicate(centers: np.ndarray, directions: np.ndarray):
    _, inverse = _unique_rows(centers)
    first: dict[int, int] = {}
    for index, group in enumerate(inverse):
        first.setdefault(int(group), index)
    keep = np.asarray(list(first.values()), dtype=int)
    return centers[keep], directions[keep]


def _repeat(origins: np.ndarray, centers: np.ndarray, directions: np.ndarray, a: float):
    all_centers = np.concatenate([origins + center for center in centers])
    all_directions = np.concatenate([np.zeros_like(origins) + direction for direction in directions])
    all_centers, all_directions = _deduplicate(all_centers, all_directions)
    return all_centers * a, all_directions * a


def _wrap_in_cell(centers: np.ndarray, cell: np.ndarray) -> None:
    fractional = centers[:, :2] @ np.linalg.inv(cell)
    fractional -= np.floor(fractional + 1e-10)
    centers[:, :2] = fractional @ cell


def _square(nx, ny, a, boundary, height):
    if boundary == "periodic":
        centers = [[(x + .5) * a, y * a, 0] for y in range(ny) for x in range(nx)]
        directions = [[a, 0, 0] for _ in range(nx * ny)]
        centers += [[x * a, (y + .5) * a, height] for y in range(ny) for x in range(nx)]
        directions += [[0, a, 0] for _ in range(nx * ny)]
        return np.asarray(centers, float), np.asarray(directions, float)
    centers = np.asarray([[.5, 0, 0], [0, .5, 0], [1, .5, 0], [.5, 1, 0]])
    directions = (np.asarray([[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]])
                  if boundary == "closed_spin" else
                  np.asarray([[0, -1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]]))
    origins = np.asarray([(x, y, 0.0) for y in range(ny) for x in range(nx)])
    return _repeat(origins, centers, directions, a)


def _honeycomb(nx, ny, a, boundary):
    if boundary == "periodic":
        angles = np.arange(np.pi, 2 * np.pi, np.pi / 3)
        centers = np.column_stack((1 + np.cos(angles), 2 / np.sqrt(3) + np.sin(angles), np.zeros(3))) / 2
        directions = np.column_stack((-np.sin(angles) / np.sqrt(3), np.cos(angles) / np.sqrt(3), np.zeros(3)))
    elif boundary == "closed_spin":
        angles = np.arange(0, 2 * np.pi, np.pi / 3)
        centers = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6))) / 2
        directions = np.column_stack((-np.sin(angles) / np.sqrt(3), np.cos(angles) / np.sqrt(3), np.zeros(6)))
    else:
        angles = np.deg2rad([60, 120, 240, 300])
        centers = np.column_stack((np.r_[0, np.cos(angles - np.pi / 3)], np.r_[0, np.sin(angles - np.pi / 3)], np.zeros(5))) / 2
        directions = np.column_stack((np.r_[np.sin(np.pi / 3), np.sin(angles + np.pi / 3)] / np.sqrt(3), np.r_[np.cos(np.pi / 3), np.cos(angles + np.pi / 3)] / np.sqrt(3), np.zeros(5)))
    result = _repeat(_origins(nx, ny), centers, directions, a)
    if boundary == "periodic":
        _wrap_in_cell(result[0], np.asarray([[nx * a, 0], [ny * a / 2, ny * np.sqrt(3) * a / 2]]))
    return result


def _triangular(nx, ny, a, boundary):
    if boundary == "periodic":
        angles = np.deg2rad([60, 120])
        centers = np.column_stack((1 + np.r_[0, np.cos(angles)], np.r_[0, np.sin(angles)], np.zeros(3))) / 2
        directions = np.column_stack((np.r_[1, -np.cos(angles)], np.r_[0, np.sin(angles)], np.zeros(3)))
    elif boundary == "closed_spin":
        angles = np.deg2rad([60, 120, 240, 300])
        centers = np.column_stack((1 + np.r_[0, np.cos(angles)], 2 * np.sin(np.pi / 3) + np.r_[0, np.sin(angles)], np.zeros(5))) / 2
        directions = np.column_stack((np.r_[1, -np.cos(angles)], np.r_[0, np.sin(angles)], np.zeros(5)))
    else:
        angles = np.arange(0, 2 * np.pi, np.pi / 3)
        centers = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6))) / 2
        directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6)))
    result = _repeat(_origins(nx, ny), centers, directions, a)
    if boundary == "periodic":
        _wrap_in_cell(result[0], np.asarray([[nx * a, 0], [ny * a / 2, ny * np.sqrt(3) * a / 2]]))
    return result


def create_lattice(geometry, nx, ny, lattice_constant, trap_separation, boundary="periodic", height_offset=0.0):
    geometry = geometry.lower(); boundary = boundary.lower().replace(" ", "_")
    if geometry not in GEOMETRY_LABELS:
        raise ValueError(f"unsupported geometry: {geometry}")
    if boundary not in BOUNDARY_LABELS:
        raise ValueError(f"unsupported boundary: {boundary}")
    if nx < 1 or ny < 1:
        raise ValueError("lattice dimensions must be positive")
    if lattice_constant <= 0 or trap_separation <= 0:
        raise ValueError("lengths must be positive")
    if geometry == "square3d" and boundary != "periodic":
        raise ValueError("height-offset square currently requires periodic boundaries")
    if geometry in ("square", "square3d"):
        centers, directions = _square(nx, ny, lattice_constant, boundary, height_offset if geometry == "square3d" else 0)
        cell = np.asarray([[nx * lattice_constant, 0], [0, ny * lattice_constant]]) if boundary == "periodic" else None
    elif geometry == "honeycomb":
        centers, directions = _honeycomb(nx, ny, lattice_constant, boundary)
        cell = np.asarray([[nx * lattice_constant, 0], [ny * lattice_constant / 2, ny * np.sqrt(3) * lattice_constant / 2]]) if boundary == "periodic" else None
    else:
        centers, directions = _triangular(nx, ny, lattice_constant, boundary)
        cell = np.asarray([[nx * lattice_constant, 0], [ny * lattice_constant / 2, ny * np.sqrt(3) * lattice_constant / 2]]) if boundary == "periodic" else None
    return _document(geometry, boundary, nx, ny, lattice_constant, trap_separation, centers, directions, cell, height_offset)


def periodic_square(nx, ny, lattice_constant, trap_separation):
    return create_lattice("square", nx, ny, lattice_constant, trap_separation)


def vertex_charges(document: IceDocument) -> np.ndarray | None:
    if len(document.trap_vertices) != len(document.traps) or not document.vertex_positions:
        return None
    charge = np.zeros(len(document.vertex_positions), dtype=np.int16)
    for trap, (start, end) in zip(document.traps, document.trap_vertices, strict=True):
        if start is not None: charge[start] -= trap.occupancy
        if end is not None: charge[end] += trap.occupancy
    return charge


def vertex_data(document: IceDocument):
    charges = vertex_charges(document)
    if charges is None: return []
    coordination = document.vertex_coordination or [0] * len(charges)
    return [(i, float(p[0]), float(p[1]), int(charges[i]), int(coordination[i])) for i, p in enumerate(document.vertex_positions)]


def periodic_vertex_charges(document: IceDocument) -> np.ndarray | None:
    charges = vertex_charges(document)
    if charges is None: return None
    if document.geometry in ("square", "square3d") and document.boundary == "periodic" and document.nx and document.ny:
        return charges.reshape(document.ny, document.nx)
    return charges


def periodic_vertex_data(document: IceDocument):
    if document.geometry not in ("square", "square3d") or document.boundary != "periodic": return []
    return [(i % (document.nx or 1), i // (document.nx or 1), x, y, q) for i, x, y, q, _ in vertex_data(document)]
