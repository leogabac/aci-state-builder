from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import numpy as np
import polars as pl

from .model import IceDocument, TrapState, canonical_axis


REQUIRED_COLUMNS = ("id", "x", "y", "z", "dx", "dy", "dz", "cx", "cy", "cz")


class StateFormatError(ValueError):
    pass


def _native(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return value.item() if hasattr(value, "item") else value


def _infer_square_metadata(traps: list[TrapState]) -> dict[str, Any]:
    half = len(traps) / 2
    n = int(round(math.sqrt(half)))
    if n < 1 or 2 * n * n != len(traps):
        return {}
    horizontal, vertical = traps[: n * n], traps[n * n:]
    if not all(abs(trap.axis[0]) > 0.999 for trap in horizontal):
        return {}
    if not all(abs(trap.axis[1]) > 0.999 for trap in vertical):
        return {}
    xs = sorted({round(float(trap.center[0]), 10) for trap in horizontal})
    if len(xs) < 2:
        return {}
    spacing = float(np.median(np.diff(xs)))
    if spacing <= 0:
        return {}
    return {"geometry": "square", "boundary": "periodic", "nx": n, "ny": n,
            "lattice_constant": spacing}


def document_from_frame(table: pl.DataFrame, *, name: str = "Untitled") -> IceDocument:
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise StateFormatError("missing required columns: " + ", ".join(missing))
    if table.is_empty():
        raise StateFormatError("the CSV contains no traps")
    if "frame" in table.columns:
        frames = table.get_column("frame").drop_nulls().unique().to_list()
        if len(frames) > 1:
            raise StateFormatError("this is a multi-frame trajectory; export one frame before editing it as a state")

    traps: list[TrapState] = []
    for row in table.iter_rows(named=True):
        try:
            axis, occupancy, scale = canonical_axis([row["dx"], row["dy"], row["dz"]])
            extras = {key: _native(value) for key, value in row.items() if key not in REQUIRED_COLUMNS}
            traps.append(TrapState(
                id=int(row["id"]), center=[row["x"], row["y"], row["z"]], axis=axis,
                occupancy=occupancy, direction_scale=scale,
                displacement=[row["cx"], row["cy"], row["cz"]], extras=extras,
            ))
        except (TypeError, ValueError) as error:
            raise StateFormatError(f"invalid row for trap {row.get('id', '?')}: {error}") from error
    lengths = [float(np.linalg.norm(trap.displacement)) for trap in traps]
    nonzero = [value for value in lengths if value > 1e-12]
    return IceDocument(
        traps=traps, trap_separation=2 * float(np.median(nonzero)) if nonzero else 3.0,
        name=name, source_columns=list(table.columns), **_infer_square_metadata(traps),
    )


def read_csv(path: str | Path) -> IceDocument:
    path = Path(path)
    try:
        table = pl.read_csv(path, infer_schema_length=10_000)
    except Exception as error:
        raise StateFormatError(f"could not read CSV: {error}") from error
    return document_from_frame(table, name=path.stem)


def to_frame(document: IceDocument, *, canonical: bool = True) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for trap in document.traps:
        direction = trap.axis * trap.occupancy if canonical else trap.direction
        displacement = (trap.ideal_displacement(document.trap_separation) if canonical
                        else trap.displayed_displacement(document.trap_separation))
        row = {
            "id": trap.id, "x": float(trap.center[0]), "y": float(trap.center[1]),
            "z": float(trap.center[2]), "dx": float(direction[0]), "dy": float(direction[1]),
            "dz": float(direction[2]), "cx": float(displacement[0]),
            "cy": float(displacement[1]), "cz": float(displacement[2]),
        }
        if not canonical:
            row.update(trap.extras)
        rows.append(row)
    frame = pl.DataFrame(rows)
    if not canonical and document.source_columns:
        columns = [column for column in document.source_columns if column in frame.columns]
        columns.extend(column for column in frame.columns if column not in columns)
        frame = frame.select(columns)
    return frame


def write_csv(document: IceDocument, path: str | Path, *, canonical: bool = True) -> None:
    to_frame(document, canonical=canonical).write_csv(path)
