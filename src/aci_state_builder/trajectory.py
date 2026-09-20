from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Callable
import csv
import json
import mmap
import os

import polars as pl


class TrajectoryFormatError(ValueError):
    pass


@dataclass(frozen=True)
class FrameIndexEntry:
    value: str
    start: int
    end: int
    rows: int
    time: str | None = None


@dataclass(frozen=True)
class TrajectoryMetadata:
    path: Path
    columns: tuple[str, ...]
    frames: tuple[FrameIndexEntry, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def particles_per_frame(self) -> int | None:
        counts = {entry.rows for entry in self.frames}
        return next(iter(counts)) if len(counts) == 1 else None


@dataclass
class FrameChunk:
    start: int
    table: pl.DataFrame
    row_offsets: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.row_offsets) - 1

    def frame(self, position: int) -> pl.DataFrame:
        local = position - self.start
        if not 0 <= local < self.count:
            raise IndexError("frame is outside this chunk")
        first, last = self.row_offsets[local:local + 2]
        return self.table.slice(first, last - first)


ProgressCallback = Callable[[int, int], None]


class IndexedCsvTrajectory:
    INDEX_VERSION = 1

    def __init__(
        self, path: str | Path, *, cache_directory: str | Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise TrajectoryFormatError(f"trajectory does not exist: {self.path}")
        self.cache_directory = (
            Path(cache_directory)
            if cache_directory is not None
            else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "aci-state-builder" / "trajectory-indexes"
        )
        self.metadata = self._load_or_build_index(progress)

    def _cache_path(self) -> Path:
        key = sha256(str(self.path).encode("utf-8")).hexdigest()[:24]
        return self.cache_directory / f"{key}.json"

    def _fingerprint(self) -> dict[str, int | str]:
        stat = self.path.stat()
        return {
            "path": str(self.path), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _load_or_build_index(self, progress: ProgressCallback | None) -> TrajectoryMetadata:
        cache_path = self._cache_path()
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if (payload.get("version") == self.INDEX_VERSION
                    and payload.get("source") == self._fingerprint()):
                return TrajectoryMetadata(
                    path=self.path, columns=tuple(payload["columns"]),
                    frames=tuple(FrameIndexEntry(**entry) for entry in payload["frames"]),
                )
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
            pass

        metadata = self._build_index(progress)
        payload = {
            "version": self.INDEX_VERSION,
            "source": self._fingerprint(),
            "columns": list(metadata.columns),
            "frames": [entry.__dict__ for entry in metadata.frames],
        }
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            temporary.replace(cache_path)
        except OSError:
            # A read-only or unavailable cache must not make a valid trajectory unusable.
            pass
        return metadata

    def _build_index(self, progress: ProgressCallback | None) -> TrajectoryMetadata:
        total = self.path.stat().st_size
        with self.path.open("rb") as stream:
            header_bytes = stream.readline()
            try:
                columns = next(csv.reader([header_bytes.decode("utf-8-sig").rstrip("\r\n")]))
            except (UnicodeDecodeError, csv.Error) as error:
                raise TrajectoryFormatError(f"could not read trajectory header: {error}") from error
            if "frame" not in columns:
                raise TrajectoryFormatError("trajectory CSV needs a frame column")
            frame_column = columns.index("frame")
            time_column = columns.index("t") if "t" in columns else None
            data_start = stream.tell()
            if data_start == total:
                raise TrajectoryFormatError("trajectory contains no frames")

            mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                entries: list[FrameIndexEntry] = []
                seen: set[bytes] = set()
                position = data_start
                frame_start = data_start
                current: bytes | None = None
                current_time: str | None = None
                rows = 0
                next_progress = data_start
                while position < total:
                    line_end = mapped.find(b"\n", position)
                    if line_end < 0:
                        line_end = total
                        next_position = total
                    else:
                        next_position = line_end + 1
                    line = mapped[position:line_end].rstrip(b"\r")
                    if line:
                        if frame_column == 0:
                            comma = line.find(b",")
                            frame = line if comma < 0 else line[:comma]
                        else:
                            parts = line.split(b",", frame_column + 1)
                            if len(parts) <= frame_column:
                                raise TrajectoryFormatError("malformed CSV row while indexing")
                            frame = parts[frame_column]
                        if current is None or frame != current:
                            if frame in seen:
                                raise TrajectoryFormatError(
                                    "frames must occupy contiguous blocks in the CSV"
                                )
                            if current is not None:
                                entries.append(FrameIndexEntry(
                                    current.decode("utf-8"), frame_start, position, rows,
                                    current_time,
                                ))
                                seen.add(current)
                            current = frame
                            frame_start = position
                            rows = 0
                            current_time = None
                            if time_column is not None:
                                try:
                                    values = next(csv.reader([line.decode("utf-8")]))
                                    current_time = values[time_column]
                                except (UnicodeDecodeError, csv.Error, IndexError):
                                    current_time = None
                        rows += 1
                    position = next_position
                    if progress is not None and position >= next_progress:
                        progress(position, total)
                        next_progress = position + 64 * 1024 * 1024
                if current is not None:
                    entries.append(FrameIndexEntry(
                        current.decode("utf-8"), frame_start, total, rows, current_time,
                    ))
            finally:
                mapped.close()
        if not entries:
            raise TrajectoryFormatError("trajectory contains no frames")
        if progress is not None:
            progress(total, total)
        return TrajectoryMetadata(self.path, tuple(columns), tuple(entries))

    def load_chunk(self, start: int, count: int) -> FrameChunk:
        if count < 1:
            raise ValueError("chunk size must be positive")
        stop = min(start + count, self.metadata.frame_count)
        if not 0 <= start < stop:
            raise IndexError("chunk starts outside the trajectory")
        entries = self.metadata.frames[start:stop]
        with self.path.open("rb") as stream:
            stream.seek(entries[0].start)
            data = stream.read(entries[-1].end - entries[0].start)
        header = ",".join(self.metadata.columns).encode("utf-8") + b"\n"
        try:
            table = pl.read_csv(BytesIO(header + data), infer_schema_length=10_000)
        except Exception as error:
            raise TrajectoryFormatError(f"could not load trajectory chunk: {error}") from error
        offsets = [0]
        for entry in entries:
            offsets.append(offsets[-1] + entry.rows)
        if offsets[-1] != table.height:
            raise TrajectoryFormatError("trajectory index no longer matches the CSV")
        return FrameChunk(start, table, tuple(offsets))


class TrajectorySession:
    def __init__(self, source: IndexedCsvTrajectory, chunk_size: int = 256, max_chunks: int = 3) -> None:
        self.source = source
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self._chunks: OrderedDict[int, FrameChunk] = OrderedDict()

    def chunk_start(self, position: int) -> int:
        if not 0 <= position < self.source.metadata.frame_count:
            raise IndexError("frame is outside the trajectory")
        return position // self.chunk_size * self.chunk_size

    def cached_frame(self, position: int) -> pl.DataFrame | None:
        start = self.chunk_start(position)
        chunk = self._chunks.get(start)
        if chunk is None:
            return None
        self._chunks.move_to_end(start)
        return chunk.frame(position)

    def has_chunk(self, start: int) -> bool:
        return start in self._chunks

    def load_chunk(self, start: int) -> FrameChunk:
        return self.source.load_chunk(start, self.chunk_size)

    def store_chunk(self, chunk: FrameChunk) -> None:
        self._chunks[chunk.start] = chunk
        self._chunks.move_to_end(chunk.start)
        while len(self._chunks) > self.max_chunks:
            self._chunks.popitem(last=False)

    def configure(self, chunk_size: int, max_chunks: int) -> None:
        if chunk_size != self.chunk_size:
            self._chunks.clear()
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        while len(self._chunks) > self.max_chunks:
            self._chunks.popitem(last=False)

    def loaded_range(self) -> tuple[int, int] | None:
        if not self._chunks:
            return None
        starts = list(self._chunks)
        first = min(starts)
        last_chunk = max(self._chunks.values(), key=lambda chunk: chunk.start)
        return first, last_chunk.start + last_chunk.count - 1
