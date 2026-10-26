from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable
import math
import shutil
import subprocess

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QBrush

from .model import IceDocument
from .trajectory import IndexedCsvTrajectory


@dataclass(frozen=True)
class VideoExportOptions:
    start: int = 0
    stop: int = 0
    stride: int = 1
    fps: int = 30
    width: int = 1920
    height: int = 1080
    theme: str = "dark"
    quality: str = "balanced"
    show_traps: bool = True
    show_boundary: bool = True
    show_charges: bool = False
    trail_length: int = 0

    def positions(self) -> range:
        return range(self.start, self.stop + 1, self.stride)

    def validated(self, frame_count: int) -> VideoExportOptions:
        if not 0 <= self.start <= self.stop < frame_count:
            raise ValueError("video frame range is outside the trajectory")
        if self.stride < 1 or self.fps < 1:
            raise ValueError("frame stride and FPS must be positive")
        if self.width < 160 or self.height < 120:
            raise ValueError("video resolution is too small")
        if self.width % 2 or self.height % 2:
            raise ValueError("video width and height must be even")
        if self.theme not in ("dark", "light"):
            raise ValueError("unknown video theme")
        if self.quality not in ("draft", "balanced", "high"):
            raise ValueError("unknown video quality")
        return self


@dataclass(frozen=True)
class VideoExportResult:
    path: Path
    rendered_frames: int


def compact_value(value: str) -> str:
    text_value = value.strip()
    if text_value.lstrip("+-").isdigit():
        return text_value
    try:
        number = float(text_value)
    except ValueError:
        return text_value
    if not math.isfinite(number):
        return text_value
    if number == 0:
        return "0"
    text = f"{number:.7g}"
    if "e" in text:
        mantissa, exponent = text.split("e", 1)
        return f"{mantissa}e{int(exponent)}"
    return text


class ScientificVideoRenderer:
    """Fixed-camera renderer with a cached static lattice layer."""

    def __init__(self, document: IceDocument, options: VideoExportOptions) -> None:
        self.document = document
        self.options = options
        self.dark = options.theme == "dark"
        self.background = QColor("#101419" if self.dark else "#fafbfc")
        self.foreground = QColor("#edf1f4" if self.dark else "#20252b")
        self.muted = QColor("#77838e" if self.dark else "#aab2b9")
        self.trap_fill = QColor("#26313a" if self.dark else "#eef1f3")
        self.particle = QColor("#f6c85f" if self.dark else "#15191d")
        self.negative = QColor("#66a8dc")
        self.positive = QColor("#ed8177")
        self.margin = max(document.trap_separation * 1.35, 1.0)
        points = np.asarray([trap.center[:2] for trap in document.traps], dtype=float)
        if document.cell_vectors:
            a, b = np.asarray(document.cell_vectors, dtype=float)
            points = np.vstack((points, [[0, 0], a, b, a + b]))
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        self.minimum = minimum - self.margin
        self.maximum = maximum + self.margin
        available_w = options.width - 96
        available_h = options.height - 112
        span = np.maximum(self.maximum - self.minimum, 1e-9)
        self.scale = min(available_w / span[0], available_h / span[1])
        drawn_w, drawn_h = span * self.scale
        self.offset_x = (options.width - drawn_w) / 2
        self.offset_y = (options.height - drawn_h) / 2 + 18
        self.static = self._static_layer()

    def point(self, x: float, y: float) -> QPointF:
        return QPointF(
            self.offset_x + (x - self.minimum[0]) * self.scale,
            self.offset_y + (self.maximum[1] - y) * self.scale,
        )

    def _peanut(self, half: float, radius: float) -> QPainterPath:
        waist = radius * .40
        path = QPainterPath(QPointF(-half - radius * .72, 0))
        path.cubicTo(-half - radius * .72, -radius * .78, -half - radius * .18, -radius, -half + radius * .30, -radius)
        path.cubicTo(-half + radius * .82, -radius, -waist, -waist, 0, -waist)
        path.cubicTo(waist, -waist, half - radius * .82, -radius, half - radius * .30, -radius)
        path.cubicTo(half + radius * .18, -radius, half + radius * .72, -radius * .78, half + radius * .72, 0)
        path.cubicTo(half + radius * .72, radius * .78, half + radius * .18, radius, half - radius * .30, radius)
        path.cubicTo(half - radius * .82, radius, waist, waist, 0, waist)
        path.cubicTo(-waist, waist, -half + radius * .82, radius, -half + radius * .30, radius)
        path.cubicTo(-half - radius * .18, radius, -half - radius * .72, radius * .78, -half - radius * .72, 0)
        path.closeSubpath()
        return path

    def _static_layer(self) -> QImage:
        image = QImage(self.options.width, self.options.height, QImage.Format.Format_RGBA8888)
        image.fill(self.background)
        painter = QPainter(image); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.options.show_boundary and len(self.document.cell_vectors) == 2:
            a, b = np.asarray(self.document.cell_vectors, dtype=float)
            polygon = [(0, 0), a, a + b, b, (0, 0)]
            painter.setPen(QPen(QColor("#55a9e8"), 2, Qt.PenStyle.DashLine))
            for first, second in zip(polygon, polygon[1:]):
                painter.drawLine(self.point(*first), self.point(*second))
        if self.options.show_traps:
            half = self.document.trap_separation * self.scale / 2
            radius = max(self.document.trap_separation * .22 * self.scale, 2.0)
            peanut = self._peanut(half, radius)
            for trap in self.document.traps:
                painter.save(); painter.translate(self.point(trap.center[0], trap.center[1]))
                painter.rotate(math.degrees(math.atan2(-trap.axis[1], trap.axis[0])))
                fill = QColor(self.trap_fill)
                if abs(float(trap.center[2])) > 1e-12:
                    fill = QColor("#314b62" if self.dark else "#dcebf6")
                painter.setPen(QPen(self.muted, max(1.0, radius * .10)))
                painter.setBrush(QBrush(fill)); painter.drawPath(peanut); painter.restore()
        raw_scale = (self.maximum[0] - self.minimum[0]) * .14
        decade = 10 ** math.floor(math.log10(max(raw_scale, 1e-12)))
        scaled = raw_scale / decade
        scale_length = (1 if scaled < 2 else 2 if scaled < 5 else 5) * decade
        scale_pixels = scale_length * self.scale
        scale_y = self.options.height - 30
        painter.setPen(QPen(self.foreground, 2))
        painter.drawLine(QPointF(30, scale_y), QPointF(30 + scale_pixels, scale_y))
        painter.drawLine(QPointF(30, scale_y - 5), QPointF(30, scale_y + 5))
        painter.drawLine(QPointF(30 + scale_pixels, scale_y - 5), QPointF(30 + scale_pixels, scale_y + 5))
        painter.setFont(QFont("Sans Serif", max(9, self.options.height // 90)))
        painter.drawText(QPointF(30, scale_y - 9), f"{scale_length:g} {self.document.units}")
        painter.end()
        return image

    def _positions(self, table) -> np.ndarray:
        return np.column_stack((
            table.get_column("x").to_numpy() + table.get_column("cx").to_numpy(),
            table.get_column("y").to_numpy() + table.get_column("cy").to_numpy(),
        )).astype(float, copy=False)

    def _charges(self, table) -> np.ndarray | None:
        if len(self.document.trap_vertices) != len(self.document.traps):
            return None
        ids = table.get_column("id").to_numpy()
        expected = np.asarray([trap.id for trap in self.document.traps])
        if not np.array_equal(ids, expected):
            return None
        directions = np.column_stack(tuple(table.get_column(name).to_numpy() for name in ("dx", "dy", "dz")))
        axes = np.asarray([trap.axis for trap in self.document.traps])
        occupancy = np.where(np.einsum("ij,ij->i", directions, axes) >= 0, 1, -1)
        charge = np.zeros(len(self.document.vertex_positions), dtype=int)
        for state, (start, end) in zip(occupancy, self.document.trap_vertices, strict=True):
            if start is not None: charge[start] -= state
            if end is not None: charge[end] += state
        return charge

    def render(self, table, frame_value: str, time_value: str | None, trails: list[np.ndarray]) -> QImage:
        image = self.static.copy()
        painter = QPainter(image); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        positions = self._positions(table)
        if trails:
            history = trails + [positions]
            for segment in range(len(history) - 1):
                color = QColor("#62b8e8"); color.setAlpha(int(35 + 145 * (segment + 1) / len(history)))
                painter.setPen(QPen(color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                for first, second in zip(history[segment], history[segment + 1], strict=True):
                    painter.drawLine(self.point(*first), self.point(*second))
        particle_radius = max(self.document.trap_separation * .135 * self.scale, 2.5)
        painter.setPen(QPen(self.background, max(1.0, particle_radius * .16)))
        painter.setBrush(QBrush(self.particle))
        for position in positions:
            painter.drawEllipse(self.point(*position), particle_radius, particle_radius)
        if self.options.show_charges:
            charges = self._charges(table)
            if charges is not None:
                radius = max(self.document.trap_separation * .28 * self.scale, 8)
                font = QFont("Sans Serif"); font.setBold(True); font.setPixelSize(max(9, int(radius * .72)))
                painter.setFont(font)
                for vertex, charge in zip(self.document.vertex_positions, charges, strict=True):
                    if charge == 0: continue
                    center = self.point(vertex[0], vertex[1])
                    painter.setPen(QPen(self.background, 1.2)); painter.setBrush(self.positive if charge > 0 else self.negative)
                    painter.drawEllipse(center, radius, radius)
                    painter.setPen(QPen(QColor("white")))
                    painter.drawText(QRectF(center.x()-radius, center.y()-radius, 2*radius, 2*radius), Qt.AlignmentFlag.AlignCenter, f"{charge:+d}")
        label_font = QFont("Sans Serif"); label_font.setPixelSize(max(14, self.options.height // 48)); label_font.setBold(True)
        painter.setFont(label_font); painter.setPen(QPen(self.foreground))
        label = f"frame {compact_value(frame_value)}"
        if time_value is not None: label += f"   ·   t = {compact_value(time_value)}"
        painter.drawText(QRectF(30, 20, self.options.width - 60, 42), Qt.AlignmentFlag.AlignLeft, label)
        painter.setFont(QFont("Sans Serif", max(9, self.options.height // 90)))
        painter.setPen(QPen(self.muted)); painter.drawText(QRectF(30, self.options.height - 42, self.options.width - 60, 24), Qt.AlignmentFlag.AlignRight, self.document.name)
        painter.end()
        return image


def export_trajectory_video(
    source: IndexedCsvTrajectory, document: IceDocument, path: str | Path,
    options: VideoExportOptions, *, progress: Callable[[int], None] | None = None,
    cancelled: Event | None = None,
) -> VideoExportResult:
    options.validated(source.metadata.frame_count)
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is required to export video")
    output = Path(path)
    if output.suffix.lower() != ".mp4": output = output.with_suffix(".mp4")
    temporary = output.with_name(output.stem + ".part.mp4")
    crf = {"draft": "25", "balanced": "19", "high": "15"}[options.quality]
    command = [
        executable, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{options.width}x{options.height}",
        "-r", str(options.fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", crf, "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary),
    ]
    renderer = ScientificVideoRenderer(document, options)
    positions = list(options.positions())
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    trails: list[np.ndarray] = []
    try:
        assert process.stdin is not None
        chunk = None
        for rendered, position in enumerate(positions, 1):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("video export cancelled")
            # Dense exports amortize CSV parsing across a block. Sparse exports
            # seek directly to selected frames so skipped data is not decoded.
            if options.stride <= 4:
                if chunk is None or not chunk.start <= position < chunk.start + chunk.count:
                    chunk = source.load_chunk(position, min(128, options.stop - position + 1))
                table = chunk.frame(position)
            else:
                table = source.load_frame(position)
            entry = source.metadata.frames[position]
            image = renderer.render(table, entry.value, entry.time, trails)
            process.stdin.write(bytes(image.constBits()))
            if options.trail_length > 0:
                trails.append(renderer._positions(table))
                trails = trails[-options.trail_length:]
            if progress is not None: progress(int(rendered * 100 / len(positions)))
        process.stdin.close()
        return_code = process.wait()
        if return_code:
            message = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            raise RuntimeError(message.strip() or f"ffmpeg exited with status {return_code}")
        temporary.replace(output)
        return VideoExportResult(output, len(positions))
    except BaseException:
        if process.poll() is None:
            process.terminate(); process.wait(timeout=5)
        temporary.unlink(missing_ok=True)
        raise
    finally:
        for pipe in (process.stdin, process.stderr):
            if pipe is not None and not pipe.closed:
                try:
                    pipe.close()
                except OSError:
                    pass
