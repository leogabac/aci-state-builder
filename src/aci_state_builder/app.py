from __future__ import annotations

from pathlib import Path
import sys
import math
import csv

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QColor, QBrush, QFont, QKeySequence, QPainter, QPainterPath, QPen, QTransform, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGraphicsItem, QGraphicsScene, QGraphicsView, QLabel,
    QHBoxLayout, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QToolBar, QToolButton, QVBoxLayout, QWidget,
)

from .formats import StateFormatError, document_from_frame, read_csv, write_csv
from .energy import EnergyParameters, calculate_energy
from .geometry import periodic_square, periodic_vertex_charges, periodic_vertex_data
from .model import IceDocument
from .presets import SQUARE_CONFIGURATIONS, randomized
from .validation import validate
from .trajectory import IndexedCsvTrajectory, TrajectoryFormatError, TrajectorySession


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(int)


class Worker(QRunnable):
    def __init__(self, function) -> None:
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.function(self.signals)
        except Exception as error:
            self.signals.error.emit(str(error))
        else:
            self.signals.result.emit(result)


class TrapItem(QGraphicsItem):
    def __init__(self, index: int, document: IceDocument, flip_callback) -> None:
        super().__init__()
        self.index = index
        self.document = document
        self.flip_callback = flip_callback
        self._press_position: QPointF | None = None
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)

    @property
    def trap(self):
        return self.document.traps[self.index]

    def boundingRect(self) -> QRectF:
        # Deliberately generous so imported off-axis displacements remain visible.
        reach = max(self.document.trap_separation * 0.85, 0.75)
        if self.trap.displacement is not None:
            reach = max(reach, float(np.linalg.norm(self.trap.displacement[:2])) * 1.3)
        return QRectF(-reach, -reach, 2 * reach, 2 * reach)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        del option, widget
        half = self.document.trap_separation / 2
        radius = max(self.document.trap_separation * 0.22, 0.14)
        waist = radius * 0.40
        angle = math.degrees(math.atan2(-self.trap.axis[1], self.trap.axis[0]))

        # A compact double-well silhouette: two flat-sided elliptical basins
        # joined by a narrow waist, close to the lithographic peanut geometry.
        peanut = QPainterPath(QPointF(-half - radius * 0.72, 0))
        peanut.cubicTo(-half - radius * 0.72, -radius * 0.78,
                       -half - radius * 0.18, -radius, -half + radius * 0.30, -radius)
        peanut.cubicTo(-half + radius * 0.82, -radius, -waist, -waist, 0, -waist)
        peanut.cubicTo(waist, -waist, half - radius * 0.82, -radius,
                       half - radius * 0.30, -radius)
        peanut.cubicTo(half + radius * 0.18, -radius, half + radius * 0.72, -radius * 0.78,
                       half + radius * 0.72, 0)
        peanut.cubicTo(half + radius * 0.72, radius * 0.78,
                       half + radius * 0.18, radius, half - radius * 0.30, radius)
        peanut.cubicTo(half - radius * 0.82, radius, waist, waist, 0, waist)
        peanut.cubicTo(-waist, waist, -half + radius * 0.82, radius,
                       -half + radius * 0.30, radius)
        peanut.cubicTo(-half - radius * 0.18, radius,
                       -half - radius * 0.72, radius * 0.78, -half - radius * 0.72, 0)
        peanut.closeSubpath()

        outline = QColor("#e78632") if self.isSelected() else QColor("#52717c")
        painter.save()
        painter.rotate(angle)
        painter.setPen(QPen(outline, max(radius * 0.13, 0.035)))
        painter.setBrush(QBrush(QColor("#dce7e8")))
        painter.drawPath(peanut)
        painter.setPen(QPen(QColor(90, 119, 128, 105), max(radius * 0.07, 0.025), Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(0, -waist * 0.72), QPointF(0, waist * 0.72))
        painter.restore()

        displacement = self.trap.displayed_displacement(self.document.trap_separation)
        occupied = QPointF(float(displacement[0]), float(-displacement[1]))
        particle_radius = radius * 0.62
        painter.setPen(QPen(QColor("#f7fbfc"), max(radius * 0.11, 0.035)))
        painter.setBrush(QBrush(QColor("#173b47")))
        painter.drawEllipse(occupied, particle_radius, particle_radius)

    def mousePressEvent(self, event) -> None:
        self._press_position = event.scenePos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        moved = self._press_position is not None and (
            event.scenePos() - self._press_position
        ).manhattanLength() > 3
        modifiers = event.modifiers()
        super().mouseReleaseEvent(event)
        selecting = modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier)
        if not moved and not selecting:
            self.flip_callback([self.index])


class ChargeItem(QGraphicsItem):
    COLORS = {
        -4: QColor("#244b9b"),
        -3: QColor("#3569b7"),
        -2: QColor("#4f8bc9"),
        -1: QColor("#88b5dc"),
        0: QColor("#d8dee1"),
        1: QColor("#efaaa2"),
        2: QColor("#df7064"),
        3: QColor("#c94a40"),
        4: QColor("#a72f2a"),
    }

    def __init__(self, q: int, radius: float) -> None:
        super().__init__()
        self.q = q
        self.radius = radius
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setZValue(5)

    def boundingRect(self) -> QRectF:
        return QRectF(-self.radius, -self.radius, 2 * self.radius, 2 * self.radius)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        del option, widget
        color = self.COLORS.get(self.q, QColor("#6b2737") if self.q > 0 else QColor("#243b6b"))
        painter.setPen(QPen(QColor("#ffffff"), self.radius * 0.12))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(self.boundingRect())
        text_color = QColor("#ffffff") if self.q else QColor("#53636b")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(text_color))
        font = QFont(painter.font())
        font.setBold(True)
        # Text rendered through QPainter's font engine can stay device-pixel
        # sized under extreme view transforms.  Convert it into a scene-space
        # path so the signed charge scales with its circle at every zoom.
        font.setPointSizeF(max(self.radius * 1.15, 0.8))
        label = "0" if self.q == 0 else f"{self.q:+d}"
        glyph = QPainterPath()
        glyph.addText(0, 0, font, label)
        painter.save()
        bounds = glyph.boundingRect()
        painter.translate(-bounds.center())
        painter.drawPath(glyph)
        painter.restore()


class IceView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#f7f8f4"))
        self.field_colatitude_deg = 90.0
        self.field_azimuth_deg = 0.0

    def wheelEvent(self, event) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)

    def set_field_direction(self, colatitude_deg: float, azimuth_deg: float) -> None:
        self.field_colatitude_deg = colatitude_deg
        self.field_azimuth_deg = azimuth_deg
        self.viewport().update()

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        del rect
        # Draw in viewport pixels: the cue stays in the corner while panning or zooming.
        painter.save()
        painter.resetTransform()
        origin = QPointF(36, 45)
        painter.setPen(QPen(QColor("#bd4f3c"), 1.5))
        painter.drawLine(origin, origin + QPointF(22, 0)); painter.drawText(origin + QPointF(25, 4), "x")
        painter.setPen(QPen(QColor("#3c8a68"), 1.5))
        painter.drawLine(origin, origin + QPointF(0, -22)); painter.drawText(origin + QPointF(-3, -26), "y")
        theta, phi = np.deg2rad([self.field_colatitude_deg, self.field_azimuth_deg])
        endpoint = origin + QPointF(
            20 * np.sin(theta) * np.cos(phi),
            -20 * np.sin(theta) * np.sin(phi),
        )
        painter.setPen(QPen(QColor("#20252b"), 2.6))
        painter.drawLine(origin, endpoint)
        painter.setBrush(QBrush(QColor("#20252b")))
        painter.drawEllipse(endpoint, 2.8, 2.8)
        painter.setPen(QPen(QColor("#4d5860")))
        painter.drawText(QPointF(12, 72), f"B, z={np.cos(theta):+.2f}")
        painter.restore()


class FieldDirectionPreview(QWidget):
    """A compact, deliberately schematic x-y-z cue for the spherical field."""
    def __init__(self, colatitude: QDoubleSpinBox, azimuth: QDoubleSpinBox, parent=None) -> None:
        super().__init__(parent)
        self.colatitude, self.azimuth = colatitude, azimuth
        self.setMinimumHeight(108)
        self.setToolTip("Field direction: theta is measured from +z and phi is measured from +x toward +y.")
        colatitude.valueChanged.connect(self.update)
        azimuth.valueChanged.connect(self.update)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        origin = QPointF(58, 72)
        x_axis, y_axis, z_axis = QPointF(34, 17), QPointF(-34, 17), QPointF(0, -43)
        axes = ((x_axis, QColor("#bd4f3c"), "x"), (y_axis, QColor("#3c8a68"), "y"), (z_axis, QColor("#4169a8"), "z"))
        for axis, color, label in axes:
            endpoint = origin + axis
            painter.setPen(QPen(color, 1.6))
            painter.drawLine(origin, endpoint)
            painter.drawText(endpoint + QPointF(3, 0), label)
        theta, phi = np.deg2rad([self.colatitude.value(), self.azimuth.value()])
        direction = (
            x_axis * (np.sin(theta) * np.cos(phi))
            + y_axis * (np.sin(theta) * np.sin(phi))
            + z_axis * np.cos(theta)
        )
        endpoint = origin + direction
        painter.setPen(QPen(QColor("#20252b"), 3.0))
        painter.drawLine(origin, endpoint)
        painter.setBrush(QBrush(QColor("#20252b")))
        painter.drawEllipse(endpoint, 3.5, 3.5)
        painter.setPen(QPen(QColor("#4d5860")))
        painter.drawText(QPointF(112, 25), "field B")


class NewDocumentDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New periodic square ice")
        form = QFormLayout(self)
        self.nx = QSpinBox(); self.nx.setRange(1, 200); self.nx.setValue(10)
        self.ny = QSpinBox(); self.ny.setRange(1, 200); self.ny.setValue(10)
        self.lattice = QDoubleSpinBox(); self.lattice.setRange(0.001, 1_000_000)
        self.lattice.setDecimals(6); self.lattice.setValue(8.374011537)
        self.separation = QDoubleSpinBox(); self.separation.setRange(0.001, 1_000_000)
        self.separation.setDecimals(6)
        self.separation.setValue(3.0)
        self.trap_height = QDoubleSpinBox(); self.trap_height.setRange(0, 1_000_000)
        self.trap_height.setDecimals(6); self.trap_height.setValue(8.0)
        self.trap_stiffness = QDoubleSpinBox(); self.trap_stiffness.setRange(0, 1_000_000)
        self.trap_stiffness.setDecimals(6); self.trap_stiffness.setValue(0.1)
        self.boundary = QComboBox(); self.boundary.addItem("Periodic")
        form.addRow("Horizontal cells", self.nx); form.addRow("Vertical cells", self.ny)
        form.addRow("Lattice constant (um)", self.lattice)
        form.addRow("Trap separation (um)", self.separation)
        form.addRow("Trap height (pN nm)", self.trap_height)
        form.addRow("Trap stiffness (pN/nm)", self.trap_stiffness)
        form.addRow("Boundary", self.boundary)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); form.addRow(buttons)

    def create_document(self) -> IceDocument:
        document = periodic_square(self.nx.value(), self.ny.value(), self.lattice.value(), self.separation.value())
        document.trap_height_pn_nm = self.trap_height.value()
        document.trap_stiffness_pn_per_nm = self.trap_stiffness.value()
        return document


class FoldablePane(QWidget):
    def __init__(self, title: str, *, expanded: bool = True, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.header = QToolButton()
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.header.setStyleSheet(
            "QToolButton { border: none; font-weight: 600; padding: 5px 2px; }"
        )
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.body = QWidget()
        self.body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.body.setVisible(expanded)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout.addWidget(self.header)
        layout.addWidget(self.body)
        self.header.toggled.connect(self._set_expanded)

    def _set_expanded(self, expanded: bool) -> None:
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.body.setVisible(expanded)


class PlaybackBar(QWidget):
    def __init__(self, seek_callback, play_callback, parent=None) -> None:
        super().__init__(parent)
        self.seek_callback = seek_callback
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(6)
        self.first = QPushButton("|◀")
        self.previous = QPushButton("◀")
        self.play = QPushButton("▶")
        self.next = QPushButton("▶")
        self.last = QPushButton("▶|")
        for button in (self.first, self.previous, self.play, self.next, self.last):
            button.setFixedWidth(38)
            layout.addWidget(button)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        layout.addWidget(self.slider, 1)
        self.position = QLabel("Frame – / –")
        self.position.setMinimumWidth(150)
        layout.addWidget(self.position)
        self.time = QLabel()
        self.time.setMinimumWidth(90)
        layout.addWidget(self.time)
        self.setVisible(False)

        self.first.clicked.connect(lambda: self.seek_callback(0))
        self.previous.clicked.connect(lambda: self.seek_callback(self.slider.value() - 1))
        self.play.clicked.connect(play_callback)
        self.next.clicked.connect(lambda: self.seek_callback(self.slider.value() + 1))
        self.last.clicked.connect(lambda: self.seek_callback(self.slider.maximum()))
        self.slider.valueChanged.connect(self.seek_callback)

    def configure(self, frame_count: int) -> None:
        self.slider.setRange(0, max(frame_count - 1, 0))
        self.setVisible(True)

    def show_position(self, position: int, frame_value: str, time_value: str | None) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(position)
        self.slider.blockSignals(False)
        self.position.setText(f"Frame {frame_value}  ({position + 1}/{self.slider.maximum() + 1})")
        self.time.setText("" if time_value is None else f"t = {time_value}")

    def set_playing(self, playing: bool) -> None:
        self.play.setText("❚❚" if playing else "▶")
        self.play.setToolTip("Pause" if playing else "Play")


class TrajectoryPane(FoldablePane):
    def __init__(self, settings_callback, extract_callback, parent=None) -> None:
        super().__init__("Trajectory", parent=parent)
        self.settings_callback = settings_callback
        form = QFormLayout(self.body)
        self.file = QLabel("No trajectory")
        self.file.setWordWrap(True)
        self.summary = QLabel()
        self.current = QLabel()
        self.loaded = QLabel()
        self.chunk_size = QSpinBox(); self.chunk_size.setRange(8, 4096); self.chunk_size.setValue(256)
        self.cached_chunks = QSpinBox(); self.cached_chunks.setRange(1, 5); self.cached_chunks.setValue(3)
        self.prefetch = QCheckBox("Load the next chunk while viewing")
        self.prefetch.setChecked(True)
        self.stride = QSpinBox(); self.stride.setRange(1, 100_000); self.stride.setValue(1)
        self.fps = QSpinBox(); self.fps.setRange(1, 120); self.fps.setValue(24)
        self.loop = QCheckBox("Loop playback")
        self.diagnostics = QComboBox(); self.diagnostics.addItems(["When paused", "Every frame", "Off"])
        self.memory = QLabel()
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setVisible(False)
        self.extract = QPushButton("Extract current frame as state")
        form.addRow("File", self.file)
        form.addRow("Contents", self.summary)
        form.addRow("Current", self.current)
        form.addRow("Loaded", self.loaded)
        form.addRow("Frames per chunk", self.chunk_size)
        form.addRow("Chunks in memory", self.cached_chunks)
        form.addRow("Memory estimate", self.memory)
        form.addRow(self.prefetch)
        form.addRow("Frame stride", self.stride)
        form.addRow("Playback (fps)", self.fps)
        form.addRow(self.loop)
        form.addRow("Diagnostics", self.diagnostics)
        form.addRow(self.progress)
        form.addRow(self.extract)
        self.metadata = None
        for box in (self.chunk_size, self.cached_chunks):
            box.valueChanged.connect(self._settings_changed)
        self.extract.clicked.connect(extract_callback)
        self.setVisible(False)

    def set_metadata(self, metadata) -> None:
        self.metadata = metadata
        particles = metadata.particles_per_frame
        particle_text = "variable" if particles is None else f"{particles:,}"
        self.file.setText(metadata.path.name)
        self.file.setToolTip(str(metadata.path))
        self.summary.setText(f"{metadata.frame_count:,} frames · {particle_text} particles/frame")
        self.setVisible(True)
        self._update_memory()

    def set_current(self, text: str) -> None:
        self.current.setText(text)

    def set_loaded_range(self, loaded: tuple[int, int] | None) -> None:
        self.loaded.setText("–" if loaded is None else f"{loaded[0] + 1:,}–{loaded[1] + 1:,}")

    def show_progress(self, percent: int, text: str = "Indexing…") -> None:
        self.setVisible(True)
        self.file.setText(text)
        self.progress.setVisible(True)
        self.progress.setValue(percent)

    def hide_progress(self) -> None:
        self.progress.setVisible(False)

    def _settings_changed(self) -> None:
        self._update_memory()
        self.settings_callback()

    def _update_memory(self) -> None:
        if self.metadata is None or self.metadata.particles_per_frame is None:
            self.memory.setText("depends on frame size")
            return
        # Numeric CSV columns normally become eight-byte values in Polars.
        cached_frames = min(
            self.metadata.frame_count,
            self.chunk_size.value() * self.cached_chunks.value(),
        )
        mib = (self.metadata.particles_per_frame * len(self.metadata.columns)
               * 8 * cached_frames / 2**20)
        self.memory.setText(f"≈ {mib:.0f} MiB maximum")


class ParameterPanel(QWidget):
    def __init__(
        self, document: IceDocument, change_callback, lattice_callback,
        configuration_callback, trajectory_settings_callback,
        extract_frame_callback, parent=None,
    ) -> None:
        super().__init__(parent)
        self.change_callback = change_callback
        self.lattice_callback = lattice_callback
        self.configuration_callback = configuration_callback
        self._loading = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.trajectory_pane = TrajectoryPane(
            trajectory_settings_callback, extract_frame_callback, self,
        )
        layout.addWidget(self.trajectory_pane)

        self.lattice_pane = FoldablePane("Lattice")
        lattice_form = QFormLayout(self.lattice_pane.body)
        self.nx = QSpinBox(); self.nx.setRange(1, 200)
        self.ny = QSpinBox(); self.ny.setRange(1, 200)
        self.lattice_constant = self._number(8.374011537, 0.001, 1_000_000, 6)
        self.box_size = QLabel()
        self.apply_lattice = QPushButton("Apply lattice")
        lattice_form.addRow("Cells Nx", self.nx)
        lattice_form.addRow("Cells Ny", self.ny)
        lattice_form.addRow("Constant a (um)", self.lattice_constant)
        lattice_form.addRow("Box Lx x Ly", self.box_size)
        lattice_form.addRow(self.apply_lattice)
        layout.addWidget(self.lattice_pane)

        self.configuration_pane = FoldablePane("Configuration")
        configuration_form = QFormLayout(self.configuration_pane.body)
        self.configuration = QComboBox()
        self.apply_configuration = QPushButton("Apply configuration")
        self.configuration_note = QLabel()
        self.configuration_note.setWordWrap(True)
        self.configuration_note.setStyleSheet("color: #59656b;")
        configuration_form.addRow("Pattern", self.configuration)
        configuration_form.addRow(self.apply_configuration)
        configuration_form.addRow(self.configuration_note)
        layout.addWidget(self.configuration_pane)

        self.traps_pane = FoldablePane("Traps", expanded=False)
        trap_form = QFormLayout(self.traps_pane.body)
        self.trap_separation = self._number(3.0, 0.001, 1_000_000, 6)
        self.trap_height = self._number(8.0, 0, 1_000_000, 6)
        self.trap_stiffness = self._number(0.1, 0, 1_000_000, 6)
        trap_form.addRow("Separation (um)", self.trap_separation)
        trap_form.addRow("Height (pN nm)", self.trap_height)
        trap_form.addRow("Stiffness (pN/nm)", self.trap_stiffness)
        layout.addWidget(self.traps_pane)

        parameters = EnergyParameters.from_mapping(document.energy_parameters)
        self.particles_pane = FoldablePane("Particles", expanded=False)
        particle_form = QFormLayout(self.particles_pane.body)
        self.radius = self._number(parameters.particle_radius_um, 0.001, 1_000_000, 6)
        self.susceptibility = self._number(parameters.susceptibility, 0, 1_000_000, 8)
        particle_form.addRow("Radius (um)", self.radius)
        particle_form.addRow("Susceptibility", self.susceptibility)
        layout.addWidget(self.particles_pane)

        self.field_pane = FoldablePane("Uniform field (spherical)")
        field_form = QFormLayout(self.field_pane.body)
        self.field = self._number(parameters.field_mT, 0, 1_000_000, 6)
        self.colatitude = self._number(parameters.field_colatitude_deg, 0, 180, 3)
        self.azimuth = self._number(parameters.field_azimuth_deg, -360, 360, 3)
        field_form.addRow("Magnitude B (mT)", self.field)
        field_form.addRow("Co-latitude theta (deg)", self.colatitude)
        field_form.addRow("Azimuth phi (deg)", self.azimuth)
        field_form.addRow(FieldDirectionPreview(self.colatitude, self.azimuth, self))
        convention = QLabel("theta: from +z   |   phi: +x toward +y")
        convention.setStyleSheet("color: #59656b;")
        field_form.addRow(convention)
        layout.addWidget(self.field_pane)

        self.energy_pane = FoldablePane("Interaction energy")
        calculation_form = QFormLayout(self.energy_pane.body)
        self.cutoff = self._number(parameters.cutoff_um, 0, 1_000_000, 6)
        self.energy_result = QLabel()
        self.energy_result.setWordWrap(True)
        self.energy_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        calculation_form.addRow("Cutoff (um; 0 = all)", self.cutoff)
        calculation_form.addRow(self.energy_result)
        layout.addWidget(self.energy_pane)
        layout.addStretch(1)

        for box in (self.radius, self.susceptibility, self.field,
                    self.colatitude, self.azimuth, self.cutoff):
            box.valueChanged.connect(self._parameters_changed)
        for box in (self.nx, self.ny, self.lattice_constant):
            box.valueChanged.connect(self._update_box_size)
        self.apply_lattice.clicked.connect(self._apply_lattice)
        self.apply_configuration.clicked.connect(self._apply_configuration)
        self.set_document(document)

    @staticmethod
    def _number(value: float, minimum: float, maximum: float, decimals: int) -> QDoubleSpinBox:
        box = QDoubleSpinBox(); box.setRange(minimum, maximum); box.setDecimals(decimals)
        box.setValue(value); box.setKeyboardTracking(False)
        return box

    def parameters(self) -> EnergyParameters:
        return EnergyParameters(
            particle_radius_um=self.radius.value(), susceptibility=self.susceptibility.value(),
            field_mT=self.field.value(), field_colatitude_deg=self.colatitude.value(),
            field_azimuth_deg=self.azimuth.value(), cutoff_um=self.cutoff.value(),
        ).validated()

    def set_document(self, document: IceDocument) -> None:
        parameters = EnergyParameters.from_mapping(document.energy_parameters)
        self._loading = True
        values = (
            (self.nx, document.nx or 1), (self.ny, document.ny or 1),
            (self.lattice_constant, document.lattice_constant or 1.0),
            (self.trap_separation, document.trap_separation),
            (self.trap_height, document.trap_height_pn_nm),
            (self.trap_stiffness, document.trap_stiffness_pn_per_nm),
            (self.radius, parameters.particle_radius_um),
            (self.susceptibility, parameters.susceptibility),
            (self.field, parameters.field_mT),
            (self.colatitude, parameters.field_colatitude_deg),
            (self.azimuth, parameters.field_azimuth_deg),
            (self.cutoff, parameters.cutoff_um),
        )
        for box, value in values:
            box.setValue(value)
        generated_square = document.geometry == "square" and document.nx is not None and document.ny is not None
        self._generated_square = generated_square
        self.apply_lattice.setEnabled(generated_square)
        selected = self.configuration.currentText()
        self.configuration.clear()
        if generated_square:
            even_shape = document.nx % 2 == 0 and document.ny % 2 == 0
            names = ["Polarized"]
            if even_shape:
                names.extend(("2-in / 2-out", "4-in / 4-out"))
            names.append("Randomize")
            self.configuration.addItems(names)
            parity_note = "" if even_shape else " Alternating patterns require even Nx and Ny."
            self.configuration_note.setText(
                "Square-lattice topology labels. Which pattern is energetically "
                f"favored depends on the field and interaction parameters.{parity_note}"
            )
        else:
            self.configuration.addItem("Randomize")
            self.configuration_note.setText(
                "Only randomization is available for this imported/custom lattice."
            )
        if selected and self.configuration.findText(selected) >= 0:
            self.configuration.setCurrentText(selected)
        self._loading = False
        self._update_box_size()

    def set_trajectory_mode(self, enabled: bool) -> None:
        self.apply_lattice.setEnabled(not enabled and self._generated_square)
        self.apply_configuration.setEnabled(not enabled)
        self.configuration.setEnabled(not enabled)

    def _update_box_size(self) -> None:
        self.box_size.setText(
            f"{self.nx.value() * self.lattice_constant.value():g} x "
            f"{self.ny.value() * self.lattice_constant.value():g} um"
        )

    def _apply_lattice(self) -> None:
        self.lattice_callback(
            self.nx.value(), self.ny.value(), self.lattice_constant.value(),
            self.trap_separation.value(), self.trap_height.value(), self.trap_stiffness.value(),
        )

    def _apply_configuration(self) -> None:
        self.configuration_callback(self.configuration.currentText())

    def _parameters_changed(self) -> None:
        if not self._loading:
            self.change_callback(self.parameters())


class StateCommand(QUndoCommand):
    def __init__(self, window, before, after, text: str) -> None:
        super().__init__(text)
        self.window, self.before, self.after = window, before, after

    def _apply(self, snapshot) -> None:
        self.window.document.restore_snapshot(snapshot)
        self.window.refresh_items()

    def redo(self) -> None:
        self._apply(self.after)

    def undo(self) -> None:
        self._apply(self.before)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ACI State Builder"); self.resize(1180, 820)
        self.document = periodic_square(10, 10, 8.374011537, 3.0)
        self.project_path: Path | None = None
        self.trajectory_session: TrajectorySession | None = None
        self.trajectory_token = 0
        self.requested_frame = 0
        self.current_trajectory_position: int | None = None
        self.loading_chunks: set[int] = set()
        self.workers: set[Worker] = set()
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(2)
        self.charge_mode = "Nonzero"
        self.charge_items: list[ChargeItem] = []
        self.trap_items: list[TrapItem] = []
        self.undo_stack = QUndoStack(self)
        self.scene = QGraphicsScene(self); self.scene.selectionChanged.connect(self.update_status)
        self.view = IceView(self.scene); self.status = QLabel()
        container = QWidget(); layout = QVBoxLayout(container); layout.setContentsMargins(0, 0, 0, 0)
        self.playback = PlaybackBar(self.seek_trajectory, self.toggle_playback, self)
        layout.addWidget(self.view); layout.addWidget(self.playback); layout.addWidget(self.status)
        self.setCentralWidget(container)
        self.play_timer = QTimer(self); self.play_timer.timeout.connect(self.advance_trajectory)
        self.seek_timer = QTimer(self); self.seek_timer.setSingleShot(True)
        self.seek_timer.timeout.connect(self._request_pending_frame)
        self._create_actions(); self._create_toolbar(); self._create_parameter_dock(); self.rebuild_scene()

    def _action(self, text: str, callback, shortcut=None) -> QAction:
        action = QAction(text, self); action.triggered.connect(callback)
        if shortcut is not None: action.setShortcut(shortcut)
        return action

    def _create_actions(self) -> None:
        self.new_action = self._action("New", self.new_document, QKeySequence.StandardKey.New)
        self.open_action = self._action("Open project", self.open_project, QKeySequence.StandardKey.Open)
        self.save_action = self._action("Save project", self.save_project, QKeySequence.StandardKey.Save)
        self.import_action = self._action("Import state", self.import_csv)
        self.trajectory_action = self._action("Open trajectory", self.open_trajectory)
        self.export_action = self._action("Export CSV", self.export_csv)
        self.flip_action = self._action("Flip selected", self.flip_selected, QKeySequence("F"))
        self.fit_action = self._action("Fit", self.fit_scene, QKeySequence("0"))
        self.validate_action = self._action("Validate", self.show_validation)
        self.undo_action = self.undo_stack.createUndoAction(self, "Undo")
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.redo_action = self.undo_stack.createRedoAction(self, "Redo")
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.play_action = self._action("Play / pause trajectory", self.toggle_playback, QKeySequence("Space"))
        self.addAction(self.play_action)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main", self); toolbar.setMovable(False); self.addToolBar(toolbar)
        for action in (self.new_action, self.open_action, self.save_action, self.import_action,
                       self.trajectory_action,
                       self.export_action, self.undo_action, self.redo_action, self.flip_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Charges "))
        self.charge_selector = QComboBox()
        self.charge_selector.addItems(["Off", "Nonzero", "All"])
        self.charge_selector.setCurrentText(self.charge_mode)
        self.charge_selector.currentTextChanged.connect(self.set_charge_mode)
        toolbar.addWidget(self.charge_selector)
        toolbar.addSeparator(); toolbar.addAction(self.fit_action); toolbar.addAction(self.validate_action)

    def _create_parameter_dock(self) -> None:
        self.parameter_panel = ParameterPanel(
            self.document, self.set_energy_parameters, self.apply_lattice_parameters,
            self.apply_configuration, self.trajectory_settings_changed,
            self.extract_trajectory_frame, self,
        )
        dock = QDockWidget("Physical parameters", self)
        dock.setObjectName("physical-parameters")
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dock.setMinimumWidth(300)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.parameter_panel)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def rebuild_scene(self) -> None:
        self.charge_items.clear()
        self.trap_items.clear()
        self.scene.clear()
        for index, trap in enumerate(self.document.traps):
            item = TrapItem(index, self.document, self.flip_indices)
            item.setPos(float(trap.center[0]), float(-trap.center[1]))
            item.setToolTip(f"id {trap.id} | center ({trap.center[0]:g}, {trap.center[1]:g})")
            if self.trajectory_session is not None:
                item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.scene.addItem(item)
            self.trap_items.append(item)
        self.rebuild_charge_overlay()
        margin = max(self.document.trap_separation, 1.0)
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-margin, -margin, margin, margin))
        self.parameter_panel.set_document(self.document)
        self.parameter_panel.set_trajectory_mode(self.trajectory_session is not None)
        self.fit_scene(); self.update_status(); self.update_energy()

    def refresh_items(self) -> None:
        for item in self.scene.items():
            if isinstance(item, TrapItem): item.update()
        self.rebuild_charge_overlay()
        self.update_status(); self.update_energy()

    def rebuild_charge_overlay(self) -> None:
        for item in self.charge_items:
            self.scene.removeItem(item)
        self.charge_items.clear()
        if self.charge_mode == "Off":
            return
        radius = max(self.document.trap_separation * 0.40, 0.5)
        for ix, iy, x, y, q in periodic_vertex_data(self.document):
            if self.charge_mode == "Nonzero" and q == 0:
                continue
            item = ChargeItem(q, radius)
            item.setPos(x, -y)
            item.setToolTip(f"periodic vertex ({ix}, {iy}) | q = {q:+d} = N_in - N_out")
            self.scene.addItem(item)
            self.charge_items.append(item)

    def set_charge_mode(self, mode: str) -> None:
        self.charge_mode = mode
        self.rebuild_charge_overlay()
        self.update_status()

    def update_energy(self) -> None:
        try:
            parameters = EnergyParameters.from_mapping(self.document.energy_parameters)
            self.view.set_field_direction(parameters.field_colatitude_deg, parameters.field_azimuth_deg)
            result = calculate_energy(self.document, parameters)
            cutoff = "all pairs" if parameters.cutoff_um == 0 else f"cutoff {parameters.cutoff_um:g} um"
            self.parameter_panel.energy_result.setText(
                f"<b>{result.total_pn_nm:.8g} pN nm</b> total<br>"
                f"{result.per_particle_pn_nm:.8g} pN nm / colloid<br>"
                f"{cutoff} &nbsp;|&nbsp; {result.elapsed_seconds * 1e3:.1f} ms"
            )
        except ValueError as error:
            self.parameter_panel.energy_result.setText(f"Energy unavailable: {error}")

    def set_energy_parameters(self, parameters: EnergyParameters) -> None:
        self.document.energy_parameters = parameters.to_dict()
        self.update_energy()

    def apply_lattice_parameters(
        self, nx: int, ny: int, lattice_constant: float, trap_separation: float,
        trap_height: float, trap_stiffness: float,
    ) -> None:
        same_shape = (nx, ny) == (self.document.nx, self.document.ny)
        if not same_shape:
            answer = QMessageBox.question(
                self, "Rebuild lattice",
                "Changing the system size creates a new lattice state. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.parameter_panel.set_document(self.document)
                return

        replacement = periodic_square(nx, ny, lattice_constant, trap_separation)
        replacement.name = self.document.name
        replacement.energy_parameters = dict(self.document.energy_parameters)
        replacement.trap_height_pn_nm = trap_height
        replacement.trap_stiffness_pn_per_nm = trap_stiffness
        if same_shape:
            replacement.source_columns = list(self.document.source_columns)
            for old, new in zip(self.document.traps, replacement.traps, strict=True):
                new.occupancy = old.occupancy
                new.direction_scale = old.direction_scale
                new.displacement = None if old.displacement is None else old.displacement.copy()
                new.extras = dict(old.extras)
        self.set_document(replacement, self.project_path)

    def fit_scene(self) -> None:
        self.view.setTransform(QTransform())
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def update_status(self) -> None:
        selected = len([item for item in self.scene.selectedItems() if isinstance(item, TrapItem)])
        charge = periodic_vertex_charges(self.document); charge_text = ""
        if charge is not None:
            unique, counts = np.unique(charge, return_counts=True)
            charge_text = " | vertices " + ", ".join(
                f"Q={int(value):+d}: {int(count)}"
                for value, count in zip(unique, counts, strict=True)
            )
        self.status.setText(f"  {self.document.name} | {len(self.document.traps)} traps | {selected} selected{charge_text}")

    def push_state_change(self, mutation, text: str) -> None:
        before = self.document.state_snapshot()
        mutation()
        after = self.document.state_snapshot()
        self.document.restore_snapshot(before)
        self.undo_stack.push(StateCommand(self, before, after, text))

    def flip_indices(self, indices: list[int]) -> None:
        if self.trajectory_session is not None:
            return
        self.push_state_change(
            lambda: self.document.flip_indices(indices),
            "Flip trap" if len(indices) == 1 else "Flip traps",
        )

    def flip_selected(self) -> None:
        indices = [item.index for item in self.scene.selectedItems() if isinstance(item, TrapItem)]
        if indices: self.flip_indices(indices)

    def apply_configuration(self, name: str) -> None:
        if self.trajectory_session is not None:
            return
        try:
            function = (
                SQUARE_CONFIGURATIONS[name]
                if name in SQUARE_CONFIGURATIONS else randomized
            )
            values = function(self.document)
            self.push_state_change(
                lambda: self.document.set_occupancies(values, idealize=True),
                f"Apply {name}",
            )
        except ValueError as error:
            QMessageBox.warning(self, "Configuration unavailable", str(error))

    def _run_worker(self, function, on_result, on_error=None, on_progress=None) -> None:
        worker = Worker(function)
        self.workers.add(worker)

        def finish(result) -> None:
            self.workers.discard(worker)
            on_result(result)

        def fail(message: str) -> None:
            self.workers.discard(worker)
            if on_error is not None:
                on_error(message)
            else:
                QMessageBox.critical(self, "Background task failed", message)

        worker.signals.result.connect(finish)
        worker.signals.error.connect(fail)
        if on_progress is not None:
            worker.signals.progress.connect(on_progress)
        self.thread_pool.start(worker)

    def open_trajectory(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open trajectory", "", "CSV trajectories (*.csv)"
        )
        if not filename:
            return
        self._open_trajectory_path(filename)

    def _open_trajectory_path(self, filename: str) -> None:
        self._leave_trajectory_mode()
        self.trajectory_token += 1
        token = self.trajectory_token
        pane = self.parameter_panel.trajectory_pane
        pane.show_progress(0, f"Indexing {Path(filename).name}…")
        self.status.setText("  Building trajectory frame index…")

        def construct(signals):
            source = IndexedCsvTrajectory(
                filename,
                progress=lambda done, total: signals.progress.emit(
                    100 if total == 0 else int(done * 100 / total)
                ),
            )
            return token, source

        def opened(result) -> None:
            result_token, source = result
            if result_token != self.trajectory_token:
                return
            self.trajectory_session = TrajectorySession(
                source, pane.chunk_size.value(), pane.cached_chunks.value()
            )
            self.requested_frame = 0
            self.current_trajectory_position = None
            pane.hide_progress(); pane.set_metadata(source.metadata)
            self.playback.configure(source.metadata.frame_count)
            self.flip_action.setEnabled(False)
            self.undo_action.setEnabled(False); self.redo_action.setEnabled(False)
            self.parameter_panel.set_trajectory_mode(True)
            self.request_trajectory_frame(0)

        def failed(message: str) -> None:
            pane.hide_progress(); pane.setVisible(False)
            QMessageBox.critical(self, "Could not open trajectory", message)
            self.update_status()

        self._run_worker(construct, opened, failed, pane.progress.setValue)

    def seek_trajectory(self, position: int) -> None:
        if self.trajectory_session is None:
            return
        maximum = self.trajectory_session.source.metadata.frame_count - 1
        self.requested_frame = min(max(int(position), 0), maximum)
        self.seek_timer.start(60)

    def _request_pending_frame(self) -> None:
        self.request_trajectory_frame(self.requested_frame)

    def request_trajectory_frame(self, position: int) -> None:
        session = self.trajectory_session
        if session is None:
            return
        maximum = session.source.metadata.frame_count - 1
        position = min(max(position, 0), maximum)
        self.requested_frame = position
        table = session.cached_frame(position)
        if table is not None:
            self._display_trajectory_frame(position, table)
            return
        start = session.chunk_start(position)
        if start in self.loading_chunks:
            return
        self.loading_chunks.add(start)
        token = self.trajectory_token
        self.status.setText(f"  Loading trajectory frames {start + 1:,}…")

        def load(signals):
            del signals
            return token, start, session.load_chunk(start)

        def loaded(result) -> None:
            result_token, loaded_start, chunk = result
            self.loading_chunks.discard(loaded_start)
            if result_token != self.trajectory_token or self.trajectory_session is not session:
                return
            session.store_chunk(chunk)
            self.parameter_panel.trajectory_pane.set_loaded_range(session.loaded_range())
            table = session.cached_frame(self.requested_frame)
            if table is not None:
                self._display_trajectory_frame(self.requested_frame, table)

        def failed(message: str) -> None:
            self.loading_chunks.discard(start)
            if token == self.trajectory_token:
                self.pause_playback()
                QMessageBox.critical(self, "Could not load trajectory frames", message)

        self._run_worker(load, loaded, failed)

    def _display_trajectory_frame(self, position: int, table) -> None:
        session = self.trajectory_session
        if session is None:
            return
        try:
            replacement = document_from_frame(table, name=session.source.path.stem)
        except StateFormatError as error:
            self.pause_playback()
            QMessageBox.critical(self, "Invalid trajectory frame", str(error))
            return
        replacement.energy_parameters = dict(self.document.energy_parameters)
        replacement.trap_height_pn_nm = self.document.trap_height_pn_nm
        replacement.trap_stiffness_pn_per_nm = self.document.trap_stiffness_pn_per_nm
        if self.current_trajectory_position is not None:
            replacement.trap_separation = self.document.trap_separation
        same_items = (
            len(replacement.traps) == len(self.trap_items)
            and all(item.trap.id == trap.id for item, trap in zip(self.trap_items, replacement.traps, strict=True))
        )
        self.document = replacement
        self.project_path = None
        if not same_items:
            self.rebuild_scene()
        else:
            for item, trap in zip(self.trap_items, replacement.traps, strict=True):
                item.prepareGeometryChange()
                item.document = replacement
                item.setPos(float(trap.center[0]), float(-trap.center[1]))
                item.update()
            self.rebuild_charge_overlay()
            self.update_status()
        self.current_trajectory_position = position
        entry = session.source.metadata.frames[position]
        self.playback.show_position(position, entry.value, entry.time)
        current_text = f"frame {entry.value}"
        if entry.time is not None:
            current_text += f" · t={entry.time}"
        self.parameter_panel.trajectory_pane.set_current(current_text)
        diagnostics = self.parameter_panel.trajectory_pane.diagnostics.currentText()
        if diagnostics == "Every frame" or (diagnostics == "When paused" and not self.play_timer.isActive()):
            self.update_energy()
        elif diagnostics == "Off":
            self.parameter_panel.energy_result.setText("Energy disabled during trajectory viewing")
        self._prefetch_next_chunk(position)

    def _prefetch_next_chunk(self, position: int) -> None:
        session = self.trajectory_session
        pane = self.parameter_panel.trajectory_pane
        if session is None or not pane.prefetch.isChecked():
            return
        start = session.chunk_start(position) + session.chunk_size
        if start >= session.source.metadata.frame_count or session.has_chunk(start) or start in self.loading_chunks:
            return
        self.loading_chunks.add(start)
        token = self.trajectory_token

        def load(signals):
            del signals
            return token, start, session.load_chunk(start)

        def loaded(result) -> None:
            result_token, loaded_start, chunk = result
            self.loading_chunks.discard(loaded_start)
            if result_token == self.trajectory_token and self.trajectory_session is session:
                session.store_chunk(chunk)
                pane.set_loaded_range(session.loaded_range())

        self._run_worker(load, loaded, lambda message: self.loading_chunks.discard(start))

    def trajectory_settings_changed(self) -> None:
        session = self.trajectory_session
        if session is None:
            return
        pane = self.parameter_panel.trajectory_pane
        # In-flight chunks were requested with the old boundaries.  Let those
        # reads finish harmlessly, but ignore them and start a fresh generation.
        self.trajectory_token += 1
        self.loading_chunks.clear()
        session.configure(pane.chunk_size.value(), pane.cached_chunks.value())
        pane.set_loaded_range(None)
        self.request_trajectory_frame(self.requested_frame)

    def toggle_playback(self) -> None:
        if self.trajectory_session is None:
            return
        if self.play_timer.isActive():
            self.pause_playback()
        else:
            fps = self.parameter_panel.trajectory_pane.fps.value()
            self.play_timer.start(max(1, round(1000 / fps)))
            self.playback.set_playing(True)

    def pause_playback(self) -> None:
        was_playing = self.play_timer.isActive()
        self.play_timer.stop(); self.playback.set_playing(False)
        if was_playing and self.trajectory_session is not None:
            if self.parameter_panel.trajectory_pane.diagnostics.currentText() == "When paused":
                self.update_energy()

    def advance_trajectory(self) -> None:
        session = self.trajectory_session
        if session is None or self.current_trajectory_position != self.requested_frame:
            return
        pane = self.parameter_panel.trajectory_pane
        position = self.requested_frame + pane.stride.value()
        if position >= session.source.metadata.frame_count:
            if pane.loop.isChecked():
                position = 0
            else:
                self.pause_playback()
                return
        self.request_trajectory_frame(position)

    def extract_trajectory_frame(self) -> None:
        if self.trajectory_session is None or self.current_trajectory_position is None:
            return
        frame_value = self.trajectory_session.source.metadata.frames[self.current_trajectory_position].value
        document = IceDocument.from_dict(self.document.to_dict())
        document.name = f"{document.name}-frame-{frame_value}"
        self.set_document(document)

    def _leave_trajectory_mode(self) -> None:
        self.pause_playback()
        self.trajectory_token += 1
        self.trajectory_session = None
        self.current_trajectory_position = None
        self.loading_chunks.clear()
        self.playback.setVisible(False)
        if hasattr(self, "parameter_panel"):
            self.parameter_panel.trajectory_pane.setVisible(False)
            self.parameter_panel.set_trajectory_mode(False)
        if hasattr(self, "flip_action"):
            self.flip_action.setEnabled(True)

    def set_document(self, document: IceDocument, path: Path | None = None) -> None:
        self._leave_trajectory_mode()
        self.document, self.project_path = document, path
        self.undo_stack.clear(); self.rebuild_scene()

    def new_document(self) -> None:
        dialog = NewDocumentDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted: self.set_document(dialog.create_document())

    def open_project(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Open project", "", "ACI projects (*.aci.json *.json)")
        if filename:
            try: self.set_document(IceDocument.load(filename), Path(filename))
            except Exception as error: QMessageBox.critical(self, "Could not open project", str(error))

    def save_project(self) -> None:
        filename = str(self.project_path) if self.project_path else ""
        if not filename:
            filename, _ = QFileDialog.getSaveFileName(self, "Save project", "state.aci.json", "ACI projects (*.aci.json)")
        if filename: self.document.save(filename); self.project_path = Path(filename)

    def import_csv(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Import state CSV", "", "CSV files (*.csv)")
        if not filename: return
        try:
            with Path(filename).open("r", encoding="utf-8-sig", newline="") as stream:
                columns = next(csv.reader([stream.readline()]))
        except (OSError, UnicodeError, csv.Error, StopIteration) as error:
            QMessageBox.critical(self, "Could not import state", str(error)); return
        if "frame" in columns:
            self._open_trajectory_path(filename)
            return
        try: document = read_csv(filename)
        except StateFormatError as error:
            QMessageBox.critical(self, "Could not import state", str(error)); return
        self.set_document(document); issues = validate(document)
        if issues: QMessageBox.warning(self, "Imported with warnings", "\n".join(issues[:20]))

    def export_csv(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Export state CSV", f"{self.document.name}.csv", "CSV files (*.csv)")
        if not filename: return
        choice = QMessageBox.question(
            self, "Canonical export", "Normalize directions and place every colloid at an ideal well?\n\n"
            "Choose No to preserve imported scales, displacements, and extra columns.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
        if choice != QMessageBox.StandardButton.Cancel:
            write_csv(self.document, filename, canonical=choice == QMessageBox.StandardButton.Yes)

    def show_validation(self) -> None:
        issues = validate(self.document)
        if issues: QMessageBox.warning(self, "Validation issues", "\n".join(issues[:40]))
        else: QMessageBox.information(self, "Validation", "The state is internally consistent.")


def main() -> int:
    application = QApplication(sys.argv); application.setApplicationName("ACI State Builder")
    window = MainWindow(); window.show(); return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
