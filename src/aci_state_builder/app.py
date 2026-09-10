from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QBrush, QFont, QKeySequence, QPainter, QPainterPath, QPen, QTransform, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGraphicsItem, QGraphicsScene, QGraphicsView, QGroupBox,
    QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QToolBar,
    QVBoxLayout, QWidget,
)

from .formats import StateFormatError, read_csv, write_csv
from .energy import EnergyParameters, calculate_energy
from .geometry import periodic_square, periodic_vertex_charges, periodic_vertex_data
from .model import IceDocument
from .presets import PRESETS
from .validation import validate


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
        half = max(self.document.trap_separation / 2, 0.5)
        radius = max(self.document.trap_separation * 0.16, 0.12)
        axis = self.trap.axis[:2]
        reach_x = abs(axis[0]) * half + radius * 2
        reach_y = abs(axis[1]) * half + radius * 2
        return QRectF(-reach_x, -reach_y, 2 * reach_x, 2 * reach_y)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        del option, widget
        axis = QPointF(float(self.trap.axis[0]), float(-self.trap.axis[1]))
        half = self.document.trap_separation / 2
        radius = max(self.document.trap_separation * 0.13, 0.10)
        first = QPointF(-axis.x() * half, -axis.y() * half)
        second = QPointF(axis.x() * half, axis.y() * half)
        occupied = second if self.trap.occupancy > 0 else first
        color = QColor("#ff9f1c") if self.isSelected() else QColor("#33515f")
        painter.setPen(QPen(color, max(radius * 0.36, 0.08)))
        painter.drawLine(first, second)
        painter.setBrush(QBrush(QColor("#e8eef0")))
        painter.setPen(QPen(color, max(radius * 0.18, 0.04)))
        painter.drawEllipse(first, radius, radius)
        painter.drawEllipse(second, radius, radius)
        painter.setBrush(QBrush(QColor("#132f3a")))
        painter.drawEllipse(occupied, radius * 0.70, radius * 0.70)

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


class ParameterPanel(QWidget):
    def __init__(self, document: IceDocument, change_callback, lattice_callback, parent=None) -> None:
        super().__init__(parent)
        self.change_callback = change_callback
        self.lattice_callback = lattice_callback
        self._loading = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        lattice = QGroupBox("Lattice")
        lattice_form = QFormLayout(lattice)
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
        layout.addWidget(lattice)

        traps = QGroupBox("Traps")
        trap_form = QFormLayout(traps)
        self.trap_separation = self._number(3.0, 0.001, 1_000_000, 6)
        self.trap_height = self._number(8.0, 0, 1_000_000, 6)
        self.trap_stiffness = self._number(0.1, 0, 1_000_000, 6)
        trap_form.addRow("Separation (um)", self.trap_separation)
        trap_form.addRow("Height (pN nm)", self.trap_height)
        trap_form.addRow("Stiffness (pN/nm)", self.trap_stiffness)
        layout.addWidget(traps)

        parameters = EnergyParameters.from_mapping(document.energy_parameters)
        particles = QGroupBox("Particles")
        particle_form = QFormLayout(particles)
        self.radius = self._number(parameters.particle_radius_um, 0.001, 1_000_000, 6)
        self.susceptibility = self._number(parameters.susceptibility, 0, 1_000_000, 8)
        particle_form.addRow("Radius (um)", self.radius)
        particle_form.addRow("Susceptibility", self.susceptibility)
        layout.addWidget(particles)

        field_group = QGroupBox("Uniform field (spherical)")
        field_form = QFormLayout(field_group)
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
        layout.addWidget(field_group)

        calculation = QGroupBox("Interaction energy")
        calculation_form = QFormLayout(calculation)
        self.cutoff = self._number(parameters.cutoff_um, 0, 1_000_000, 6)
        self.energy_result = QLabel()
        self.energy_result.setWordWrap(True)
        self.energy_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        calculation_form.addRow("Cutoff (um; 0 = all)", self.cutoff)
        calculation_form.addRow(self.energy_result)
        layout.addWidget(calculation)
        layout.addStretch(1)

        for box in (self.radius, self.susceptibility, self.field,
                    self.colatitude, self.azimuth, self.cutoff):
            box.valueChanged.connect(self._parameters_changed)
        for box in (self.nx, self.ny, self.lattice_constant):
            box.valueChanged.connect(self._update_box_size)
        self.apply_lattice.clicked.connect(self._apply_lattice)
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
        self.apply_lattice.setEnabled(generated_square)
        self._loading = False
        self._update_box_size()

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
        self.charge_mode = "Nonzero"
        self.charge_items: list[ChargeItem] = []
        self.undo_stack = QUndoStack(self)
        self.scene = QGraphicsScene(self); self.scene.selectionChanged.connect(self.update_status)
        self.view = IceView(self.scene); self.status = QLabel()
        container = QWidget(); layout = QVBoxLayout(container); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view); layout.addWidget(self.status)
        self.setCentralWidget(container)
        self._create_actions(); self._create_toolbar(); self._create_parameter_dock(); self.rebuild_scene()

    def _action(self, text: str, callback, shortcut=None) -> QAction:
        action = QAction(text, self); action.triggered.connect(callback)
        if shortcut is not None: action.setShortcut(shortcut)
        return action

    def _create_actions(self) -> None:
        self.new_action = self._action("New", self.new_document, QKeySequence.StandardKey.New)
        self.open_action = self._action("Open project", self.open_project, QKeySequence.StandardKey.Open)
        self.save_action = self._action("Save project", self.save_project, QKeySequence.StandardKey.Save)
        self.import_action = self._action("Import CSV", self.import_csv)
        self.export_action = self._action("Export CSV", self.export_csv)
        self.flip_action = self._action("Flip selected", self.flip_selected, QKeySequence("F"))
        self.fit_action = self._action("Fit", self.fit_scene, QKeySequence("0"))
        self.validate_action = self._action("Validate", self.show_validation)
        self.undo_action = self.undo_stack.createUndoAction(self, "Undo")
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.redo_action = self.undo_stack.createRedoAction(self, "Redo")
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main", self); toolbar.setMovable(False); self.addToolBar(toolbar)
        for action in (self.new_action, self.open_action, self.save_action, self.import_action,
                       self.export_action, self.undo_action, self.redo_action, self.flip_action):
            toolbar.addAction(action)
        toolbar.addSeparator()
        for name in PRESETS:
            toolbar.addAction(self._action(name, lambda checked=False, n=name: self.apply_preset(n)))
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
            self.document, self.set_energy_parameters, self.apply_lattice_parameters, self,
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
        self.scene.clear()
        for index, trap in enumerate(self.document.traps):
            item = TrapItem(index, self.document, self.flip_indices)
            item.setPos(float(trap.center[0]), float(-trap.center[1]))
            item.setToolTip(f"id {trap.id} | center ({trap.center[0]:g}, {trap.center[1]:g})")
            self.scene.addItem(item)
        self.rebuild_charge_overlay()
        margin = max(self.document.trap_separation, 1.0)
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-margin, -margin, margin, margin))
        self.parameter_panel.set_document(self.document)
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
        self.push_state_change(
            lambda: self.document.flip_indices(indices),
            "Flip trap" if len(indices) == 1 else "Flip traps",
        )

    def flip_selected(self) -> None:
        indices = [item.index for item in self.scene.selectedItems() if isinstance(item, TrapItem)]
        if indices: self.flip_indices(indices)

    def apply_preset(self, name: str) -> None:
        try:
            values = PRESETS[name](self.document)
            self.push_state_change(
                lambda: self.document.set_occupancies(values, idealize=True),
                f"Apply {name}",
            )
        except ValueError as error: QMessageBox.warning(self, "Preset unavailable", str(error))

    def set_document(self, document: IceDocument, path: Path | None = None) -> None:
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
