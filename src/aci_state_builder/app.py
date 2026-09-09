from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QBrush, QKeySequence, QPainter, QPen, QTransform, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGraphicsItem, QGraphicsScene, QGraphicsView, QLabel, QMainWindow,
    QMessageBox, QSpinBox, QToolBar, QVBoxLayout, QWidget,
)

from .formats import StateFormatError, read_csv, write_csv
from .geometry import periodic_square, periodic_vertex_charges
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


class IceView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#f7f8f4"))

    def wheelEvent(self, event) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)


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
        self.boundary = QComboBox(); self.boundary.addItem("Periodic")
        form.addRow("Horizontal cells", self.nx); form.addRow("Vertical cells", self.ny)
        form.addRow("Lattice constant (um)", self.lattice)
        form.addRow("Trap separation (um)", self.separation); form.addRow("Boundary", self.boundary)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); form.addRow(buttons)

    def create_document(self) -> IceDocument:
        return periodic_square(self.nx.value(), self.ny.value(), self.lattice.value(), self.separation.value())


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
        self.undo_stack = QUndoStack(self)
        self.scene = QGraphicsScene(self); self.scene.selectionChanged.connect(self.update_status)
        self.view = IceView(self.scene); self.status = QLabel()
        container = QWidget(); layout = QVBoxLayout(container); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view); layout.addWidget(self.status); self.setCentralWidget(container)
        self._create_actions(); self._create_toolbar(); self.rebuild_scene()

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
        toolbar.addSeparator(); toolbar.addAction(self.fit_action); toolbar.addAction(self.validate_action)

    def rebuild_scene(self) -> None:
        self.scene.clear()
        for index, trap in enumerate(self.document.traps):
            item = TrapItem(index, self.document, self.flip_indices)
            item.setPos(float(trap.center[0]), float(-trap.center[1]))
            item.setToolTip(f"id {trap.id} | center ({trap.center[0]:g}, {trap.center[1]:g})")
            self.scene.addItem(item)
        margin = max(self.document.trap_separation, 1.0)
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-margin, -margin, margin, margin))
        self.fit_scene(); self.update_status()

    def refresh_items(self) -> None:
        for item in self.scene.items():
            if isinstance(item, TrapItem): item.update()
        self.update_status()

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
