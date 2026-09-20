from pathlib import Path
import os
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtTest import QTest

from aci_state_builder.app import MainWindow


class TrajectoryUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _wait_until(predicate, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            QApplication.processEvents()
            if predicate():
                return True
            time.sleep(0.005)
        return False

    def test_async_open_and_frame_change_reuse_graphics_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous_cache = os.environ.get("XDG_CACHE_HOME")
            os.environ["XDG_CACHE_HOME"] = str(root / "cache")
            path = root / "trajectory.csv"
            lines = ["frame,id,x,y,z,dx,dy,dz,t,cx,cy,cz"]
            for frame in range(4):
                for particle in range(8):
                    horizontal = particle < 4
                    dx, dy = ((3, 0) if horizontal else (0, 3))
                    cx, cy = ((1.5 if frame % 2 else -1.5, 0) if horizontal
                              else (0, 1.5 if frame % 2 else -1.5))
                    lines.append(
                        f"{frame},{particle},{particle % 4 * 8},{particle // 4 * 8},0,"
                        f"{dx},{dy},0,{frame * .1},{cx},{cy},0"
                    )
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            window = MainWindow()
            try:
                window.show()
                QApplication.processEvents()
                window._open_trajectory_path(str(path))
                self.assertTrue(self._wait_until(lambda: window.current_trajectory_position == 0))
                items = [id(item) for item in window.trap_items]
                window.comparison_action("mark_a")
                window.request_trajectory_frame(3)
                self.assertTrue(self._wait_until(lambda: window.current_trajectory_position == 3))
                window.comparison_action("mark_b")
                self.assertEqual(items, [id(item) for item in window.trap_items])
                self.assertEqual(window.playback.slider.value(), 3)
                self.assertFalse(window.parameter_panel.trajectory_pane.isHidden())
                self.assertEqual(window.mode_badge.text().strip(), "TRAJECTORY")
                self.assertIsNotNone(window.comparison_item)
                self.assertEqual(window.comparison_a.position, 0)
                self.assertEqual(window.comparison_b.position, 3)
                window.comparison_action("side_by_side")
                QApplication.processEvents()
                dialog = window.comparison_dialog
                self.assertIsNotNone(dialog)
                dialog.left.traps.setChecked(False)
                self.assertFalse(dialog.right.traps.isChecked())
                dialog.left.view.zoom_by(1.25)
                self.assertAlmostEqual(
                    dialog.left.view.transform().m11(),
                    dialog.right.view.transform().m11(),
                )
                trap_id = window.comparison_a.ids[0]
                dialog._highlight(trap_id, True)
                self.assertTrue(dialog.left.items_by_id[trap_id].isSelected())
                self.assertTrue(dialog.right.items_by_id[trap_id].isSelected())
                window.parameter_panel.overlay_pane.traps.setChecked(False)
                self.assertTrue(all(not item.show_body for item in window.trap_items))
                self.assertIsNotNone(window.boundary_item)
                window.parameter_panel.overlay_pane.trails.setChecked(True)
                self.assertIsNotNone(window.trail_item)
                self.assertEqual(window.trail_item.positions.shape, (4, 8, 2))
                png = root / "canvas.png"; svg = root / "canvas.svg"
                window._write_view_image(window.view, png, 1.0, False)
                window._write_view_image(window.view, svg, 1.0, True)
                self.assertGreater(png.stat().st_size, 100)
                self.assertGreater(svg.stat().st_size, 100)
                window.presentation_action.setChecked(True)
                window.toggle_presentation_mode(True)
                self.assertTrue(window.main_toolbar.isHidden())
                self.assertTrue(window.parameter_dock.isHidden())
                window.presentation_action.setChecked(False)
                window.toggle_presentation_mode(False)
                horizontal = window.view.horizontalScrollBar()
                vertical = window.view.verticalScrollBar()
                before = horizontal.value(), vertical.value()
                QTest.mousePress(
                    window.view.viewport(), Qt.MouseButton.MiddleButton,
                    pos=QPoint(160, 140),
                )
                QTest.mouseMove(window.view.viewport(), QPoint(190, 160))
                QTest.mouseRelease(
                    window.view.viewport(), Qt.MouseButton.MiddleButton,
                    pos=QPoint(190, 160),
                )
                self.assertNotEqual((horizontal.value(), vertical.value()), before)
            finally:
                window._leave_trajectory_mode()
                window.thread_pool.waitForDone(5000)
                window.close()
                if previous_cache is None:
                    os.environ.pop("XDG_CACHE_HOME", None)
                else:
                    os.environ["XDG_CACHE_HOME"] = previous_cache

    def test_view_zoom_and_middle_button_pan(self) -> None:
        window = MainWindow()
        try:
            window.show()
            QApplication.processEvents()
            initial_scale = window.view.transform().m11()
            window.zoom_in()
            self.assertAlmostEqual(window.view.transform().m11(), initial_scale * 1.25)
            window.actual_size()
            self.assertAlmostEqual(window.view.transform().m11(), 1.0)

            horizontal = window.view.horizontalScrollBar()
            vertical = window.view.verticalScrollBar()
            horizontal.setValue((horizontal.minimum() + horizontal.maximum()) // 2)
            vertical.setValue((vertical.minimum() + vertical.maximum()) // 2)
            before = horizontal.value(), vertical.value()
            QTest.mousePress(
                window.view.viewport(), Qt.MouseButton.MiddleButton,
                pos=QPoint(160, 140),
            )
            self.assertTrue(window.view._middle_panning)
            self.assertEqual(
                window.view.viewportUpdateMode(),
                window.view.ViewportUpdateMode.FullViewportUpdate,
            )
            QTest.mouseMove(window.view.viewport(), QPoint(185, 155))
            QTest.mouseRelease(
                window.view.viewport(), Qt.MouseButton.MiddleButton,
                pos=QPoint(185, 155),
            )
            self.assertFalse(window.view._middle_panning)
            self.assertEqual(
                window.view.viewportUpdateMode(),
                window.view.ViewportUpdateMode.FullViewportUpdate,
            )
            self.assertNotEqual((horizontal.value(), vertical.value()), before)
        finally:
            window.close()

    def test_dark_application_palette_reaches_canvas(self) -> None:
        original = QApplication.palette()
        dark = QPalette(original)
        dark.setColor(QPalette.ColorRole.Window, QColor("#202225"))
        dark.setColor(QPalette.ColorRole.Base, QColor("#17191b"))
        dark.setColor(QPalette.ColorRole.AlternateBase, QColor("#282b2f"))
        dark.setColor(QPalette.ColorRole.Text, QColor("#f1f3f4"))
        dark.setColor(QPalette.ColorRole.Button, QColor("#30343a"))
        dark.setColor(QPalette.ColorRole.Mid, QColor("#666c73"))
        dark.setColor(QPalette.ColorRole.Highlight, QColor("#4d91d8"))
        dark.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        QApplication.setPalette(dark)
        window = MainWindow()
        try:
            self.assertEqual(
                window.view.backgroundBrush().color(),
                dark.color(QPalette.ColorRole.Base),
            )
        finally:
            window.close()
            QApplication.setPalette(original)


if __name__ == "__main__":
    unittest.main()
