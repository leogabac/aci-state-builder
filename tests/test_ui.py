from pathlib import Path
import os
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

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
                window._open_trajectory_path(str(path))
                self.assertTrue(self._wait_until(lambda: window.current_trajectory_position == 0))
                items = [id(item) for item in window.trap_items]
                window.request_trajectory_frame(3)
                self.assertTrue(self._wait_until(lambda: window.current_trajectory_position == 3))
                self.assertEqual(items, [id(item) for item in window.trap_items])
                self.assertEqual(window.playback.slider.value(), 3)
                self.assertFalse(window.parameter_panel.trajectory_pane.isHidden())
            finally:
                window._leave_trajectory_mode()
                window.thread_pool.waitForDone(5000)
                window.close()
                if previous_cache is None:
                    os.environ.pop("XDG_CACHE_HOME", None)
                else:
                    os.environ["XDG_CACHE_HOME"] = previous_cache


if __name__ == "__main__":
    unittest.main()
