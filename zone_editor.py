from __future__ import annotations

import argparse
from pathlib import Path

from modules.config_loader import load_config
from modules.coordinate_mapper import CoordinateMapper
from modules.input_source import InputSource
from modules.zone_manager import ZoneManager


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit WARNING and DANGER zones.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml. Defaults to config.yaml beside zone_editor.py.",
    )
    return parser.parse_args()


class ZoneEditor:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for the zone editor") from exc

        self.cv2 = cv2
        self.config = load_config(config_path)
        self.input_source = InputSource(self.config)
        self.zone_path = self.config["_resolved_paths"]["zones"]
        try:
            self.zones = ZoneManager.load(self.zone_path)
        except Exception:
            self.zones = ZoneManager()
        self.current_zone = "WARNING"
        self.current_points: list[tuple[float, float]] = []
        self.mapper = None

    def run(self) -> None:
        window_name = self.config["display"]["window_name"] + " - Zone Editor"
        self.cv2.namedWindow(window_name, self.cv2.WINDOW_NORMAL)
        self.cv2.setMouseCallback(window_name, self.on_mouse)

        try:
            while True:
                ok, source_frame = self.input_source.read()
                if not ok:
                    print("Input ended.")
                    break

                source_h, source_w = source_frame.shape[:2]
                if self.mapper is None:
                    self.mapper = CoordinateMapper(
                        source_size=(source_w, source_h),
                        display_size=(int(self.config["display"]["width"]), int(self.config["display"]["height"])),
                        keep_aspect_ratio=bool(self.config["display"]["keep_aspect_ratio"]),
                    )

                display_frame = self.mapper.resize_frame_for_display(source_frame)
                self.draw_editor(display_frame)
                self.cv2.imshow(window_name, display_frame)
                key = self.cv2.waitKey(1) & 0xFF

                if key == ord("1"):
                    self.current_zone = "WARNING"
                    self.current_points = []
                    print("Drawing WARNING zone")
                elif key == ord("2"):
                    self.current_zone = "DANGER"
                    self.current_points = []
                    print("Drawing DANGER zone")
                elif key == ord("z"):
                    if self.current_points:
                        self.current_points.pop()
                elif key == ord("c"):
                    self.confirm_current_polygon()
                elif key == ord("r"):
                    self.current_points = []
                elif key == ord("s"):
                    self.zones.save(self.zone_path)
                    print(f"Saved zones to {self.zone_path}")
                elif key == ord("q"):
                    break
        finally:
            self.input_source.release()
            self.cv2.destroyAllWindows()

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if event != self.cv2.EVENT_LBUTTONDOWN or self.mapper is None:
            return
        source_point = self.mapper.display_point_to_source((x, y), clamp=True)
        self.current_points.append(source_point)
        print(f"Added {self.current_zone} point: {source_point[0]:.1f}, {source_point[1]:.1f}")

    def confirm_current_polygon(self) -> None:
        if len(self.current_points) < 3:
            print("Need at least 3 points to confirm a polygon")
            return
        if self.current_zone == "WARNING":
            self.zones.warning_zone = list(self.current_points)
        else:
            self.zones.danger_zone = list(self.current_points)
        self.zones.validate()
        print(f"Confirmed {self.current_zone} polygon with {len(self.current_points)} points")
        self.current_points = []

    def draw_editor(self, frame) -> None:
        colors = {"WARNING": (0, 220, 255), "DANGER": (0, 0, 255)}
        for name, polygon in self.zones.zones().items():
            self.draw_polygon(frame, polygon, colors[name], closed=True)
        self.draw_polygon(frame, self.current_points, colors[self.current_zone], closed=False)
        self.cv2.putText(
            frame,
            f"Zone: {self.current_zone} | 1 WARNING  2 DANGER  z undo  c confirm  r reset  s save  q quit",
            (20, 32),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colors[self.current_zone],
            2,
            self.cv2.LINE_AA,
        )

    def draw_polygon(self, frame, source_points, color, closed: bool) -> None:
        if not source_points or self.mapper is None:
            return
        import numpy as np

        display_points = self.mapper.source_polygon_to_display(source_points)
        for point in display_points:
            self.cv2.circle(frame, point, 5, color, -1)
        if len(display_points) >= 2:
            points = np.array(display_points, dtype=np.int32)
            self.cv2.polylines(frame, [points], closed, color, 2)


if __name__ == "__main__":
    args = parse_args()
    ZoneEditor(args.config).run()
