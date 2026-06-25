from __future__ import annotations

from pathlib import Path
from typing import Any


class InputSourceError(RuntimeError):
    """Raised when a camera or video source cannot be opened."""


class InputSource:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise InputSourceError("opencv-python is required for camera/video input") from exc

        self.cv2 = cv2
        self.config = config
        self.input_type = config["input"]["type"]
        self.cap = self._open()

    def _open(self):
        if self.input_type == "camera":
            camera_cfg = self.config["camera"]
            cap = self.cv2.VideoCapture(int(camera_cfg["index"]))
            cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, int(camera_cfg["width"]))
            cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, int(camera_cfg["height"]))
            cap.set(self.cv2.CAP_PROP_FPS, float(camera_cfg["fps"]))
        elif self.input_type == "video":
            video_path = Path(self.config["_resolved_paths"]["video"])
            cap = self.cv2.VideoCapture(str(video_path))
        else:
            raise InputSourceError(f"Unsupported input.type: {self.input_type}")

        if not cap.isOpened():
            raise InputSourceError(f"Could not open {self.input_type} input")
        return cap

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            return True, frame

        if self.input_type == "video" and bool(self.config["video"]["loop"]):
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            return self.cap.read()

        return False, None

    def fps(self) -> float:
        fps = float(self.cap.get(self.cv2.CAP_PROP_FPS) or 0)
        return fps if fps > 0 else float(self.config["camera"].get("fps", 30))

    def frame_size(self) -> tuple[int, int]:
        width = int(self.cap.get(self.cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self.cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return width, height

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
