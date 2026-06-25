from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


class VideoRecorder:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.enabled = bool(config["recording"]["enabled"])
        self.save_mode = config["recording"]["save_mode"]
        self.raw_writer = None
        self.annotated_writer = None
        self.frames_written = 0
        self.max_frames = self._max_frames()
        self.cv2 = None

        if self.enabled:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("opencv-python is required for video recording") from exc
            self.cv2 = cv2

    def _max_frames(self) -> int:
        max_duration = float(self.config["recording"]["max_duration_sec"])
        fps = float(self.config["recording"]["fps"])
        if max_duration <= 0:
            return 0
        return int(max_duration * fps)

    def _make_writer(self, suffix: str, frame_size: tuple[int, int]):
        output_dir = Path(self.config["_resolved_paths"]["recording_output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.config["recording"]["filename_prefix"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"{prefix}_{suffix}_{timestamp}.mp4"
        codec = self.config["recording"]["codec"]
        fourcc = self.cv2.VideoWriter_fourcc(*codec)
        fps = float(self.config["recording"]["fps"])
        writer = self.cv2.VideoWriter(str(path), fourcc, fps, frame_size)
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {path}")
        print(f"Recording {suffix} video to {path}")
        return writer

    def write(self, raw_frame=None, annotated_frame=None) -> None:
        if not self.enabled:
            return
        if self.max_frames and self.frames_written >= self.max_frames:
            return

        if self.save_mode in {"raw", "both"} and raw_frame is not None:
            if self.raw_writer is None:
                height, width = raw_frame.shape[:2]
                self.raw_writer = self._make_writer("raw", (width, height))
            self.raw_writer.write(raw_frame)

        if self.save_mode in {"annotated", "both"} and annotated_frame is not None:
            if self.annotated_writer is None:
                height, width = annotated_frame.shape[:2]
                self.annotated_writer = self._make_writer("annotated", (width, height))
            self.annotated_writer.write(annotated_frame)

        self.frames_written += 1

    def release(self) -> None:
        if self.raw_writer is not None:
            self.raw_writer.release()
        if self.annotated_writer is not None:
            self.annotated_writer.release()
