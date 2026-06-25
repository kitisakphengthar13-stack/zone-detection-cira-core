from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Point = tuple[float, float]
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class CoordinateMapper:
    source_size: tuple[int, int]
    display_size: tuple[int, int]
    keep_aspect_ratio: bool = True

    def __post_init__(self) -> None:
        source_w, source_h = self.source_size
        display_w, display_h = self.display_size
        if source_w <= 0 or source_h <= 0:
            raise ValueError("source_size must contain positive width and height")
        if display_w <= 0 or display_h <= 0:
            raise ValueError("display_size must contain positive width and height")

    @property
    def scale_x(self) -> float:
        if self.keep_aspect_ratio:
            return self.scale
        return self.display_size[0] / self.source_size[0]

    @property
    def scale_y(self) -> float:
        if self.keep_aspect_ratio:
            return self.scale
        return self.display_size[1] / self.source_size[1]

    @property
    def scale(self) -> float:
        return min(
            self.display_size[0] / self.source_size[0],
            self.display_size[1] / self.source_size[1],
        )

    @property
    def rendered_size(self) -> tuple[int, int]:
        if not self.keep_aspect_ratio:
            return self.display_size
        width = int(round(self.source_size[0] * self.scale))
        height = int(round(self.source_size[1] * self.scale))
        return width, height

    @property
    def offset(self) -> tuple[int, int]:
        if not self.keep_aspect_ratio:
            return 0, 0
        rendered_w, rendered_h = self.rendered_size
        return (
            (self.display_size[0] - rendered_w) // 2,
            (self.display_size[1] - rendered_h) // 2,
        )

    def source_point_to_display(self, point: Point) -> tuple[int, int]:
        x, y = point
        offset_x, offset_y = self.offset
        return (
            int(round(x * self.scale_x + offset_x)),
            int(round(y * self.scale_y + offset_y)),
        )

    def display_point_to_source(self, point: Point, clamp: bool = True) -> tuple[float, float]:
        x, y = point
        offset_x, offset_y = self.offset
        source_x = (x - offset_x) / self.scale_x
        source_y = (y - offset_y) / self.scale_y

        if clamp:
            source_x = min(max(source_x, 0.0), float(self.source_size[0] - 1))
            source_y = min(max(source_y, 0.0), float(self.source_size[1] - 1))

        return source_x, source_y

    def source_bbox_to_display(self, bbox: BBox) -> tuple[int, int, int, int]:
        x1, y1 = self.source_point_to_display((bbox[0], bbox[1]))
        x2, y2 = self.source_point_to_display((bbox[2], bbox[3]))
        return x1, y1, x2, y2

    def source_polygon_to_display(self, polygon: Iterable[Point]) -> list[tuple[int, int]]:
        return [self.source_point_to_display(point) for point in polygon]

    def resize_frame_for_display(self, source_frame: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for display resizing") from exc

        display_w, display_h = self.display_size
        if not self.keep_aspect_ratio:
            return cv2.resize(source_frame, (display_w, display_h))

        rendered_w, rendered_h = self.rendered_size
        resized = cv2.resize(source_frame, (rendered_w, rendered_h))
        display = np.zeros((display_h, display_w, source_frame.shape[2]), dtype=source_frame.dtype)
        offset_x, offset_y = self.offset
        display[offset_y : offset_y + rendered_h, offset_x : offset_x + rendered_w] = resized
        return display
