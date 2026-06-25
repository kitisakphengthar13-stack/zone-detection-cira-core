from __future__ import annotations

from typing import Protocol


BBox = tuple[float, float, float, float]
Point = tuple[float, float]


class BBoxClassifier(Protocol):
    def classify_bbox(self, bbox: BBox) -> str:
        ...


def bottom_center_from_bbox(bbox: BBox) -> Point:
    x1, _y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, float(y2))


def classify_bbox(bbox: BBox, zone_manager: BBoxClassifier) -> tuple[str, Point]:
    bottom_center = bottom_center_from_bbox(bbox)
    return zone_manager.classify_bbox(bbox), bottom_center
