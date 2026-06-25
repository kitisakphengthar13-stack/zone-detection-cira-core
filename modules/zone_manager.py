from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


Point = tuple[float, float]
BBox = tuple[float, float, float, float]
Polygon = list[Point]
EPSILON = 1e-9


class ZoneError(ValueError):
    """Raised when zones.json is invalid."""


class ZoneManager:
    def __init__(self, warning_zone: Polygon | None = None, danger_zone: Polygon | None = None) -> None:
        self.warning_zone = warning_zone or []
        self.danger_zone = danger_zone or []
        self.validate()

    @classmethod
    def load(cls, path: str | Path) -> "ZoneManager":
        zone_path = Path(path)
        if not zone_path.exists():
            raise ZoneError(f"Zone file not found: {zone_path}")

        with zone_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if "zones" in data and isinstance(data["zones"], dict):
            data = data["zones"]

        return cls(
            warning_zone=_normalize_polygon(data.get("WARNING", []), "WARNING"),
            danger_zone=_normalize_polygon(data.get("DANGER", []), "DANGER"),
        )

    def save(self, path: str | Path) -> None:
        self.validate()
        zone_path = Path(path)
        zone_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "WARNING": [[float(x), float(y)] for x, y in self.warning_zone],
            "DANGER": [[float(x), float(y)] for x, y in self.danger_zone],
        }
        with zone_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")

    def validate(self) -> None:
        _validate_polygon(self.warning_zone, "WARNING")
        _validate_polygon(self.danger_zone, "DANGER")

    def classify_point(self, point: Point) -> str:
        if point_in_polygon(point, self.danger_zone):
            return "DANGER"
        if point_in_polygon(point, self.warning_zone):
            return "WARNING"
        return "SAFE"

    def classify_bbox(self, bbox: BBox) -> str:
        bbox_polygon = bbox_to_polygon(bbox)
        if polygons_intersect(bbox_polygon, self.danger_zone):
            return "DANGER"
        if polygons_intersect(bbox_polygon, self.warning_zone):
            return "WARNING"
        return "SAFE"

    def zones(self) -> dict[str, Polygon]:
        return {"WARNING": self.warning_zone, "DANGER": self.danger_zone}


def bbox_to_polygon(bbox: BBox) -> Polygon:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def polygons_intersect(first: Iterable[Point], second: Iterable[Point]) -> bool:
    first_polygon = list(first)
    second_polygon = list(second)
    if len(first_polygon) < 3 or len(second_polygon) < 3:
        return False

    if any(point_in_polygon(point, second_polygon) for point in first_polygon):
        return True
    if any(point_in_polygon(point, first_polygon) for point in second_polygon):
        return True

    for first_start, first_end in _polygon_edges(first_polygon):
        for second_start, second_end in _polygon_edges(second_polygon):
            if _segments_intersect(first_start, first_end, second_start, second_end):
                return True
    return False


def point_in_polygon(point: Point, polygon: Iterable[Point]) -> bool:
    polygon = list(polygon)
    if len(polygon) < 3:
        return False

    try:
        import cv2

        contour = np.array(polygon, dtype=np.float32)
        return cv2.pointPolygonTest(contour, point, False) >= 0
    except ImportError:
        return _point_in_polygon_fallback(point, polygon)


def _point_in_polygon_fallback(point: Point, polygon: Polygon) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if _point_on_segment(x, y, previous_x, previous_y, current_x, current_y):
            return True
        intersects = (current_y > y) != (previous_y > y)
        if intersects:
            x_intersection = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x <= x_intersection:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _point_on_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
    if abs(cross) > EPSILON:
        return False
    return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)


def _polygon_edges(polygon: Polygon) -> Iterable[tuple[Point, Point]]:
    for index, start in enumerate(polygon):
        yield start, polygon[(index + 1) % len(polygon)]


def _segments_intersect(first_start: Point, first_end: Point, second_start: Point, second_end: Point) -> bool:
    if _point_on_segment(second_start[0], second_start[1], first_start[0], first_start[1], first_end[0], first_end[1]):
        return True
    if _point_on_segment(second_end[0], second_end[1], first_start[0], first_start[1], first_end[0], first_end[1]):
        return True
    if _point_on_segment(first_start[0], first_start[1], second_start[0], second_start[1], second_end[0], second_end[1]):
        return True
    if _point_on_segment(first_end[0], first_end[1], second_start[0], second_start[1], second_end[0], second_end[1]):
        return True

    first_cross_start = _cross(first_start, first_end, second_start)
    first_cross_end = _cross(first_start, first_end, second_end)
    second_cross_start = _cross(second_start, second_end, first_start)
    second_cross_end = _cross(second_start, second_end, first_end)
    return (
        first_cross_start * first_cross_end < -EPSILON
        and second_cross_start * second_cross_end < -EPSILON
    )


def _cross(origin: Point, end: Point, point: Point) -> float:
    return (end[0] - origin[0]) * (point[1] - origin[1]) - (end[1] - origin[1]) * (point[0] - origin[0])


def _normalize_polygon(value: object, name: str) -> Polygon:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ZoneError(f"{name} zone must be a list of points")
    normalized = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ZoneError(f"{name} point #{index} must be [x, y]")
        normalized.append((float(point[0]), float(point[1])))
    return normalized


def _validate_polygon(polygon: Polygon, name: str) -> None:
    if polygon and len(polygon) < 3:
        raise ZoneError(f"{name} zone must have at least 3 points or be empty")
