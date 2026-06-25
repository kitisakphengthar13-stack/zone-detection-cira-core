from __future__ import annotations

from modules.person_state import bottom_center_from_bbox, classify_bbox
from modules.zone_manager import ZoneManager


def test_classifies_point_with_danger_priority_when_zones_overlap():
    manager = ZoneManager(
        warning_zone=[(0, 0), (100, 0), (100, 100), (0, 100)],
        danger_zone=[(50, 50), (150, 50), (150, 150), (50, 150)],
    )

    assert manager.classify_point((75, 75)) == "DANGER"


def test_classifies_warning_and_safe_points():
    manager = ZoneManager(
        warning_zone=[(0, 0), (100, 0), (100, 100), (0, 100)],
        danger_zone=[(200, 200), (300, 200), (300, 300), (200, 300)],
    )

    assert manager.classify_point((20, 20)) == "WARNING"
    assert manager.classify_point((150, 150)) == "SAFE"


def test_bottom_center_logic_and_bbox_classification():
    manager = ZoneManager(
        warning_zone=[],
        danger_zone=[(40, 10), (80, 10), (80, 30), (40, 30)],
    )
    bbox = (0, 0, 50, 100)

    assert bottom_center_from_bbox(bbox) == (25, 100)
    assert classify_bbox(bbox, manager) == ("DANGER", (25, 100))


def test_bbox_fully_outside_zones_classifies_safe():
    manager = ZoneManager(
        warning_zone=[(100, 100), (200, 100), (200, 200), (100, 200)],
        danger_zone=[(300, 100), (400, 100), (400, 200), (300, 200)],
    )

    assert classify_bbox((0, 0, 50, 50), manager)[0] == "SAFE"


def test_bbox_partially_overlaps_warning_classifies_warning():
    manager = ZoneManager(
        warning_zone=[(100, 100), (200, 100), (200, 200), (100, 200)],
        danger_zone=[],
    )

    assert classify_bbox((50, 120, 120, 180), manager)[0] == "WARNING"


def test_bbox_partially_overlaps_danger_classifies_danger():
    manager = ZoneManager(
        warning_zone=[],
        danger_zone=[(300, 100), (400, 100), (400, 200), (300, 200)],
    )

    assert classify_bbox((250, 120, 320, 180), manager)[0] == "DANGER"


def test_bbox_touching_warning_edge_classifies_warning():
    manager = ZoneManager(
        warning_zone=[(100, 100), (200, 100), (200, 200), (100, 200)],
        danger_zone=[],
    )

    assert classify_bbox((50, 120, 100, 180), manager)[0] == "WARNING"


def test_bbox_touching_danger_edge_classifies_danger():
    manager = ZoneManager(
        warning_zone=[],
        danger_zone=[(300, 100), (400, 100), (400, 200), (300, 200)],
    )

    assert classify_bbox((250, 120, 300, 180), manager)[0] == "DANGER"


def test_bbox_overlapping_warning_and_danger_classifies_danger():
    manager = ZoneManager(
        warning_zone=[(100, 100), (250, 100), (250, 250), (100, 250)],
        danger_zone=[(200, 100), (300, 100), (300, 200), (200, 200)],
    )

    assert classify_bbox((190, 120, 210, 180), manager)[0] == "DANGER"
