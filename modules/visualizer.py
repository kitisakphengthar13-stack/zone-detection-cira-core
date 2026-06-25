from __future__ import annotations

from typing import Any

import numpy as np


STATE_COLORS = {
    "SAFE": (0, 180, 0),
    "WARNING": (0, 220, 255),
    "DANGER": (0, 0, 255),
}


def draw_visualization(
    display_frame: np.ndarray,
    *,
    mapper,
    zones: dict[str, list[tuple[float, float]]],
    people: list[dict[str, Any]],
    system_status: dict | None,
    config: dict[str, Any],
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for visualization") from exc

    vis_cfg = config["visualization"]
    output = display_frame.copy()

    if vis_cfg["draw_zone"]:
        _draw_zones(cv2, output, mapper, zones, vis_cfg)

    for person in people:
        state = person["state"]
        color = STATE_COLORS.get(state, (255, 255, 255))
        if vis_cfg["draw_bbox"]:
            x1, y1, x2, y2 = mapper.source_bbox_to_display(person["bbox"])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, int(vis_cfg["bbox_thickness"]))

        if vis_cfg["draw_bottom_center"]:
            center = mapper.source_point_to_display(person["bottom_center"])
            cv2.circle(output, center, int(vis_cfg["point_radius"]), (180, 180, 180), -1)

        if vis_cfg["draw_track_id"]:
            x1, y1, _x2, _y2 = mapper.source_bbox_to_display(person["bbox"])
            label = f"ID {person['track_id']} | RAW {state}"
            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(vis_cfg["font_scale"]),
                color,
                int(vis_cfg["text_thickness"]),
                cv2.LINE_AA,
            )

    if system_status:
        label = f"CONFIRMED STATE: {system_status['state']} | STOP: {system_status['machine_stop']}"
        color = STATE_COLORS.get(system_status["state"], (255, 255, 255))
    else:
        label = "CONFIRMED STATE: WAITING BUFFER"
        color = (255, 255, 255)
    cv2.putText(
        output,
        label,
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(vis_cfg["font_scale"]) * 1.2,
        color,
        int(vis_cfg["text_thickness"]) + 1,
        cv2.LINE_AA,
    )

    return output


def _draw_zones(cv2, frame, mapper, zones, vis_cfg) -> None:
    overlay = frame.copy()
    alpha = float(vis_cfg["zone_alpha"])
    thickness = int(vis_cfg["line_thickness"])

    for name, color in (("WARNING", STATE_COLORS["WARNING"]), ("DANGER", STATE_COLORS["DANGER"])):
        polygon = zones.get(name) or []
        if len(polygon) < 3:
            continue
        points = np.array(mapper.source_polygon_to_display(polygon), dtype=np.int32)
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(frame, [points], True, color, thickness)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
