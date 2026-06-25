from __future__ import annotations

import argparse
from pathlib import Path

from modules.config_loader import load_config, resolve_tracker_config
from modules.coordinate_mapper import CoordinateMapper
from modules.input_source import InputSource
from modules.person_state import classify_bbox
from modules.status_writer import (
    StatusWriter,
    aggregate_status,
    build_empty_scene_status,
)
from modules.track_buffer import EmptySceneConfirmer, TrackBufferManager
from modules.video_recorder import VideoRecorder
from modules.visualizer import draw_visualization
from modules.zone_manager import ZoneManager


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the zone safety monitor.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml. Defaults to config.yaml beside main.py.",
    )
    return parser.parse_args()


def load_yolo_model(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required to run YOLO tracking") from exc
    return YOLO(str(model_path))


def extract_person_tracks(results, person_class_id: int) -> list[dict]:
    people: list[dict] = []
    if not results:
        return people

    boxes = getattr(results[0], "boxes", None)
    if boxes is None or boxes.id is None:
        return people

    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    track_ids = boxes.id.cpu().numpy().astype(int)

    for bbox, class_id, track_id in zip(xyxy, classes, track_ids):
        if int(class_id) != int(person_class_id):
            continue
        people.append(
            {
                "track_id": int(track_id),
                "bbox": tuple(float(value) for value in bbox),
            }
        )
    return people


def main(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    config = load_config(config_path)
    zones = ZoneManager.load(config["_resolved_paths"]["zones"])
    input_source = InputSource(config)
    model = load_yolo_model(config["_resolved_paths"]["model"])
    track_buffers = TrackBufferManager(
        buffer_size=int(config["buffer"]["size"]),
        max_missing_frames=int(config["tracker"]["max_missing_frames"]),
    )
    empty_scene = EmptySceneConfirmer(
        enabled=bool(config["empty_scene"]["enabled"]),
        confirm_frames=int(config["empty_scene"]["confirm_frames"]),
    )
    status_writer = StatusWriter(
        config["_resolved_paths"]["output"],
        atomic_write=bool(config["output"]["atomic_write"]),
    )
    recorder = VideoRecorder(config)

    display_enabled = bool(config["display"]["enabled"])
    if display_enabled:
        import cv2

        cv2.namedWindow(config["display"]["window_name"], cv2.WINDOW_NORMAL)

    mapper = None
    frame_id = 0

    try:
        while True:
            ok, source_frame = input_source.read()
            if not ok:
                print("Input ended.")
                break

            frame_id += 1
            source_h, source_w = source_frame.shape[:2]
            if mapper is None:
                mapper = CoordinateMapper(
                    source_size=(source_w, source_h),
                    display_size=(int(config["display"]["width"]), int(config["display"]["height"])),
                    keep_aspect_ratio=bool(config["display"]["keep_aspect_ratio"]),
                )

            results = model.track(
                source_frame,
                conf=float(config["model"]["conf_threshold"]),
                iou=float(config["model"]["iou_threshold"]),
                classes=[int(config["model"]["person_class_id"])],
                tracker=resolve_tracker_config(config),
                persist=bool(config["tracker"]["persist"]),
                device=config["model"]["device"],
                verbose=False,
            )

            tracked_people = extract_person_tracks(results, int(config["model"]["person_class_id"]))
            active_states: dict[int, str] = {}
            people_for_display = []
            for person in tracked_people:
                state, bottom_center = classify_bbox(person["bbox"], zones)
                active_states[person["track_id"]] = state
                people_for_display.append(
                    {
                        **person,
                        "state": state,
                        "bottom_center": bottom_center,
                    }
                )

            track_buffers.update(active_states)
            empty_scene_confirmed = empty_scene.update(len(active_states))
            confirmed_tracks = track_buffers.get_active_confirmed_states()

            status = aggregate_status(
                confirmed_tracks,
                buffer_size=int(config["buffer"]["size"]),
                frame_id=frame_id,
                confirm_rule=config["buffer"]["confirm_rule"],
            )
            if status is None and empty_scene_confirmed:
                status = build_empty_scene_status(
                    buffer_size=int(config["buffer"]["size"]),
                    frame_id=frame_id,
                    state=config["empty_scene"]["state"],
                    machine_stop=int(config["empty_scene"]["machine_stop"]),
                    confirm_rule=config["buffer"]["confirm_rule"],
                )

            if status is not None:
                status_writer.write(status)

            annotated_frame = None
            if display_enabled or config["recording"]["save_mode"] in {"annotated", "both"}:
                display_frame = mapper.resize_frame_for_display(source_frame)
                annotated_frame = draw_visualization(
                    display_frame,
                    mapper=mapper,
                    zones=zones.zones(),
                    people=people_for_display,
                    system_status=status,
                    config=config,
                )

            recorder.write(raw_frame=source_frame, annotated_frame=annotated_frame)

            if display_enabled:
                cv2.imshow(config["display"]["window_name"], annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        input_source.release()
        recorder.release()
        if display_enabled:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
