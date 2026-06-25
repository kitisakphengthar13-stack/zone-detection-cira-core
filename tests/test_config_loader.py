from __future__ import annotations

from pathlib import Path

from modules.config_loader import load_config, resolve_tracker_config


def write_config(config_path: Path, tracker_config: str = "bytetrack.yaml") -> None:
    config_path.write_text(
        f"""
app:
  name: "zone_safety_monitor"

input:
  type: "video"

camera:
  index: 0
  width: 1280
  height: 720
  fps: 30

video:
  path: "videos/input.mp4"
  loop: false

display:
  enabled: false
  width: 1280
  height: 720
  window_name: "Zone Safety Monitor"
  keep_aspect_ratio: true

model:
  path: "models/yolo.pt"
  device: "cpu"
  conf_threshold: 0.5
  iou_threshold: 0.5
  person_class_id: 0

tracker:
  type: "bytetrack"
  config: "{tracker_config}"
  persist: true
  max_missing_frames: 30

zones:
  file: "zones.json"
  check_method: "bottom_center"
  priority: ["DANGER", "WARNING", "SAFE"]

buffer:
  size: 10
  confirm_rule: "per_track_10_same_frames"
  write_only_when_confirmed: true

empty_scene:
  enabled: true
  confirm_frames: 10
  state: "SAFE"
  machine_stop: 0

output:
  file: "outputs/zone_status.json"
  write_mode: "overwrite"
  atomic_write: true

recording:
  enabled: false
  output_dir: "outputs/videos"
  save_mode: "annotated"
  filename_prefix: "zone_safety"
  codec: "mp4v"
  fps: 30
  use_display_size: true
  max_duration_sec: 0

visualization:
  draw_zone: true
  draw_bbox: true
  draw_track_id: true
  draw_bottom_center: true
  zone_alpha: 0.25
  line_thickness: 2
  bbox_thickness: 2
  point_radius: 5
  font_scale: 0.6
  text_thickness: 2
""".lstrip(),
        encoding="utf-8",
    )


def test_relative_config_paths_resolve_from_config_directory(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    other_dir = tmp_path / "other"
    project_dir.mkdir()
    other_dir.mkdir()
    config_path = project_dir / "config.yaml"
    write_config(config_path)

    monkeypatch.chdir(other_dir)
    config = load_config(config_path)

    assert config["_base_dir"] == project_dir
    assert config["_resolved_paths"]["video"] == project_dir / "videos" / "input.mp4"
    assert config["_resolved_paths"]["model"] == project_dir / "models" / "yolo.pt"
    assert config["_resolved_paths"]["zones"] == project_dir / "zones.json"
    assert config["_resolved_paths"]["output"] == project_dir / "outputs" / "zone_status.json"
    assert config["_resolved_paths"]["recording_output_dir"] == project_dir / "outputs" / "videos"


def test_tracker_config_resolves_local_file_but_keeps_builtin_name(tmp_path):
    config_path = tmp_path / "config.yaml"
    local_tracker = tmp_path / "tracker" / "custom.yaml"
    local_tracker.parent.mkdir()
    local_tracker.write_text("tracker_type: bytetrack\n", encoding="utf-8")

    write_config(config_path)
    config = load_config(config_path)
    assert resolve_tracker_config(config) == "bytetrack.yaml"

    write_config(config_path, "tracker/custom.yaml")
    config = load_config(config_path)
    assert resolve_tracker_config(config) == str(local_tracker)
