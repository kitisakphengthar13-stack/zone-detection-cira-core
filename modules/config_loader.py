from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_SECTIONS = (
    "app",
    "input",
    "camera",
    "video",
    "display",
    "model",
    "tracker",
    "zones",
    "buffer",
    "empty_scene",
    "output",
    "recording",
    "visualization",
)

REQUIRED_FIELDS = {
    "input": ("type",),
    "camera": ("index", "width", "height", "fps"),
    "video": ("path", "loop"),
    "display": ("enabled", "width", "height", "window_name", "keep_aspect_ratio"),
    "model": ("path", "device", "conf_threshold", "iou_threshold", "person_class_id"),
    "tracker": ("type", "config", "persist", "max_missing_frames"),
    "zones": ("file", "check_method", "priority"),
    "buffer": ("size", "confirm_rule", "write_only_when_confirmed"),
    "empty_scene": ("enabled", "confirm_frames", "state", "machine_stop"),
    "output": ("file", "write_mode", "atomic_write"),
    "recording": (
        "enabled",
        "output_dir",
        "save_mode",
        "filename_prefix",
        "codec",
        "fps",
        "use_display_size",
        "max_duration_sec",
    ),
    "visualization": (
        "draw_zone",
        "draw_bbox",
        "draw_track_id",
        "draw_bottom_center",
        "zone_alpha",
        "line_thickness",
        "bbox_thickness",
        "point_radius",
        "font_scale",
        "text_thickness",
    ),
}


class ConfigError(ValueError):
    """Raised when config.yaml is missing required structure."""


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ConfigError(f"Config must be a YAML mapping: {config_path}")

    validate_config(config)
    config["_base_dir"] = config_path.parent
    config["_resolved_paths"] = resolve_paths(config, config_path.parent)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in REQUIRED_TOP_LEVEL_SECTIONS:
        if section not in config:
            raise ConfigError(f"Missing required config section: {section}")
        if not isinstance(config[section], dict):
            raise ConfigError(f"Config section must be a mapping: {section}")

    for section, fields in REQUIRED_FIELDS.items():
        for field in fields:
            if field not in config[section]:
                raise ConfigError(f"Missing required config field: {section}.{field}")

    input_type = config["input"]["type"]
    if input_type not in {"camera", "video"}:
        raise ConfigError("input.type must be 'camera' or 'video'")

    save_mode = config["recording"]["save_mode"]
    if save_mode not in {"raw", "annotated", "both"}:
        raise ConfigError("recording.save_mode must be 'raw', 'annotated', or 'both'")

    confirm_rule = config["buffer"]["confirm_rule"]
    if confirm_rule != "per_track_10_same_frames":
        raise ConfigError("buffer.confirm_rule must be 'per_track_10_same_frames'")


def resolve_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Path]:
    return {
        "video": _resolve(base_dir, config["video"]["path"]),
        "model": _resolve(base_dir, config["model"]["path"]),
        "zones": _resolve(base_dir, config["zones"]["file"]),
        "output": _resolve(base_dir, config["output"]["file"]),
        "recording_output_dir": _resolve(base_dir, config["recording"]["output_dir"]),
    }


def resolve_tracker_config(config: dict[str, Any]) -> str:
    raw_value = str(config["tracker"]["config"])
    base_dir = Path(config.get("_base_dir", "."))
    candidate = _resolve(base_dir, raw_value)
    if candidate.exists() or Path(raw_value).is_absolute():
        return str(candidate)
    return raw_value


def _resolve(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
