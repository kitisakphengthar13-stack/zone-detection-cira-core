from __future__ import annotations

import json
import os

import pytest

import modules.status_writer as status_writer
from modules.status_writer import (
    aggregate_status,
    atomic_write_json,
    build_empty_scene_status,
)


def test_all_safe_aggregates_to_safe():
    status = aggregate_status({1: "SAFE", 2: "SAFE"}, buffer_size=10, frame_id=12)

    assert status["state"] == "SAFE"
    assert status["machine_stop"] == 0
    assert status["person_total"] == 2
    assert status["safe_count"] == 2
    assert status["warning_count"] == 0
    assert status["danger_count"] == 0


def test_safe_plus_warning_aggregates_to_warning():
    status = aggregate_status({1: "SAFE", 2: "WARNING"}, buffer_size=10, frame_id=12)

    assert status["state"] == "WARNING"
    assert status["machine_stop"] == 0
    assert status["safe_count"] == 1
    assert status["warning_count"] == 1


def test_danger_wins_over_warning_and_safe():
    status = aggregate_status(
        {1: "SAFE", 2: "WARNING", 3: "DANGER"},
        buffer_size=10,
        frame_id=1520,
    )

    assert status == {
        "state": "DANGER",
        "machine_stop": 1,
        "person_total": 3,
        "safe_count": 1,
        "warning_count": 1,
        "danger_count": 1,
        "confirm_rule": "per_track_10_same_frames",
        "buffer_size": 10,
        "frame_id": 1520,
        "persons": [
            {"track_id": 1, "state": "SAFE", "confirmed": True},
            {"track_id": 2, "state": "WARNING", "confirmed": True},
            {"track_id": 3, "state": "DANGER", "confirmed": True},
        ],
    }


def test_no_confirmed_tracks_returns_no_output():
    assert aggregate_status({}, buffer_size=10, frame_id=99) is None


def test_empty_scene_status_shape():
    status = build_empty_scene_status(buffer_size=10, frame_id=1520)

    assert status == {
        "state": "SAFE",
        "machine_stop": 0,
        "person_total": 0,
        "safe_count": 0,
        "warning_count": 0,
        "danger_count": 0,
        "confirm_rule": "per_track_10_same_frames",
        "buffer_size": 10,
        "frame_id": 1520,
        "persons": [],
    }


def test_atomic_write_creates_valid_json(tmp_path):
    output_path = tmp_path / "outputs" / "zone_status.json"
    payload = build_empty_scene_status(buffer_size=10, frame_id=7)

    atomic_write_json(output_path, payload)

    with output_path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    assert loaded == payload


def test_atomic_write_retries_permission_error_then_succeeds(tmp_path, monkeypatch):
    output_path = tmp_path / "outputs" / "zone_status.json"
    payload = build_empty_scene_status(buffer_size=10, frame_id=8)
    calls = []
    real_replace = os.replace

    def flaky_replace(source, destination):
        calls.append((source, destination))
        if len(calls) < 3:
            raise PermissionError("locked")
        real_replace(source, destination)

    monkeypatch.setattr(status_writer.os, "replace", flaky_replace)
    monkeypatch.setattr(status_writer.time, "sleep", lambda _delay: None)

    atomic_write_json(output_path, payload)

    assert len(calls) == 3
    with output_path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    assert loaded == payload


def test_atomic_write_raises_clear_error_and_removes_temp_after_retries(tmp_path, monkeypatch):
    output_path = tmp_path / "outputs" / "zone_status.json"
    payload = build_empty_scene_status(buffer_size=10, frame_id=9)

    def locked_replace(_source, _destination):
        raise PermissionError("locked")

    monkeypatch.setattr(status_writer.os, "replace", locked_replace)
    monkeypatch.setattr(status_writer.time, "sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match="file may be locked by another process"):
        atomic_write_json(output_path, payload, replace_retries=2)

    assert not output_path.exists()
    assert list((tmp_path / "outputs").glob(".zone_status.json.*.tmp")) == []
