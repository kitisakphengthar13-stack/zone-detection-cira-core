from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


STATE_PRIORITY = ("DANGER", "WARNING", "SAFE")
CONFIRM_RULE = "per_track_10_same_frames"


def aggregate_status(
    confirmed_tracks: dict[int, str],
    *,
    buffer_size: int,
    frame_id: int,
    confirm_rule: str = CONFIRM_RULE,
) -> dict | None:
    if not confirmed_tracks:
        return None

    counts = {"SAFE": 0, "WARNING": 0, "DANGER": 0}
    persons = []
    for track_id, state in sorted(confirmed_tracks.items()):
        if state not in counts:
            raise ValueError(f"Unknown person state: {state}")
        counts[state] += 1
        persons.append({"track_id": int(track_id), "state": state, "confirmed": True})

    if counts["DANGER"] > 0:
        system_state = "DANGER"
        machine_stop = 1
    elif counts["WARNING"] > 0:
        system_state = "WARNING"
        machine_stop = 0
    else:
        system_state = "SAFE"
        machine_stop = 0

    return {
        "state": system_state,
        "machine_stop": machine_stop,
        "person_total": len(persons),
        "safe_count": counts["SAFE"],
        "warning_count": counts["WARNING"],
        "danger_count": counts["DANGER"],
        "confirm_rule": confirm_rule,
        "buffer_size": int(buffer_size),
        "frame_id": int(frame_id),
        "persons": persons,
    }


def build_empty_scene_status(
    *,
    buffer_size: int,
    frame_id: int,
    state: str = "SAFE",
    machine_stop: int = 0,
    confirm_rule: str = CONFIRM_RULE,
) -> dict:
    return {
        "state": state,
        "machine_stop": int(machine_stop),
        "person_total": 0,
        "safe_count": 0,
        "warning_count": 0,
        "danger_count": 0,
        "confirm_rule": confirm_rule,
        "buffer_size": int(buffer_size),
        "frame_id": int(frame_id),
        "persons": [],
    }


def atomic_write_json(
    path: str | Path,
    payload: dict,
    *,
    replace_retries: int = 5,
    replace_retry_delay: float = 0.05,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        _replace_with_retries(
            temp_path,
            output_path,
            retries=replace_retries,
            delay=replace_retry_delay,
        )
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


def _replace_with_retries(source: Path, destination: Path, *, retries: int, delay: float) -> None:
    attempts = max(1, int(retries) + 1)
    last_error = None
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(max(0.0, float(delay)))

    raise RuntimeError(
        f"Could not replace output file after {attempts} attempts: {destination}. "
        "The file may be locked by another process such as CIRA Core, VS Code, "
        "Notepad, antivirus, or file indexing."
    ) from last_error


class StatusWriter:
    def __init__(self, output_file: str | Path, atomic_write: bool = True) -> None:
        self.output_file = Path(output_file)
        self.atomic_write = bool(atomic_write)

    def write(self, payload: dict) -> None:
        if self.atomic_write:
            atomic_write_json(self.output_file, payload)
            return

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
