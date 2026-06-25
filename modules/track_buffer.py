from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class TrackState:
    buffer_size: int
    states: deque[str] = field(default_factory=deque)
    active: bool = False
    missing_frames: int = 0

    def append(self, state: str) -> None:
        if self.states.maxlen != self.buffer_size:
            self.states = deque(self.states, maxlen=self.buffer_size)
        self.states.append(state)

    def confirmed_state(self) -> str | None:
        if len(self.states) != self.buffer_size:
            return None
        first = self.states[0]
        if all(state == first for state in self.states):
            return first
        return None


class TrackBufferManager:
    def __init__(self, buffer_size: int = 10, max_missing_frames: int = 30) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be zero or greater")
        self.buffer_size = int(buffer_size)
        self.max_missing_frames = int(max_missing_frames)
        self.tracks: dict[int, TrackState] = {}

    def update(self, active_track_states: dict[int, str]) -> None:
        active_ids = {int(track_id) for track_id in active_track_states}

        for track_id, track in list(self.tracks.items()):
            if track_id not in active_ids:
                track.active = False
                track.missing_frames += 1

        for track_id, state in active_track_states.items():
            track_id = int(track_id)
            track = self.tracks.setdefault(
                track_id,
                TrackState(buffer_size=self.buffer_size, states=deque(maxlen=self.buffer_size)),
            )
            track.active = True
            track.missing_frames = 0
            track.append(state)

        self._remove_stale_tracks()

    def get_active_confirmed_states(self) -> dict[int, str]:
        confirmed: dict[int, str] = {}
        for track_id, track in self.tracks.items():
            if not track.active:
                continue
            state = track.confirmed_state()
            if state is not None:
                confirmed[track_id] = state
        return confirmed

    def _remove_stale_tracks(self) -> None:
        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missing_frames > self.max_missing_frames
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]


class EmptySceneConfirmer:
    def __init__(self, enabled: bool = True, confirm_frames: int = 10) -> None:
        if confirm_frames <= 0:
            raise ValueError("confirm_frames must be positive")
        self.enabled = bool(enabled)
        self.confirm_frames = int(confirm_frames)
        self.empty_frames = 0

    def update(self, active_track_count: int) -> bool:
        if not self.enabled:
            self.empty_frames = 0
            return False
        if active_track_count == 0:
            self.empty_frames += 1
        else:
            self.empty_frames = 0
        return self.confirmed

    @property
    def confirmed(self) -> bool:
        return self.enabled and self.empty_frames >= self.confirm_frames
