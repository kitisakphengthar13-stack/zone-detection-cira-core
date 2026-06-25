from __future__ import annotations

from modules.track_buffer import TrackBufferManager


def test_ten_same_safe_confirms():
    manager = TrackBufferManager(buffer_size=10, max_missing_frames=30)
    for _ in range(10):
        manager.update({1: "SAFE"})

    assert manager.get_active_confirmed_states() == {1: "SAFE"}


def test_ten_same_warning_confirms():
    manager = TrackBufferManager(buffer_size=10, max_missing_frames=30)
    for _ in range(10):
        manager.update({1: "WARNING"})

    assert manager.get_active_confirmed_states() == {1: "WARNING"}


def test_ten_same_danger_confirms():
    manager = TrackBufferManager(buffer_size=10, max_missing_frames=30)
    for _ in range(10):
        manager.update({1: "DANGER"})

    assert manager.get_active_confirmed_states() == {1: "DANGER"}


def test_mixed_buffer_does_not_confirm():
    manager = TrackBufferManager(buffer_size=10, max_missing_frames=30)
    for _ in range(9):
        manager.update({1: "SAFE"})
    manager.update({1: "WARNING"})

    assert manager.get_active_confirmed_states() == {}


def test_short_buffer_does_not_confirm():
    manager = TrackBufferManager(buffer_size=10, max_missing_frames=30)
    for _ in range(9):
        manager.update({1: "DANGER"})

    assert manager.get_active_confirmed_states() == {}


def test_missing_track_is_excluded_from_output_but_kept_internally():
    manager = TrackBufferManager(buffer_size=10, max_missing_frames=30)
    for _ in range(10):
        manager.update({1: "SAFE"})

    manager.update({})

    assert manager.get_active_confirmed_states() == {}
    assert 1 in manager.tracks


def test_stale_missing_track_is_removed_after_max_missing_frames():
    manager = TrackBufferManager(buffer_size=10, max_missing_frames=2)
    for _ in range(10):
        manager.update({1: "SAFE"})

    manager.update({})
    manager.update({})
    assert 1 in manager.tracks

    manager.update({})
    assert 1 not in manager.tracks
