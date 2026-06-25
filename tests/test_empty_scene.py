from __future__ import annotations

from modules.track_buffer import EmptySceneConfirmer


def test_empty_scene_confirms_after_configured_frames():
    confirmer = EmptySceneConfirmer(enabled=True, confirm_frames=10)

    for _ in range(9):
        assert confirmer.update(active_track_count=0) is False

    assert confirmer.update(active_track_count=0) is True


def test_empty_scene_resets_when_track_is_active():
    confirmer = EmptySceneConfirmer(enabled=True, confirm_frames=3)

    assert confirmer.update(active_track_count=0) is False
    assert confirmer.update(active_track_count=0) is False
    assert confirmer.update(active_track_count=1) is False
    assert confirmer.update(active_track_count=0) is False
    assert confirmer.update(active_track_count=0) is False
    assert confirmer.update(active_track_count=0) is True
