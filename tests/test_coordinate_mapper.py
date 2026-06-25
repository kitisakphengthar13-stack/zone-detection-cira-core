from __future__ import annotations

import numpy as np

from modules.coordinate_mapper import CoordinateMapper


def test_letterbox_maps_source_to_display_with_offsets():
    mapper = CoordinateMapper(
        source_size=(1920, 1080),
        display_size=(1280, 720),
        keep_aspect_ratio=True,
    )

    assert mapper.source_point_to_display((960, 540)) == (640, 360)
    assert mapper.display_point_to_source((640, 360)) == (960, 540)
    assert mapper.source_bbox_to_display((0, 0, 1920, 1080)) == (0, 0, 1280, 720)


def test_letterbox_maps_with_vertical_padding():
    mapper = CoordinateMapper(
        source_size=(1000, 1000),
        display_size=(1280, 720),
        keep_aspect_ratio=True,
    )

    assert mapper.rendered_size == (720, 720)
    assert mapper.offset == (280, 0)
    assert mapper.source_point_to_display((0, 0)) == (280, 0)
    assert mapper.display_point_to_source((280, 0)) == (0, 0)


def test_non_aspect_mapping_scales_x_and_y_independently():
    mapper = CoordinateMapper(
        source_size=(100, 200),
        display_size=(200, 100),
        keep_aspect_ratio=False,
    )

    assert mapper.source_point_to_display((50, 100)) == (100, 50)
    assert mapper.display_point_to_source((100, 50)) == (50, 100)


def test_resize_frame_for_display_preserves_target_size():
    mapper = CoordinateMapper(
        source_size=(100, 100),
        display_size=(200, 100),
        keep_aspect_ratio=True,
    )
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 255

    display = mapper.resize_frame_for_display(frame)

    assert display.shape == (100, 200, 3)
