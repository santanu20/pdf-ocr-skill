"""Unit tests for CoordinateTransform.

Per testing.md §5: golden-output tests with explicit tolerance.
Per testing.md §3.1: ≥90% strong assertions, no raw float ==.

Run: pytest scripts/scientific_parser/tests/test_coords.py -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.ingest.coords import (
    CoordinateTransform,
    make_transform_from_page,
)

# === Constructor / validation ========================================= #


def test_construct_valid_transform_at_150_dpi_returns_object():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    assert t.pdf_w_pt == pytest.approx(595.0)
    assert t.pdf_h_pt == pytest.approx(842.0)
    assert t.render_dpi == pytest.approx(150.0)


def test_construct_zero_pdf_width_raises_value_error_with_field_name():
    with pytest.raises(ValueError, match="pdf_w_pt"):
        CoordinateTransform(pdf_w_pt=0.0, pdf_h_pt=842.0, render_dpi=150.0)


def test_construct_negative_pdf_height_raises_value_error_with_field_name():
    with pytest.raises(ValueError, match="pdf_h_pt"):
        CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=-1.0, render_dpi=150.0)


def test_construct_zero_dpi_raises_value_error_with_field_name():
    with pytest.raises(ValueError, match="render_dpi"):
        CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=0.0)


def test_construct_negative_dpi_raises_value_error():
    with pytest.raises(ValueError, match="render_dpi"):
        CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=-72.0)


# === Scale factor ==================================================== #


def test_scale_at_72_dpi_is_identity_one_to_one():
    """DPI 72 = PDF native: 1 pt == 1 px. Scale factor must be 1.0 exactly."""
    t = CoordinateTransform(pdf_w_pt=100.0, pdf_h_pt=100.0, render_dpi=72.0)
    assert t.scale == pytest.approx(1.0)


def test_scale_at_150_dpi_is_150_over_72():
    t = CoordinateTransform(pdf_w_pt=100.0, pdf_h_pt=100.0, render_dpi=150.0)
    expected = 150.0 / 72.0
    assert t.scale == pytest.approx(expected, rel=1e-9)


def test_scale_at_300_dpi_is_300_over_72():
    t = CoordinateTransform(pdf_w_pt=100.0, pdf_h_pt=100.0, render_dpi=300.0)
    assert t.scale == pytest.approx(300.0 / 72.0, rel=1e-9)


def test_render_dimensions_at_150_dpi_a4_match_known_values():
    """A4 = 595x842 pt. At 150 DPI = 1240x1754 px (the legacy magic numbers).

    This proves our transform reproduces the OLD hardcoded behavior exactly
    when render_dpi=150 on A4. Ref: region_dispatcher.py:97-98 legacy code.
    """
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    assert t.render_w_px == pytest.approx(1240.0, abs=0.5)
    assert t.render_h_px == pytest.approx(1754.0, abs=0.5)


# === pdf_to_render =================================================== #


def test_pdf_to_render_at_72_dpi_is_identity_transform():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=72.0)
    bbox_in = (10.0, 20.0, 100.0, 200.0)
    bbox_out = t.pdf_to_render(bbox_in)
    assert bbox_out == pytest.approx(bbox_in)


def test_pdf_to_render_at_150_dpi_scales_by_dpi_over_72():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    bbox_in = (10.0, 20.0, 100.0, 200.0)
    s = 150.0 / 72.0
    expected = tuple(v * s for v in bbox_in)
    assert t.pdf_to_render(bbox_in) == pytest.approx(expected)


def test_pdf_to_render_degenerate_bbox_x_inverted_raises_value_error():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    with pytest.raises(ValueError, match="x1"):
        t.pdf_to_render((100.0, 20.0, 50.0, 200.0))


def test_pdf_to_render_degenerate_bbox_y_inverted_raises_value_error():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    with pytest.raises(ValueError, match="y1"):
        t.pdf_to_render((10.0, 200.0, 100.0, 20.0))


def test_pdf_to_render_zero_area_bbox_raises_value_error():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    with pytest.raises(ValueError, match="x1"):
        t.pdf_to_render((10.0, 20.0, 10.0, 200.0))


def test_pdf_to_render_three_tuple_raises_value_error():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    with pytest.raises(ValueError, match="length-4"):
        t.pdf_to_render((10.0, 20.0, 100.0))  # type: ignore[arg-type]


def test_pdf_to_render_nan_component_raises_value_error():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    with pytest.raises(ValueError, match="NaN"):
        t.pdf_to_render((10.0, float("nan"), 100.0, 200.0))


# === Roundtrip identity ============================================= #


def test_roundtrip_pdf_to_render_to_pdf_recovers_input_at_96_dpi():
    """render_to_pdf ∘ pdf_to_render = identity (within float tolerance)."""
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=96.0)
    bbox_in = (72.0, 144.0, 360.0, 720.0)
    recovered = t.render_to_pdf(t.pdf_to_render(bbox_in))
    assert recovered == pytest.approx(bbox_in, rel=1e-9)


def test_roundtrip_at_300_dpi_preserves_bbox_within_relative_tolerance():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=300.0)
    bbox_in = (50.0, 60.0, 500.0, 700.0)
    recovered = t.render_to_pdf(t.pdf_to_render(bbox_in))
    assert recovered == pytest.approx(bbox_in, rel=1e-9)


# === clip_rect_pdf ================================================== #


def test_clip_rect_pdf_in_bounds_returns_input_unchanged():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    bbox_in = (10.0, 20.0, 100.0, 200.0)
    assert t.clip_rect_pdf(bbox_in) == pytest.approx(bbox_in)


def test_clip_rect_pdf_out_of_bounds_clamps_to_page_dimensions():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    # x1=1000 > 595, y1=1000 > 842
    clipped = t.clip_rect_pdf((10.0, 20.0, 1000.0, 1000.0))
    assert clipped == pytest.approx((10.0, 20.0, 595.0, 842.0))


def test_clip_rect_pdf_negative_origin_clamps_to_zero():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    clipped = t.clip_rect_pdf((-50.0, -30.0, 100.0, 200.0))
    assert clipped == pytest.approx((0.0, 0.0, 100.0, 200.0))


def test_clip_rect_pdf_degenerate_input_raises_value_error():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    with pytest.raises(ValueError, match="x1"):
        t.clip_rect_pdf((100.0, 20.0, 50.0, 200.0))


# === make_transform_from_page ===================================== #


def test_make_transform_from_page_reads_rect_dimensions():
    class FakePage:
        class rect:
            width = 595.0
            height = 842.0

    t = make_transform_from_page(FakePage(), render_dpi=150.0)
    assert t.pdf_w_pt == pytest.approx(595.0)
    assert t.pdf_h_pt == pytest.approx(842.0)
    assert t.render_dpi == pytest.approx(150.0)


def test_make_transform_from_page_default_dpi_is_150():
    class FakePage:
        class rect:
            width = 595.0
            height = 842.0

    t = make_transform_from_page(FakePage())
    assert t.render_dpi == pytest.approx(150.0)


def test_make_transform_from_page_zero_dimension_raises_value_error():
    class FakePage:
        class rect:
            width = 0.0
            height = 842.0

    with pytest.raises(ValueError, match="pdf_w_pt"):
        make_transform_from_page(FakePage(), render_dpi=150.0)


# === Frozen dataclass immutability ================================ #


def test_transform_is_immutable_frozen_dataclass_rejects_assignment():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    with pytest.raises((AttributeError, Exception)):
        t.render_dpi = 300.0  # type: ignore[misc]


# === Specialist crop/resize spaces (reviewer #4) ================== #


def test_default_crop_offset_resize_are_identity():
    t = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    assert t.crop_offset == (0.0, 0.0)
    assert t.resize_scale == 1.0
    # identity: render_to_model_input == input
    b = (10.0, 20.0, 30.0, 40.0)
    assert t.render_to_model_input(b) == b


def test_render_to_model_input_applies_offset_and_resize():
    t = CoordinateTransform(
        pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0,
        crop_offset=(100.0, 200.0), resize_scale=0.5,
    )
    # render bbox (110, 220, 130, 240) -> crop-relative (10,20,30,40) -> x0.5
    out = t.render_to_model_input((110.0, 220.0, 130.0, 240.0))
    assert out == (5.0, 10.0, 15.0, 20.0)


def test_model_input_to_render_is_inverse():
    t = CoordinateTransform(
        pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0,
        crop_offset=(100.0, 200.0), resize_scale=0.5,
    )
    original = (110.0, 220.0, 130.0, 240.0)
    mi = t.render_to_model_input(original)
    back = t.model_input_to_render(mi)
    for a, b in zip(original, back):
        assert a == pytest.approx(b)


def test_model_input_to_pdf_round_trips_through_render():
    t = CoordinateTransform(
        pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0,
        crop_offset=(100.0, 200.0), resize_scale=2.0,
    )
    mi = (5.0, 10.0, 15.0, 20.0)
    pdf = t.model_input_to_pdf(mi)
    # render = mi/2 + offset = (102.5, 205, 107.5, 210); pdf = render * 72/150
    inv = 72.0 / 150.0
    assert pdf[0] == pytest.approx(102.5 * inv)


def test_for_crop_builds_child_transform():
    parent = CoordinateTransform(pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0)
    child = parent.for_crop((100.0, 200.0, 400.0, 600.0), target_resize=0.5)
    assert child.crop_offset == (100.0, 200.0)
    assert child.resize_scale == 0.5
    assert child.pdf_w_pt == 595.0  # inherits page dims


def test_invalid_crop_offset_raises():
    with pytest.raises(ValueError):
        CoordinateTransform(
            pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0,
            crop_offset=(1.0,),  # wrong length
        )


def test_invalid_resize_scale_raises():
    with pytest.raises(ValueError):
        CoordinateTransform(
            pdf_w_pt=595.0, pdf_h_pt=842.0, render_dpi=150.0, resize_scale=0.0,
        )
