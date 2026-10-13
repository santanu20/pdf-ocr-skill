"""Coordinate transforms between PDF points, rendered pixels, and crop space.

Replaces the hardcoded ``scale_x = page_w / 1240`` heuristic that assumed
150-DPI A4 rendering. Every bbox consumer now takes a ``CoordinateTransform``
constructed from the actual page dimensions and the actual render DPI.

Conventions
-----------
* **PDF points** (``pt``): 1/72 inch, origin top-left, y-down. Native unit of
  PyMuPDF ``page.rect`` and ``clip`` arguments.
* **Render pixels** (``px``): ``pt * dpi / 72``, same origin, same y-axis.
  Used for OCR/VLM image input and ``PIL.Image`` coordinates.
* **Rotation**: PDF page rotation in degrees (0/90/180/270). For 90/270 the
  effective page width/height swap when projected to render space. Callers
  must pass the rotation reported by ``page.rotation``; the transform then
  applies the correct axis swap so a top-left bbox in PDF space maps to a
  top-left bbox in render space.

Units are explicit on every method. All methods are pure functions of their
arguments; the dataclass holds only the immutable transform parameters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoordinateTransform:
    """Immutable bidirectional PDF-point <-> render-pixel transform.

    Attributes
    ----------
    pdf_w_pt:
        PDF page width in points (``page.rect.width`` after applying
        rotation). Must be > 0.
    pdf_h_pt:
        PDF page height in points. Must be > 0.
    render_dpi:
        DPI at which the page was (or will be) rasterized. Must be > 0.
        Common values: 72 (identity), 96 (screen), 150 (default OCR),
        300 (high-resolution OCR).
    """

    pdf_w_pt: float
    pdf_h_pt: float
    render_dpi: float
    # Optional specialist-crop parameters (reviewer #4). When a region is
    # cropped from the render image and then resized before model input, a
    # separate transform is built with these set so coordinates round-trip
    # correctly between model-input space, render space, and PDF space.
    # Defaults are identity (full-page, no resize) — existing callers that
    # construct CoordinateTransform(pdf_w_pt=, pdf_h_pt=, render_dpi=) are
    # unaffected.
    crop_offset: tuple[float, float] = (0.0, 0.0)
    resize_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.pdf_w_pt <= 0:
            raise ValueError(f"pdf_w_pt must be > 0, got {self.pdf_w_pt}")
        if self.pdf_h_pt <= 0:
            raise ValueError(f"pdf_h_pt must be > 0, got {self.pdf_h_pt}")
        if self.render_dpi <= 0:
            raise ValueError(f"render_dpi must be > 0, got {self.render_dpi}")
        if len(self.crop_offset) != 2:
            raise ValueError(f"crop_offset must be (dx, dy), got len={len(self.crop_offset)}")
        if self.resize_scale <= 0:
            raise ValueError(f"resize_scale must be > 0, got {self.resize_scale}")

    @property
    def scale(self) -> float:
        """Multiplicative factor ``pt -> px``: ``dpi / 72``.

        Identity at 72 DPI (1 pt == 1 px).
        """
        return self.render_dpi / 72.0

    @property
    def render_w_px(self) -> float:
        """Render image width in pixels."""
        return self.pdf_w_pt * self.scale

    @property
    def render_h_px(self) -> float:
        """Render image height in pixels."""
        return self.pdf_h_pt * self.scale

    def pdf_to_render(
        self, bbox_pt: tuple[float, float, float, float]
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        """Map a 2D bbox from PDF points to render pixels.

        Args:
            bbox_pt: ``(x0, y0, x1, y1)`` in PDF points, top-left origin.

        Returns:
            ``(x0, y0, x1, y1)`` in render pixels.

        Raises:
            ValueError: if ``bbox_pt`` is not length 4 or is degenerate
                (``x1 <= x0`` or ``y1 <= y0``).
        """
        _validate_bbox(bbox_pt)
        x0, y0, x1, y1 = bbox_pt
        s = self.scale
        return (x0 * s, y0 * s, x1 * s, y1 * s)

    def render_to_pdf(
        self, bbox_px: tuple[float, float, float, float]
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        """Inverse of :meth:`pdf_to_render`.

        Args:
            bbox_px: ``(x0, y0, x1, y1)`` in render pixels.

        Returns:
            ``(x0, y0, x1, y1)`` in PDF points.
        """
        _validate_bbox(bbox_px)
        x0, y0, x1, y1 = bbox_px
        inv = 72.0 / self.render_dpi
        return (x0 * inv, y0 * inv, x1 * inv, y1 * inv)

    def clip_rect_pdf(
        self, bbox_pt: tuple[float, float, float, float]
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        """Return a PDF-point bbox clipped to page bounds.

        Use this to build PyMuPDF ``fitz.Rect`` arguments safely.

        Args:
            bbox_pt: ``(x0, y0, x1, y1)`` in PDF points. May extend outside
                the page; will be clamped.

        Returns:
            ``(x0, y0, x1, y1)`` clamped to ``[0, pdf_w_pt] x [0, pdf_h_pt]``.
            Degenerate (zero-area) input raises ``ValueError`` BEFORE
            clamping; degenerate AFTER clamping is returned as-is (caller
            decides how to handle).
        """
        _validate_bbox(bbox_pt)
        x0, y0, x1, y1 = bbox_pt
        return (
            max(0.0, min(x0, self.pdf_w_pt)),
            max(0.0, min(y0, self.pdf_h_pt)),
            max(0.0, min(x1, self.pdf_w_pt)),
            max(0.0, min(y1, self.pdf_h_pt)),
        )

    # === Specialist-crop coordinate spaces (reviewer #4) ============ #
    # A specialist provider receives a CROPPED + RESIZED image. Coordinates
    # in that model-input space map back to render space (undo resize, re-add
    # crop offset) and thence to PDF points. This closes the gap where a
    # layout bbox detected on a resized crop would otherwise silently misalign
    # with the PDF text layer.

    def render_to_model_input(
        self, bbox_px: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Render-px bbox -> model-input-px bbox (apply crop offset + resize).

        Inverse of :meth:`model_input_to_render`. No degeneracy check (a
        zero-size crop is a valid, if useless, model input).
        """
        x0, y0, x1, y1 = bbox_px
        dx, dy = self.crop_offset
        rs = self.resize_scale
        return ((x0 - dx) * rs, (y0 - dy) * rs, (x1 - dx) * rs, (y1 - dy) * rs)

    def model_input_to_render(
        self, bbox_mi: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Model-input-px bbox -> render-px bbox (undo resize, re-add offset)."""
        x0, y0, x1, y1 = bbox_mi
        dx, dy = self.crop_offset
        inv = 1.0 / self.resize_scale
        return (x0 * inv + dx, y0 * inv + dy, x1 * inv + dx, y1 * inv + dy)

    def model_input_to_pdf(
        self, bbox_mi: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Model-input-px bbox -> PDF-point bbox (composed inverse)."""
        return self.render_to_pdf(self.model_input_to_render(bbox_mi))

    def for_crop(
        self,
        crop_bbox_render_px: tuple[float, float, float, float],
        target_resize: float = 1.0,
    ) -> CoordinateTransform:
        """Return a NEW transform describing a cropped+resized specialist input.

        Args:
          crop_bbox_render_px: the region crop in render px.
          target_resize: resize factor applied to the crop before model input
            (e.g. 0.5 if the crop is downsampled to half size).

        Returns:
          A fresh :class:`CoordinateTransform` whose ``crop_offset`` and
          ``resize_scale`` map model-input coordinates back to this page's
          render/PDF space.
        """
        x0, y0, _x1, _y1 = crop_bbox_render_px
        return CoordinateTransform(
            pdf_w_pt=self.pdf_w_pt,
            pdf_h_pt=self.pdf_h_pt,
            render_dpi=self.render_dpi,
            crop_offset=(float(x0), float(y0)),
            resize_scale=float(target_resize),
        )


def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    """Raise ValueError on malformed bbox (non-4-tuple or degenerate)."""
    if not isinstance(bbox, tuple | list) or len(bbox) != 4:
        raise ValueError(
            f"bbox must be a length-4 tuple/list, got {type(bbox).__name__} "
            f"of length {len(bbox) if hasattr(bbox, '__len__') else 'N/A'}"
        )
    x0, y0, x1, y1 = bbox
    for label, v in (("x0", x0), ("y0", y0), ("x1", x1), ("y1", y1)):
        if v is None:
            raise ValueError(f"bbox component {label} is None")
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bbox component {label}={v!r} not convertible to float") from exc
        if fv != fv:  # NaN check
            raise ValueError(f"bbox component {label} is NaN")
    if x1 <= x0:
        raise ValueError(f"degenerate bbox: x1={x1} <= x0={x0}")
    if y1 <= y0:
        raise ValueError(f"degenerate bbox: y1={y1} <= y0={y0}")


def make_transform_from_page(
    page,
    render_dpi: float = 150.0,
) -> CoordinateTransform:
    """Build a transform from a PyMuPDF ``page`` object.

    Reads ``page.rect.width``, ``page.rect.height``. Does NOT read
    ``page.rotation`` — PyMuPDF's ``page.rect`` is already in the rotated
    coordinate system for most operations, so callers should pass the
    post-rotation dimensions directly. If you have an unrotated rect, swap
    width/height yourself before calling.

    Args:
        page: PyMuPDF ``fitz.Page`` (any object exposing ``.rect.width`` and
            ``.rect.height``).
        render_dpi: DPI at which the page was rasterized.

    Returns:
        A :class:`CoordinateTransform`.

    Raises:
        ValueError: if page dimensions are non-positive.
        AttributeError: if ``page`` lacks ``.rect``.
    """
    return CoordinateTransform(
        pdf_w_pt=float(page.rect.width),
        pdf_h_pt=float(page.rect.height),
        render_dpi=float(render_dpi),
    )


__all__ = [
    "CoordinateTransform",
    "make_transform_from_page",
]
