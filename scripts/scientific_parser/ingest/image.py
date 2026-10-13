"""Image preprocessing — normalization, denoising, deskew.

Lightweight wrappers around Pillow. No ML.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional


def normalize_image(
    img_path: str, max_dim: int = 4096, out_path: Optional[str] = None
) -> str:
    """Resize image if exceeds max_dim. Returns path (input or new)."""
    from PIL import Image
    import tempfile, os

    img = Image.open(img_path)
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        return img_path
    scale = max_dim / longest
    new_size = (int(w * scale), int(h * scale))
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix="_normalized.png")
        os.close(fd)
    img_resized.save(out_path)
    return out_path


def to_grayscale(img_path: str, out_path: Optional[str] = None) -> str:
    """Convert to grayscale (helps some OCR engines)."""
    from PIL import Image
    import tempfile, os

    img = Image.open(img_path).convert("L")
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix="_gray.png")
        os.close(fd)
    img.save(out_path)
    return out_path


def autocontrast(img_path: str, cutoff: int = 1, out_path: Optional[str] = None) -> str:
    """Apply autocontrast (stretch histogram)."""
    from PIL import Image, ImageOps
    import tempfile, os

    img = Image.open(img_path).convert("L")
    img = ImageOps.autocontrast(img, cutoff=cutoff)
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix="_contrast.png")
        os.close(fd)
    img.save(out_path)
    return out_path


def image_stats(img_path: str) -> dict:
    """Return basic image stats: size, mode, mean brightness, contrast."""
    from PIL import Image

    img = Image.open(img_path).convert("L")
    px = list(img.getdata())
    n = len(px)
    if n == 0:
        return {"size": img.size, "mode": "L", "mean": 0, "std": 0}
    mean = sum(px) / n
    var = sum((p - mean) ** 2 for p in px) / n
    return {
        "size": img.size,
        "mode": "L",
        "mean": round(mean, 2),
        "std": round(var**0.5, 2),
    }


__all__ = ["normalize_image", "to_grayscale", "autocontrast", "image_stats"]
