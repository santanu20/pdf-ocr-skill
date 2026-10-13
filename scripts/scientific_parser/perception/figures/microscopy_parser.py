"""Microscopy image parser — SEM/TEM/AFM/BSE/CL metadata."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class MicroscopyData:
    microscope_type: Optional[str] = None  # "SEM" | "TEM" | "AFM" | "BSE" | "CL"
    magnification: Optional[str] = None
    scale_bar: Optional[str] = None
    sample_id: Optional[str] = None
    confidence: float = 0.3


def parse_microscopy(
    caption: Optional[str] = None, nearby_text: Optional[str] = None
) -> MicroscopyData:
    import re

    data = MicroscopyData()
    text = " ".join(filter(None, [caption, nearby_text]))
    tl = text.lower()
    if "sem " in tl or "sem-" in tl or "sem image" in tl:
        data.microscope_type = "SEM"
        data.confidence = 0.8
    elif "tem " in tl or "tem-" in tl:
        data.microscope_type = "TEM"
        data.confidence = 0.8
    elif "afm" in tl:
        data.microscope_type = "AFM"
        data.confidence = 0.7
    elif "bse" in tl or "backscatter" in tl:
        data.microscope_type = "BSE"
        data.confidence = 0.7
    elif "cl " in tl or "cathodoluminescence" in tl:
        data.microscope_type = "CL"
        data.confidence = 0.7
    # Magnification pattern: 1000x, 10 kx, ×500
    m = re.search(r"(\d[\d,]*)\s*[×x]\s*(?:mag)?", text, re.IGNORECASE)
    if m:
        data.magnification = m.group(1) + "x"
    return data


__all__ = ["MicroscopyData", "parse_microscopy"]
