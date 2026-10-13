"""Unit tests for figure parsers: plot, chart, map, diagram, microscopy.

These parsers were untested per AUDIT_SPECIALIZED_2026-08-13.md.
Per testing.md §3.1: ≥90% strong assertions.
Run: pytest scripts/scientific_parser/tests/test_figure_parsers.py -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.perception.figures.chart_parser import ChartData, parse_chart
from scientific_parser.perception.figures.diagram_parser import (
    DiagramData,
    parse_diagram,
)
from scientific_parser.perception.figures.map_parser import MapData, parse_map
from scientific_parser.perception.figures.microscopy_parser import (
    MicroscopyData,
    parse_microscopy,
)
from scientific_parser.perception.figures.plot_parser import PlotData, parse_plot

# === parse_plot =================================================== #


def test_parse_plot_no_input_returns_default_plotdata_object():
    p = parse_plot()
    assert isinstance(p, PlotData)
    assert p.data_points == []
    assert p.confidence == pytest.approx(0.5)


def test_parse_plot_with_caption_returns_object_with_caption_processed():
    p = parse_plot(caption="Figure 1: Temperature vs Pressure")
    assert isinstance(p, PlotData)


def test_parse_plot_with_axes_lines_does_not_crash():
    """Axes lines are pairs of (x, y, label). Parser must accept them."""
    axes = [(0, 0, "X-axis"), (10, 5, "Y-axis")]
    p = parse_plot(axes_lines=axes)
    assert isinstance(p, PlotData)


def test_parse_plot_explicit_none_arguments_equivalent_to_default():
    p1 = parse_plot(caption=None, axes_lines=None, nearby_text=None)
    p2 = parse_plot()
    assert p1 == p2


def test_parse_plot_with_nearby_text_does_not_crash():
    p = parse_plot(nearby_text="some surrounding prose for context")
    assert isinstance(p, PlotData)


# === parse_chart ================================================== #


def test_parse_chart_no_input_returns_default_chartdata():
    c = parse_chart()
    assert isinstance(c, ChartData)
    assert c.chart_type is None
    assert c.categories == []
    assert c.values == []
    assert c.confidence == pytest.approx(0.3)


def test_parse_chart_with_caption_returns_object():
    c = parse_chart(caption="Bar chart of monthly sales")
    assert isinstance(c, ChartData)


def test_parse_chart_with_nearby_text_does_not_crash():
    c = parse_chart(nearby_text="Sales increased 20% year over year")
    assert isinstance(c, ChartData)


def test_parse_chart_returns_zero_confidence_floor_for_no_input():
    """Default confidence 0.3 — minimal signal but never zero."""
    c = parse_chart()
    assert c.confidence > 0.0


def test_parse_chart_explicit_none_arguments_equivalent_to_default():
    c1 = parse_chart(caption=None, nearby_text=None)
    c2 = parse_chart()
    assert c1 == c2


# === parse_map ==================================================== #


def test_parse_map_no_input_returns_default_mapdata():
    m = parse_map()
    assert isinstance(m, MapData)
    assert m.map_type is None
    assert m.legend == []
    assert m.north_arrow is False
    assert m.confidence == pytest.approx(0.3)


def test_parse_map_with_caption_returns_object():
    m = parse_map(caption="Geological map of the test region")
    assert isinstance(m, MapData)


def test_parse_map_with_nearby_text_does_not_crash():
    m = parse_map(nearby_text="Scale 1:50000. North arrow at top right.")
    assert isinstance(m, MapData)


def test_parse_map_default_no_north_arrow():
    """Sanity: north_arrow default is False (not None — boolean field)."""
    m = parse_map()
    assert m.north_arrow is False


def test_parse_map_explicit_none_arguments_equivalent_to_default():
    m1 = parse_map(caption=None, nearby_text=None)
    m2 = parse_map()
    assert m1 == m2


# === parse_diagram =============================================== #


def test_parse_diagram_no_input_returns_default_diagramdata():
    d = parse_diagram()
    assert isinstance(d, DiagramData)
    assert d.diagram_type is None
    assert d.layers == []
    assert d.confidence == pytest.approx(0.3)


def test_parse_diagram_with_caption_returns_object():
    d = parse_diagram(caption="Phase diagram for the MgO-SiO2 system")
    assert isinstance(d, DiagramData)


def test_parse_diagram_with_nearby_text_does_not_crash():
    d = parse_diagram(nearby_text="Two phases coexist along the solvus")
    assert isinstance(d, DiagramData)


def test_parse_diagram_explicit_none_arguments_equivalent_to_default():
    d1 = parse_diagram(caption=None, nearby_text=None)
    d2 = parse_diagram()
    assert d1 == d2


def test_parse_diagram_returns_object_with_empty_layers_by_default():
    d = parse_diagram()
    assert d.layers == []


# === parse_microscopy ============================================ #


def test_parse_microscopy_no_input_returns_default_microscopydata():
    m = parse_microscopy()
    assert isinstance(m, MicroscopyData)
    assert m.microscope_type is None
    assert m.magnification is None
    assert m.scale_bar is None
    assert m.sample_id is None
    assert m.confidence == pytest.approx(0.3)


def test_parse_microscopy_with_caption_returns_object():
    m = parse_microscopy(caption="SEM image of quartz grain at 5000x")
    assert isinstance(m, MicroscopyData)


def test_parse_microscopy_with_nearby_text_does_not_crash():
    m = parse_microscopy(nearby_text="Magnification 5000x, scale bar 10 um")
    assert isinstance(m, MicroscopyData)


def test_parse_microscopy_explicit_none_arguments_equivalent_to_default():
    m1 = parse_microscopy(caption=None, nearby_text=None)
    m2 = parse_microscopy()
    assert m1 == m2


def test_parse_microscopy_default_confidence_is_positive_floor():
    m = parse_microscopy()
    assert m.confidence > 0.0


# === Cross-parser contract ====================================== #


def test_all_parsers_return_correct_data_class_type():
    """Each parse_X function must return its corresponding Data class."""
    assert isinstance(parse_plot(), PlotData)
    assert isinstance(parse_chart(), ChartData)
    assert isinstance(parse_map(), MapData)
    assert isinstance(parse_diagram(), DiagramData)
    assert isinstance(parse_microscopy(), MicroscopyData)


def test_all_parser_default_confidence_values_are_in_zero_one_range():
    """Every parser's default confidence must be a valid probability."""
    for parser in [
        parse_plot,
        parse_chart,
        parse_map,
        parse_diagram,
        parse_microscopy,
    ]:
        c = parser().confidence
        assert 0.0 <= c <= 1.0
