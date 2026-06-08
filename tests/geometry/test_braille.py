import math

import numpy as np
import pytest
from PIL import Image
from pytest_regressions.image_regression import ImageRegressionFixture
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from tactilegen.geometry.braille import (
    STANDARD_CELL_SPACING,
    STANDARD_DOT_DIAMETER,
    STANDARD_DOT_RADIUS,
    STANDARD_INTRA_CELL_SPACING,
    STANDARD_LINE_SPACING,
    BrailleGeometry,
    BraillePlacement,
    create_braille_displacement_map,
    create_standard_braille_displacement_map,
    draw_pending_box,
    render_braille,
    render_braille_on_image,
    render_standard_braille,
    render_standard_braille_on_image,
)

# =============================================================================
# Regression test inputs
# =============================================================================


def _make_image_base(width: int = 400, height: int = 200) -> Image.Image:
    gradient = np.linspace(40, 200, width, dtype=np.uint8)
    arr = np.tile(gradient, (height, 1))
    arr = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr)


def _full_canvas_placement(text: str, w: int, h: int, padding: float = 0.1):
    return BraillePlacement(text=text, x=0, y=0, width=w, height=h, padding=padding)


_RENDER_BRAILLE_RECIPES: list[tuple[str, dict]] = [
    (
        "preview_hello",
        {
            "text": "hello",
            "width": 320,
            "height": 80,
            "dot_color": (30, 30, 30),
            "bg_color": (245, 245, 245),
        },
    ),
    (
        "preview_uppercase_world",
        {
            "text": "Hello World",
            "width": 480,
            "height": 80,
            "dot_color": (30, 30, 30),
            "bg_color": (255, 255, 255),
        },
    ),
    (
        "preview_digits",
        {
            "text": "Room 314",
            "width": 480,
            "height": 80,
            "dot_color": (0, 0, 0),
            "bg_color": (255, 255, 255),
        },
    ),
]

_BRAILLE_PLACEMENTS = [
    BraillePlacement(text="hello", x=20, y=20, width=160, height=50, enabled=True),
    BraillePlacement(text="WORLD", x=200, y=20, width=160, height=50, enabled=True),
    BraillePlacement(text="2026", x=20, y=100, width=160, height=50, enabled=True),
    BraillePlacement(text="hidden", x=200, y=100, width=160, height=50, enabled=False),
]

_BRAILLE_ON_IMAGE_RECIPES: list[tuple[str, dict]] = [
    (
        "overlay",
        {
            "image": _make_image_base(400, 200),
            "placements": _BRAILLE_PLACEMENTS,
            "dot_color": (20, 20, 20),
            "draw_bbox": False,
        },
    ),
    (
        "overlay_with_boxes",
        {
            "image": _make_image_base(400, 200),
            "placements": _BRAILLE_PLACEMENTS,
            "dot_color": (20, 20, 20),
            "draw_bbox": True,
        },
    ),
]

_RENDER_STANDARD_BRAILLE_RECIPES: list[tuple[str, dict]] = [
    (
        "render_standard_short",
        {
            "text": "hello",
            "width": 320,
            "height": 80,
            "plate_size": 0.12,
            "dot_color": (30, 30, 30),
            "bg_color": (245, 245, 245),
        },
    ),
    (
        "render_standard_wrapped",
        {
            "text": "the quick brown fox jumps over the lazy dog",
            "width": 320,
            "height": 240,
            "plate_size": 0.05,
            "dot_color": (0, 0, 0),
            "bg_color": (255, 255, 255),
        },
    ),
]

_STANDARD_BRAILLE_ON_IMAGE_RECIPES: list[tuple[str, dict]] = [
    (
        "standard_on_image_single_line",
        {
            "image": Image.new("RGB", (256, 256), (255, 255, 255)),
            "text": "hello",
            "plate_size": 0.12,
        },
    ),
    (
        "standard_on_image_multi_line",
        {
            "image": Image.new("RGB", (256, 256), (255, 255, 255)),
            "text": "the quick brown fox jumps over the lazy dog",
            "plate_size": 0.12,
        },
    ),
    (
        "standard_on_image_capitals_digits",
        {
            "image": Image.new("RGB", (256, 256), (255, 255, 255)),
            "text": "Room 314 Open",
            "plate_size": 0.12,
        },
    ),
]

_PENDING_BOX_RECIPES: list[tuple[str, dict]] = [
    # Both corners clicked
    (
        "pending_box",
        {
            "image": Image.new("RGB", (240, 120), (250, 250, 250)),
            "start_point": (30, 30),
            "end_point": (210, 90),
            "text": "braille",
            "box_color": (100, 150, 255),
        },
    ),
    # Only the first corner clicked
    (
        "pending_box_no_end",
        {
            "image": Image.new("RGB", (240, 120), (250, 250, 250)),
            "start_point": (60, 40),
            "end_point": None,
            "text": "",
            "box_color": (100, 150, 255),
        },
    ),
]

_DISP_RECIPES: list[tuple[str, dict]] = [
    (
        "disp_basic",
        {
            "placements": [_full_canvas_placement("tactile", w=256, h=128)],
            "width": 256,
            "height": 128,
        },
    ),
    (
        "disp_uppercase_digits",
        {
            "placements": [_full_canvas_placement("Cell 2B", w=256, h=128)],
            "width": 256,
            "height": 128,
        },
    ),
]

_STANDARD_DISP_RECIPES: list[tuple[str, dict]] = [
    (
        "standard_single_line",
        {"text": "hello", "plate_size": 0.12, "resolution": 256},
    ),
    (
        "standard_multi_line",
        {
            "text": "the quick brown fox jumps over the lazy dog",
            "plate_size": 0.12,
            "resolution": 256,
        },
    ),
    (
        "standard_capitals_digits",
        {"text": "Room 314 Open", "plate_size": 0.12, "resolution": 256},
    ),
]


# =============================================================================
# Test BrailleGeometry
# =============================================================================


class TestBrailleGeometry:
    def test_empty_text_has_zero_dimensions(self):
        g = BrailleGeometry("")
        assert g.standard_width() == 0.0
        assert g.standard_height() == 0.0

    def test_blank_cell_contributes_no_dots_but_takes_width(self):
        g = BrailleGeometry(" ")
        assert g.position_braille_in_box((0, 0), (10, 10))[0] == []
        # A single blank cell still occupies a cell's worth of width.
        assert g.standard_width() == pytest.approx(
            STANDARD_INTRA_CELL_SPACING + STANDARD_DOT_DIAMETER
        )

    def test_single_character_width_matches_one_cell(self):
        g = BrailleGeometry("a")  # braille "a" is dot 1 only
        expected_w = STANDARD_INTRA_CELL_SPACING + STANDARD_DOT_DIAMETER
        assert g.standard_width() == pytest.approx(expected_w)

    def test_multi_cell_width_spans_cell_spacing(self):
        g = BrailleGeometry("ab")
        expected = (
            1 * STANDARD_CELL_SPACING
            + STANDARD_INTRA_CELL_SPACING
            + STANDARD_DOT_DIAMETER
        )
        assert g.standard_width() == pytest.approx(expected)

    def test_multi_line_height_uses_line_spacing(self):
        g = BrailleGeometry("a\nb")
        expected = (
            1 * STANDARD_LINE_SPACING
            + 2 * STANDARD_INTRA_CELL_SPACING
            + STANDARD_DOT_DIAMETER
        )
        assert g.standard_height() == pytest.approx(expected)

    def test_position_in_box_uses_uniform_scale_and_centres(self):
        g = BrailleGeometry("a")
        sw, sh = g.standard_width(), g.standard_height()
        # Box exactly 3× standard size, centred at origin.
        box_w, box_h = sw * 3, sh * 3
        pts, r = g.position_braille_in_box(
            (-box_w / 2, -box_h / 2), (box_w / 2, box_h / 2)
        )
        assert r == pytest.approx(STANDARD_DOT_RADIUS * 3)
        assert len(pts) == 1
        # 'a' dot sits at the top-left of the scaled bounding box.
        x, y = pts[0]
        assert x == pytest.approx(-(STANDARD_INTRA_CELL_SPACING / 2) * 3)
        assert y == pytest.approx(-STANDARD_INTRA_CELL_SPACING * 3)

    def test_position_in_box_picks_tighter_axis(self):
        g = BrailleGeometry("ab")  # wide, short
        sw, sh = g.standard_width(), g.standard_height()
        # Box much taller than wide → width is the limiting axis.
        pts, r = g.position_braille_in_box((0.0, 0.0), (sw * 2, sh * 100))
        assert r == pytest.approx(STANDARD_DOT_RADIUS * 2)

    def test_corners_can_be_in_any_order(self):
        g = BrailleGeometry("a")
        a, _ = g.position_braille_in_box((0.0, 0.0), (1.0, 1.0))
        b, _ = g.position_braille_in_box((1.0, 1.0), (0.0, 0.0))
        assert len(a) == len(b) == 1
        assert math.isclose(a[0][0], b[0][0])
        assert math.isclose(a[0][1], b[0][1])

    def test_empty_geometry_in_box_returns_no_points(self):
        g = BrailleGeometry("")
        pts, r = g.position_braille_in_box((0.0, 0.0), (1.0, 1.0))
        assert pts == []
        assert r == STANDARD_DOT_RADIUS


# =============================================================================
# Test image rendering
# =============================================================================


class TestRenderBrailleOnImage:
    def test_returns_image_same_size(self):
        base = Image.new("RGB", (200, 100), (255, 255, 255))
        placements = [BraillePlacement(text="hi", x=10, y=10, width=100, height=40)]
        out = render_braille_on_image(base, placements)
        assert out.size == base.size

    def test_disabled_placement_not_rendered(self):
        base = Image.new("RGB", (200, 100), (255, 255, 255))
        disabled = [
            BraillePlacement(text="hi", x=10, y=10, width=100, height=40, enabled=False)
        ]
        out = render_braille_on_image(base, disabled)
        # Disabled placement -> image stays all white
        assert (np.array(out) == 255).all()

    def test_does_not_mutate_input(self):
        base = Image.new("RGB", (200, 100), (255, 255, 255))
        placements = [BraillePlacement(text="hi", x=10, y=10, width=100, height=40)]
        before = np.array(base).copy()
        render_braille_on_image(base, placements)
        after = np.array(base)
        assert (before == after).all()

    @pytest.mark.parametrize(("name", "args"), _BRAILLE_ON_IMAGE_RECIPES)
    def test_render_braille_on_image(
        self,
        name: str,
        args: dict,
        image_regression: ImageRegressionFixture,
    ):
        actual = render_braille_on_image(**args)
        image_regression.check(actual, basename=name)


class TestRenderBraille:
    def test_empty_text_returns_blank_bg(self):
        img = render_braille("", width=64, height=32, bg_color=(255, 255, 255))
        arr = np.array(img)
        # All pixels should be background color
        assert (arr == 255).all()

    @pytest.mark.parametrize(("name", "args"), _RENDER_BRAILLE_RECIPES)
    def test_render_braille(
        self,
        name: str,
        args: dict,
        image_regression: ImageRegressionFixture,
    ):
        actual = render_braille(**args)
        image_regression.check(actual, basename=name)


class TestRenderStandardBrailleOnImage:
    @pytest.mark.parametrize(("name", "args"), _STANDARD_BRAILLE_ON_IMAGE_RECIPES)
    def test_render_standard_braille_on_image(
        self,
        name: str,
        args: dict,
        image_regression: ImageRegressionFixture,
    ):
        actual = render_standard_braille_on_image(**args)
        image_regression.check(actual, basename=name)


class TestRenderStandardBraille:
    def test_empty_text_returns_blank_bg(self):
        img = render_standard_braille("", width=64, height=32, bg_color=(255, 255, 255))
        assert (np.array(img) == 255).all()

    @pytest.mark.parametrize(("name", "args"), _RENDER_STANDARD_BRAILLE_RECIPES)
    def test_render_standard_braille(
        self,
        name: str,
        args: dict,
        image_regression: ImageRegressionFixture,
    ):
        actual = render_standard_braille(**args)
        image_regression.check(actual, basename=name)


class TestDrawPendingBox:
    @pytest.mark.parametrize(("name", "args"), _PENDING_BOX_RECIPES)
    def test_draw_pending_box(
        self,
        name: str,
        args: dict,
        image_regression: ImageRegressionFixture,
    ):
        actual = draw_pending_box(**args)
        image_regression.check(actual, basename=name)


# =============================================================================
# Test displacement map
# =============================================================================


class TestCreateBrailleDisplacementMap:
    def test_returns_correct_shape(self):
        dm = create_braille_displacement_map(
            [_full_canvas_placement("abc", 128, 64)], width=128, height=64
        )
        assert dm.shape == (64, 128)
        assert np.issubdtype(dm.dtype, np.floating)

    def test_values_in_range(self):
        dm = create_braille_displacement_map(
            [_full_canvas_placement("hello world", 256, 128)],
            width=256,
            height=128,
        )
        assert dm.min() >= 0
        assert dm.max() <= 1.0 + 1e-6

    def test_empty_text_returns_zeros(self):
        dm = create_braille_displacement_map(
            [_full_canvas_placement("", 64, 64)], width=64, height=64
        )
        assert (dm == 0).all()

    def test_empty_placements_returns_zeros(self):
        dm = create_braille_displacement_map([], width=64, height=64)
        assert (dm == 0).all()

    def test_disabled_placement_is_skipped(self):
        placement = BraillePlacement(
            text="a", x=0, y=0, width=128, height=64, enabled=False
        )
        dm = create_braille_displacement_map([placement], width=128, height=64)
        assert (dm == 0).all()

    def test_dot_height_scales_max(self):
        dm_full = create_braille_displacement_map(
            [_full_canvas_placement("a", 128, 64)],
            width=128,
            height=64,
            dot_height=1.0,
        )
        dm_half = create_braille_displacement_map(
            [_full_canvas_placement("a", 128, 64)],
            width=128,
            height=64,
            dot_height=0.5,
        )
        assert dm_full.max() > dm_half.max()
        assert dm_full.max() == pytest.approx(2 * dm_half.max(), rel=0.05)

    def test_multiple_placements_composite(self):
        # Two non-overlapping placements should both contribute dark regions.
        dm = create_braille_displacement_map(
            [
                BraillePlacement(text="a", x=0, y=0, width=64, height=64),
                BraillePlacement(text="b", x=64, y=0, width=64, height=64),
            ],
            width=128,
            height=64,
        )
        # Sanity: dots in both halves.
        assert dm[:, :64].max() > 0
        assert dm[:, 64:].max() > 0

    @pytest.mark.parametrize(("name", "args"), _DISP_RECIPES)
    def test_create_braille_displacement_map(
        self,
        name: str,
        args: dict,
        ndarrays_regression: NDArraysRegressionFixture,
    ):
        dm = create_braille_displacement_map(**args)
        ndarrays_regression.check({"displacement": dm}, basename=name)


class TestCreateStandardBrailleDisplacementMap:
    @pytest.mark.parametrize(("name", "args"), _STANDARD_DISP_RECIPES)
    def test_create_standard_braille_displacement_map(
        self,
        name: str,
        args: dict,
        ndarrays_regression: NDArraysRegressionFixture,
    ):
        dm = create_standard_braille_displacement_map(**args)
        ndarrays_regression.check({"displacement": dm}, basename=name)
