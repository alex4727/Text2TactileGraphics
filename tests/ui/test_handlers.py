from unittest.mock import Mock, create_autospec

import gradio as gr
import numpy as np
import pytest
from PIL import Image

from text2tactilegraphics import TexturedSegment
from text2tactilegraphics.generation.base_image_generation import BaseImageGenerator
from text2tactilegraphics.generation.segmentation import SegmentationEngine
from text2tactilegraphics.generation.texture_generation import (
    GeometryEstimator,
    TextureGenerator,
)
from text2tactilegraphics.generation.tileable_patch_generation import (
    IntraTilePatchGenerator,
)
from text2tactilegraphics.geometry.braille import BraillePlacement
from text2tactilegraphics.ui import handlers
from text2tactilegraphics.ui.handlers import (
    add_click,
    generate_base_image,
    generate_base_mesh,
    generate_displacement,
    generate_final_mesh,
    generate_mesh_with_textures,
    generate_texture_geometry,
    generate_texture_image,
    generate_tiled_preview,
    get_seg_overlay,
    handle_braille_overlay_click,
    make_tileable,
    proceed_to_stage2,
    proceed_to_step2,
    proceed_to_step3,
    proceed_to_step4,
    render_custom_braille_overlay,
    save_braille,
    save_segment,
    segment_with_text,
)
from text2tactilegraphics.ui.state import AppState


def _make_image(h: int = 64, w: int = 64) -> Image.Image:
    return Image.new("RGB", (w, h), (10, 20, 30))


def _make_mask(val: float = 0.5, h: int = 64, w: int = 64) -> np.ndarray:
    return np.eye(h, w) * val  # type: ignore


def _make_evt_mock(x: int, y: int) -> Mock:
    evt = Mock(spec=gr.SelectData)
    evt.index = (x, y)
    return evt


def _make_base_gen_mock(img: Image.Image) -> Mock:
    mock = create_autospec(BaseImageGenerator)
    mock.generate.return_value = img
    return mock


def _make_seg_engine_mock(
    segments: list[tuple[np.ndarray, float]] | None = None,
) -> Mock:
    mock = create_autospec(SegmentationEngine)
    if segments is None:
        segments = [(_make_mask(), 0.9)]
    mock.segment_with_text.return_value = segments
    return mock


def _make_texture_gen_mock(img: Image.Image) -> Mock:
    mock = create_autospec(TextureGenerator)
    mock.generate.return_value = img
    return mock


def _make_geom_estimator_mock() -> Mock:
    mock = create_autospec(GeometryEstimator)
    # Single-channel depth in [0, 1].
    mock.compute_depth.return_value = np.full((4, 4), 0.5, dtype=np.float32)
    # Tangent-space normal map in [-1, 1] (straight-up normal).
    _normal = np.zeros((4, 4, 3), dtype=np.float32)
    _normal[..., 2] = 1.0
    mock.compute_normal.return_value = _normal
    return mock


def _make_tiling_gen_mock(img: Image.Image) -> Mock:
    mock = create_autospec(IntraTilePatchGenerator)
    mock.make_tileable.return_value = img
    return mock


@pytest.fixture
def mock_create_tactile(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock = Mock(return_value="/tmp/mesh.glb")
    monkeypatch.setattr(handlers, "create_tactile_graphic", mock)
    return mock


# =============================================================================
# Test moving between stages
# =============================================================================


class TestProceedToStage2:
    def test_should_select_stage2_tab_when_base_image_present(self):
        base_img = _make_image()
        result = proceed_to_stage2(base_img)
        assert isinstance(result, gr.Tabs)
        assert result.selected == "stage2"

    def test_should_raise_gradio_error_when_base_image_absent(self):
        with pytest.raises(gr.Error, match="base image"):
            proceed_to_stage2(None)


class TestProceedToStep2:
    def test_should_select_step2_tab_when_mask_image_present(self):
        result = proceed_to_step2(_make_image())
        assert isinstance(result, gr.Tabs)
        assert result.selected == "step2"

    def test_should_raise_gradio_error_when_mask_image_absent(self):
        with pytest.raises(gr.Error, match="region"):
            proceed_to_step2(None)


class TestProceedToStep3:
    def test_should_select_step3_tab_when_geometry_image_present(self):
        result = proceed_to_step3(_make_image())
        assert isinstance(result, gr.Tabs)
        assert result.selected == "step3"

    def test_should_raise_gradio_error_when_geometry_image_absent(self):
        with pytest.raises(gr.Error, match="texture"):
            proceed_to_step3(None)


class TestProceedToStep4:
    def test_should_select_step4_tab_when_tileable_patch_image_present(self):
        result = proceed_to_step4(_make_image())
        assert isinstance(result, gr.Tabs)
        assert result.selected == "step4"

    def test_should_raise_gradio_error_when_tileable_patch_image_absent(self):
        with pytest.raises(gr.Error, match="tileable"):
            proceed_to_step4(None)


# =============================================================================
# Test Stage 1
# =============================================================================


class TestGenerateBaseImage:
    @pytest.fixture
    def generated_img(self) -> Image.Image:
        return Image.new("RGB", (16, 16), (50, 100, 200))

    @pytest.fixture
    def mock_base_gen(self, generated_img: Image.Image) -> Mock:
        return _make_base_gen_mock(generated_img)

    @pytest.fixture
    def app_state(self, mock_base_gen: Mock) -> AppState:
        app_state = AppState()
        app_state.get_base_gen = lambda: mock_base_gen
        return app_state

    def test_should_return_image_from_generator_on_success(
        self, app_state: AppState, generated_img: Image.Image
    ):
        result = generate_base_image("a dolphin", "qwen_edit", 4, 42, app_state)
        assert result == generated_img

    def test_should_raise_gradio_error_when_prompt_is_blank(
        self, app_state: AppState, mock_base_gen: Mock
    ):
        with pytest.raises(gr.Error, match="prompt"):
            generate_base_image("   ", "qwen_edit", 4, 42, app_state)
        # Generator must not be invoked when the prompt fails validation.
        mock_base_gen.generate.assert_not_called()

    def test_should_wrap_generator_failure_in_gradio_error(
        self, app_state: AppState, mock_base_gen: Mock
    ):
        mock_base_gen.generate.side_effect = RuntimeError("boom")
        with pytest.raises(gr.Error, match="failed"):
            generate_base_image("a dolphin", "qwen_edit", 4, 42, app_state)


class TestGenerateBaseMesh:
    def test_should_return_glb_path_on_success(self, mock_create_tactile: Mock):
        result = generate_base_mesh(_make_image())
        assert result == "/tmp/mesh.glb"

    def test_should_raise_gradio_error_when_base_image_is_none(
        self, mock_create_tactile: Mock
    ):
        with pytest.raises(gr.Error, match="base image"):
            generate_base_mesh(None)
        mock_create_tactile.assert_not_called()

    def test_should_wrap_failure_in_gradio_error(self, mock_create_tactile: Mock):
        mock_create_tactile.side_effect = RuntimeError("boom")
        with pytest.raises(gr.Error, match="Mesh generation failed"):
            generate_base_mesh(_make_image())


# =============================================================================
# Test Stage 2
# =============================================================================


class TestGetSegOverlay:
    def test_should_return_image(self):
        base = _make_image()
        result = get_seg_overlay(base, _make_mask(), [(10, 20), (30, 40)], [1, 0])
        assert isinstance(result, Image.Image)
        assert result.size == base.size

    def test_should_return_base_image_unchanged_when_mask_is_none(self):
        base = _make_image()
        result = get_seg_overlay(base, None, [], [])
        assert result is base


class TestHandleImageClick:
    def test_should_append_add_point_when_not_in_subtract_mode(self):
        points, labels = add_click([], [], False, _make_evt_mock(10, 20))
        assert points == [(10, 20)]
        assert labels == [1]

    def test_should_append_subtract_point_when_in_subtract_mode(self):
        points, labels = add_click([], [], True, _make_evt_mock(10, 20))
        assert points == [(10, 20)]
        assert labels == [0]

    def test_should_preserve_existing_points_and_labels(self):
        existing_points = [(1, 1), (2, 2)]
        existing_labels = [1, 0]
        points, labels = add_click(
            existing_points, existing_labels, False, _make_evt_mock(3, 3)
        )
        assert points == [(1, 1), (2, 2), (3, 3)]
        assert labels == [1, 0, 1]


class TestSegmentWithText:
    @pytest.fixture
    def mock_seg_engine(self) -> Mock:
        return _make_seg_engine_mock()

    @pytest.fixture
    def app_state(self, mock_seg_engine: Mock) -> AppState:
        app_state = AppState()
        app_state.get_seg_engine = lambda: mock_seg_engine
        return app_state

    def test_should_return_mask_and_preview_images_on_success(
        self, app_state: AppState
    ):
        mask, overlay_img = segment_with_text(_make_image(), "dolphin", app_state)

        assert isinstance(mask, np.ndarray)
        assert isinstance(overlay_img, Image.Image)

    def test_should_return_union_of_all_candidate_masks(self, app_state: AppState):
        # Three disjoint masks — the union should light up all three regions.
        a = np.zeros((4, 4), dtype=bool)
        a[0, 0] = True
        b = np.zeros((4, 4), dtype=bool)
        b[1, 1] = True
        c = np.zeros((4, 4), dtype=bool)
        c[2, 2] = True
        app_state.get_seg_engine = lambda: _make_seg_engine_mock(
            [(a, 0.2), (b, 0.95), (c, 0.5)]
        )

        mask, _ = segment_with_text(_make_image(), "x", app_state)

        expected = np.zeros((4, 4), dtype=bool)
        expected[0, 0] = expected[1, 1] = expected[2, 2] = True
        np.testing.assert_array_equal(mask, expected)

    def test_should_raise_gradio_error_when_base_image_is_none(
        self, app_state: AppState, mock_seg_engine: Mock
    ):
        with pytest.raises(gr.Error, match="base image"):
            segment_with_text(None, "dolphin", app_state)
        # Engine must not run when input validation fails.
        mock_seg_engine.segment_with_text.assert_not_called()

    def test_should_raise_gradio_error_when_prompt_is_blank(
        self, app_state: AppState, mock_seg_engine: Mock
    ):
        with pytest.raises(gr.Error, match="prompt"):
            segment_with_text(_make_image(), "   ", app_state)
        mock_seg_engine.segment_with_text.assert_not_called()

    def test_should_raise_gradio_error_when_engine_returns_no_segments(
        self, app_state: AppState
    ):
        app_state.get_seg_engine = lambda: _make_seg_engine_mock(segments=[])

        with pytest.raises(gr.Error, match="No segments found"):
            segment_with_text(_make_image(), "dolphin", app_state)

    def test_should_wrap_engine_failure_in_gradio_error(
        self, app_state: AppState, mock_seg_engine: Mock
    ):
        mock_seg_engine.segment_with_text.side_effect = RuntimeError("boom")

        with pytest.raises(gr.Error, match="failed"):
            segment_with_text(_make_image(), "dolphin", app_state)


class TestGenerateTextureImage:
    @pytest.fixture
    def generated_img(self) -> Image.Image:
        return Image.new("RGB", (16, 16), (200, 100, 50))

    @pytest.fixture
    def mock_texture_gen(self, generated_img: Image.Image) -> Mock:
        return _make_texture_gen_mock(generated_img)

    @pytest.fixture
    def app_state(self, mock_texture_gen: Mock) -> AppState:
        app_state = AppState()
        app_state.get_texture_gen = lambda: mock_texture_gen
        return app_state

    def test_should_return_image_from_generator_on_success(
        self, app_state: AppState, generated_img: Image.Image
    ):
        result = generate_texture_image("an avocado skin", 4, 42, app_state)
        assert result == generated_img

    def test_should_raise_gradio_error_when_prompt_is_blank(
        self, app_state: AppState, mock_texture_gen: Mock
    ):
        with pytest.raises(gr.Error, match="prompt"):
            generate_texture_image("   ", 4, 42, app_state)
        # Generator must not be invoked when the prompt fails validation.
        mock_texture_gen.generate.assert_not_called()

    def test_should_wrap_generator_failure_in_gradio_error(
        self, app_state: AppState, mock_texture_gen: Mock
    ):
        mock_texture_gen.generate.side_effect = RuntimeError("boom")
        with pytest.raises(gr.Error, match="Texture generation failed"):
            generate_texture_image("an avocado skin", 4, 42, app_state)


class TestGenerateTextureGeometry:
    @pytest.fixture
    def mock_geometry(self) -> Mock:
        return _make_geom_estimator_mock()

    @pytest.fixture
    def app_state(self, mock_geometry: Mock) -> AppState:
        app_state = AppState()
        app_state.get_geom_estimator = lambda: mock_geometry
        return app_state

    def test_should_return_geometry_array_and_preview_image_for_normal_mode(
        self, app_state: AppState
    ):
        app_state.config.geometry_type = "normal"

        geometry_arr, geometry_img = generate_texture_geometry(
            _make_image(), True, app_state
        )

        assert isinstance(geometry_arr, np.ndarray)
        assert isinstance(geometry_img, Image.Image)

    def test_should_return_geometry_array_and_preview_image_for_depth_mode(
        self, app_state: AppState
    ):
        app_state.config.geometry_type = "depth"

        geometry_arr, geometry_img = generate_texture_geometry(
            _make_image(), True, app_state
        )

        assert isinstance(geometry_arr, np.ndarray)
        assert isinstance(geometry_img, Image.Image)

    def test_should_wrap_generator_failure_in_gradio_error(
        self, app_state: AppState, mock_geometry: Mock
    ):
        app_state.config.geometry_type = "normal"
        mock_geometry.compute_normal.side_effect = RuntimeError("boom")

        with pytest.raises(gr.Error, match="failed"):
            generate_texture_geometry(_make_image(), True, app_state)


class TestGenerateTiling:
    def test_should_return_tiled_preview(self):
        tile = _make_image(32, 32)
        tiled_preview = generate_tiled_preview(tile)

        assert isinstance(tiled_preview, Image.Image)
        # 3×3 of the input tile.
        assert tiled_preview.size == (96, 96)

    def test_should_wrap_failure_in_gradio_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            handlers, "tile_image", Mock(side_effect=RuntimeError("boom"))
        )

        with pytest.raises(gr.Error, match="failed"):
            generate_tiled_preview(_make_image(32, 32))


class TestGenerateDisplacement:
    def test_should_return_displacement_image(self):
        tiled_preview = _make_image(32 * 3, 32 * 3)
        displacement_img = generate_displacement(tiled_preview, "opengl")

        assert isinstance(displacement_img, Image.Image)
        assert displacement_img.size == (96, 96)

    def test_should_wrap_failure_in_gradio_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            handlers,
            "tileable_patch_to_displacement",
            Mock(side_effect=RuntimeError("boom")),
        )

        with pytest.raises(gr.Error, match="failed"):
            generate_displacement(_make_image(32 * 3, 32 * 3), "opengl")


class TestMakeTileable:
    @pytest.fixture
    def tileable_patch(self) -> Image.Image:
        return Image.new("RGB", (8, 8), (0, 255, 0))

    @pytest.fixture
    def mock_tiling_gen(self, tileable_patch: Image.Image) -> Mock:
        return _make_tiling_gen_mock(tileable_patch)

    @pytest.fixture
    def app_state(self, mock_tiling_gen: Mock) -> AppState:
        app_state = AppState()
        app_state.get_tiling_gen = lambda method: mock_tiling_gen
        app_state.config.geometry_type = "normal"
        return app_state

    @pytest.fixture
    def make_tileable_args(self, app_state: AppState) -> dict:
        return dict(
            geometry_img=_make_image(),
            method="inpainting",
            steps=10,
            seed=42,
            use_highpass=True,
            highpass_freq_threshold=50,
            highpass_method="per_channel",
            app_state=app_state,
        )

    def test_should_return_tileable_patch_on_success(
        self, make_tileable_args: dict, tileable_patch: Image.Image
    ):
        result = make_tileable(**make_tileable_args)
        assert result == tileable_patch

    def test_should_raise_gradio_error_when_geometry_is_none(
        self, make_tileable_args: dict, mock_tiling_gen: Mock
    ):
        make_tileable_args["geometry_img"] = None
        with pytest.raises(gr.Error, match="texture"):
            make_tileable(**make_tileable_args)
        # Generator must not be invoked when input validation fails.
        mock_tiling_gen.make_tileable.assert_not_called()

    def test_should_wrap_generator_failure_in_gradio_error(
        self, make_tileable_args: dict, mock_tiling_gen: Mock
    ):
        mock_tiling_gen.make_tileable.side_effect = RuntimeError("boom")

        with pytest.raises(gr.Error, match="Tiling failed"):
            make_tileable(**make_tileable_args)

    def test_should_raise_assertion_error_for_high_pass_in_depth_mode(
        self, app_state: AppState, make_tileable_args: dict, mock_tiling_gen: Mock
    ):
        app_state.config.geometry_type = "depth"

        with pytest.raises(AssertionError, match="normal mode"):
            make_tileable(**make_tileable_args)

        mock_tiling_gen.make_tileable.assert_not_called()


class TestGenerateMeshWithTextures:
    @pytest.fixture
    def app_state(self) -> AppState:
        return AppState()

    @pytest.fixture
    def mesh_args(self) -> dict:
        return dict(
            base_img=_make_image(),
            mask=_make_mask(),
            tileable_patch=_make_image(),
            normal_format="opengl",
            displacement_scale=0.005,
            displacement_direction="normal",
            tile_repeat=3,
            segments=[],
        )

    def test_should_return_glb_path_on_success(
        self, mock_create_tactile: Mock, mesh_args: dict
    ):
        result = generate_mesh_with_textures(**mesh_args)
        assert result == "/tmp/mesh.glb"

    def test_should_raise_gradio_error_when_base_image_is_none(
        self, mock_create_tactile: Mock, mesh_args: dict
    ):
        mesh_args["base_img"] = None
        with pytest.raises(gr.Error, match="base image"):
            generate_mesh_with_textures(**mesh_args)
        mock_create_tactile.assert_not_called()

    def test_should_raise_gradio_error_when_mask_is_none(
        self, mock_create_tactile: Mock, mesh_args: dict
    ):
        mesh_args["mask"] = None
        with pytest.raises(gr.Error, match="region"):
            generate_mesh_with_textures(**mesh_args)
        mock_create_tactile.assert_not_called()

    def test_should_raise_gradio_error_when_tileable_patch_is_none(
        self, mock_create_tactile: Mock, mesh_args: dict
    ):
        mesh_args["tileable_patch"] = None
        with pytest.raises(gr.Error, match="tileable"):
            generate_mesh_with_textures(**mesh_args)
        mock_create_tactile.assert_not_called()

    def test_should_wrap_failure_in_gradio_error(
        self, mock_create_tactile: Mock, mesh_args: dict
    ):
        mock_create_tactile.side_effect = RuntimeError("boom")
        with pytest.raises(gr.Error, match="Mesh generation failed"):
            generate_mesh_with_textures(**mesh_args)


class TestSaveSegment:
    @pytest.fixture
    def save_args(self) -> dict:
        return dict(
            base_img=_make_image(),
            mask=_make_mask(),
            tileable_patch=_make_image(),
            displacement_scale=0.007,
            displacement_direction="normal",
            tile_repeat=4,
            segments=[],
        )

    def test_should_append_new_segment_to_empty_list(self, save_args: dict):
        result = save_segment(**save_args)
        assert len(result) == 1
        assert isinstance(result[0], TexturedSegment)

    def test_should_preserve_existing_segments(self, save_args: dict):
        existing = TexturedSegment(mask=_make_mask(), tileable_patch=Image.Image())
        save_args["segments"] = [existing]

        result = save_segment(**save_args)

        assert len(result) == 2
        assert result[0] is existing

    def test_should_persist_displacement_settings_on_new_segment(self, save_args: dict):
        save_args["displacement_scale"] = 0.012
        save_args["displacement_direction"] = "z"
        save_args["tile_repeat"] = 7

        result = save_segment(**save_args)

        new_seg = result[-1]
        assert new_seg.displacement_scale == 0.012
        assert new_seg.displacement_direction == "z"
        assert new_seg.tile_repeat == 7

    def test_should_copy_mask_and_tileable_patch_onto_new_segment(
        self, save_args: dict
    ):
        # _current_working_segment defensively `.copy()`s the mask/patch so mutations can't leak into saved segments.
        mask = save_args["mask"]
        patch = save_args["tileable_patch"]

        new_seg = save_segment(**save_args)[-1]

        np.testing.assert_array_equal(new_seg.mask, mask)
        assert new_seg.mask is not mask
        assert new_seg.tileable_patch is not patch

    def test_should_raise_gradio_error_when_base_image_is_none(self, save_args: dict):
        save_args["base_img"] = None
        with pytest.raises(gr.Error, match="base image"):
            save_segment(**save_args)

    def test_should_raise_gradio_error_when_mask_is_none(self, save_args: dict):
        save_args["mask"] = None
        with pytest.raises(gr.Error, match="region"):
            save_segment(**save_args)

    def test_should_raise_gradio_error_when_tileable_patch_is_none(
        self, save_args: dict
    ):
        save_args["tileable_patch"] = None
        with pytest.raises(gr.Error, match="tileable"):
            save_segment(**save_args)

    def test_should_not_mutate_segments_on_validation_failure(self, save_args: dict):
        existing = TexturedSegment(mask=_make_mask(), tileable_patch=Image.Image())
        save_args["segments"] = [existing]
        save_args["mask"] = None  # Trigger validation error

        with pytest.raises(gr.Error):
            save_segment(**save_args)

        assert save_args["segments"] == [existing]


# =============================================================================
# Test Stage 3
# =============================================================================


def _make_braille_placement(text: str = "hi", enabled: bool = True) -> BraillePlacement:
    return BraillePlacement(text=text, x=10, y=20, width=80, height=40, enabled=enabled)


class TestHandleBrailleOverlayClick:
    def test_should_record_start_corner_on_first_click(self):
        result = handle_braille_overlay_click(
            _make_image(), (None, None), _make_evt_mock(15, 25)
        )
        assert result == ((15, 25), None)

    def test_should_close_box_on_second_click(self):
        # First click was at (15, 25); second click at (200, 100) closes it.
        result = handle_braille_overlay_click(
            _make_image(), ((15, 25), None), _make_evt_mock(200, 100)
        )
        # Corners normalized: top-left then bottom-right.
        assert result == ((15, 25), (200, 100))

    def test_should_normalize_corners_when_drawn_bottom_right_to_top_left(self):
        # User dragged from (200, 100) to (15, 25) — the second click is
        # above-left of the start. The function should still return
        # (top-left, bottom-right).
        result = handle_braille_overlay_click(
            _make_image(), ((200, 100), None), _make_evt_mock(15, 25)
        )
        assert result == ((15, 25), (200, 100))

    def test_should_raise_gradio_error_when_base_image_is_none(self):
        with pytest.raises(gr.Error, match="base image"):
            handle_braille_overlay_click(None, (None, None), _make_evt_mock(15, 25))

    def test_should_raise_gradio_error_when_second_click_makes_box_too_small(self):
        with pytest.raises(gr.Error, match="Box too small"):
            handle_braille_overlay_click(
                _make_image(), ((15, 25), None), _make_evt_mock(20, 30)
            )


class TestRenderCustomBrailleOverlay:
    def test_should_return_none_when_base_image_is_none(self):
        result = render_custom_braille_overlay(None, [], (None, None))
        assert result is None

    def test_should_return_image_when_only_base_image_is_given(self):
        result = render_custom_braille_overlay(_make_image(), [], (None, None))
        assert isinstance(result, Image.Image)


class TestSaveBraille:
    def test_should_append_new_placement(self):
        existing = _make_braille_placement("first")
        placements = [existing]

        result = save_braille("second", ((10, 20), (90, 60)), placements)

        assert len(result) == 2
        assert result[0] is existing
        new = result[1]
        assert isinstance(new, BraillePlacement)
        assert new.text == "second"
        assert new.x == 10
        assert new.y == 20
        assert new.width == 80
        assert new.height == 40
        assert new.enabled is True

    def test_should_strip_whitespace_from_text(self):
        result = save_braille("  hello  ", ((0, 0), (50, 30)), [])
        assert result[-1].text == "hello"

    def test_should_raise_gradio_error_when_text_is_blank(self):
        placements: list[BraillePlacement] = []
        with pytest.raises(gr.Error, match="text"):
            save_braille("   ", ((0, 0), (50, 30)), placements)
        # No partial commit on validation failure.
        assert placements == []

    def test_should_raise_gradio_error_when_box_is_incomplete(self):
        placements: list[BraillePlacement] = []
        with pytest.raises(gr.Error, match="box"):
            save_braille("hello", ((1, 2), None), placements)
        assert placements == []


class TestGenerateFinalMesh:
    @pytest.fixture
    def app_state(self) -> AppState:
        return AppState()

    @pytest.fixture
    def mesh_args(self) -> dict:
        return dict(
            base_img=_make_image(),
            segments=[],
            normal_format="opengl",
            braille_mode="standard",
            standard_braille_text="dolphin",
            plate_size=0.12,
            flat_top_ratio=0.3,
            bottom_padding=0.001,
            braille_placements=[],
            dot_height=0.0005,
            flatten_plate=True,
            plate_thickness=0.004,
        )

    def test_should_raise_gradio_error_when_base_image_is_none(
        self, app_state: AppState, mock_create_tactile: Mock, mesh_args: dict
    ):
        mesh_args["base_img"] = None
        with pytest.raises(gr.Error, match="base image"):
            generate_final_mesh(**mesh_args)
        mock_create_tactile.assert_not_called()

    def test_should_wrap_failure_in_gradio_error(
        self, app_state: AppState, mock_create_tactile: Mock, mesh_args: dict
    ):
        mock_create_tactile.side_effect = RuntimeError("boom")
        with pytest.raises(gr.Error, match="failed"):
            generate_final_mesh(**mesh_args)
