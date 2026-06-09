"""Event handlers for the Text2TactileGraphics Gradio UI."""

from typing import Literal

import gradio as gr
import numpy as np
from PIL import Image

from text2tactilegraphics import TexturedSegment
from text2tactilegraphics.generation.base_image_generation import BaseImageModel
from text2tactilegraphics.generation.segmentation import (
    apply_mask_overlay,
    draw_points_on_image,
)
from text2tactilegraphics.generation.utils import (
    depth_to_image,
    displacement_to_image,
    mask_to_image,
    normal_to_image,
    tile_image,
)
from text2tactilegraphics.geometry.braille import (
    BraillePlacement,
    draw_pending_box,
    render_braille_on_image,
)
from text2tactilegraphics.geometry.displacement import (
    DisplacementDirection,
    NormalFormat,
    tileable_patch_to_displacement,
)
from text2tactilegraphics.geometry.filtering import (
    HighpassMethod,
    apply_high_pass_to_normal_map,
)
from text2tactilegraphics.geometry.tactile_graphics import create_tactile_graphic
from text2tactilegraphics.ui.state import AppState, TilingMethod

# ----------------------------------------------------------------- shared helpers

_Point = tuple[int, int]
_Box = tuple[_Point | None, _Point | None]


# =============================================================================
# Moving between stages
# =============================================================================


def proceed_to_stage2(base_img: Image.Image | None) -> gr.Tabs:
    if not base_img:
        raise gr.Error("Create a base image first")
    return gr.Tabs(selected="stage2")


def proceed_to_step2(mask_img: Image.Image | None) -> gr.Tabs:
    """Stage 2: advance from Step 1 (Select region) → Step 2 (Create texture)."""
    if mask_img is None:
        raise gr.Error("Select a region first")
    return gr.Tabs(selected="step2")


def proceed_to_step3(geometry_img: Image.Image | None) -> gr.Tabs:
    """Stage 2: advance from Step 2 (Create texture) → Step 3 (Make tileable)."""
    if geometry_img is None:
        raise gr.Error("Create a texture first")
    return gr.Tabs(selected="step3")


def proceed_to_step4(tileable_patch_img: Image.Image | None) -> gr.Tabs:
    """Stage 2: advance from Step 3 (Make tileable) → Step 4 (Preview & save)."""
    if tileable_patch_img is None:
        raise gr.Error("Create a tileable texture first")
    return gr.Tabs(selected="step4")


# =============================================================================
# Stage 1
# =============================================================================


def generate_base_image(
    prompt: str,
    model: BaseImageModel,
    steps: int,
    seed: int,
    app_state: AppState,
) -> Image.Image:
    if not prompt.strip():
        raise gr.Error("Please enter a prompt")

    try:
        base_gen = app_state.get_base_gen()
        image = base_gen.generate(prompt, model=model, steps=steps, seed=seed)

        app_state.save_image(image, "generate_base_image__base.png")

        gr.Info("Image generation finished")
        return image
    except Exception as e:
        raise gr.Error(f"Image generation failed: {e}") from e


def generate_base_mesh(base_image: Image.Image | None) -> str:
    """Generate a textureless 3D mesh from the base image."""
    if base_image is None:
        raise gr.Error("Create a base image first")

    try:
        glb_path = create_tactile_graphic(base_image)
        gr.Info("Mesh generation finished")
        return glb_path
    except Exception as e:
        raise gr.Error(f"Mesh generation failed: {e}") from e


# =============================================================================
# Stage 2 — interactive segmentation
# =============================================================================


def get_seg_overlay(
    base_image: Image.Image,
    mask: np.ndarray | None,
    points: list[tuple[int, int]],
    labels: list[int],
) -> Image.Image:
    """Returns a visualization of the segmentation mask overlaid on the base image."""
    if mask is None:
        return base_image

    vis = apply_mask_overlay(base_image, mask, color=(255, 100, 100), opacity=0.5)
    vis = draw_points_on_image(vis, points, labels)
    return vis


def segment_with_text(
    base_image: Image.Image | None, text_prompt: str, app_state: AppState
) -> tuple[np.ndarray, Image.Image]:
    """Run SAM3 text segmentation and return the union of all candidate masks."""
    if base_image is None:
        raise gr.Error("Create a base image first (Stage 1)")
    if not text_prompt.strip():
        raise gr.Error("Please enter a prompt")

    try:
        seg_engine = app_state.get_seg_engine()
        segments = seg_engine.segment_with_text(
            base_image, text_prompt, confidence_threshold=0.5
        )

        if not segments:
            raise gr.Error(f"No segments found for '{text_prompt}'")

        # Union of all candidate masks. `np.maximum.reduce` works for both
        # boolean masks and float [0, 1] confidence masks.
        mask = np.maximum.reduce([m for m, _score in segments])

        app_state.save_array(mask, "segment_with_text__mask.npy")

        gr.Info("Auto-segmentation finished")
        return mask, get_seg_overlay(base_image, mask, [], [])
    except Exception as e:
        raise gr.Error(f"Segmentation failed: {e}")


def add_click(
    points: list[tuple[int, int]] | None,
    labels: list[int] | None,
    subtract_mode: bool,
    evt: gr.SelectData,
) -> tuple[list[tuple[int, int]], list[int]]:
    if points is None:
        points = []
    if labels is None:
        labels = []

    x, y = evt.index
    label = 0 if subtract_mode else 1  # 1=add, 0=subtract
    points.append((x, y))
    labels.append(label)

    return points, labels


def segment_with_click(
    base_image: Image.Image,
    points: list[tuple[int, int]],
    labels: list[int],
    app_state: AppState,
) -> tuple[np.ndarray, Image.Image]:
    """Run SAM3 click segmentation.

    Returns mask, image version of the mask, and overlay visualization.
    """
    try:
        seg_engine = app_state.get_seg_engine()
        mask = seg_engine.segment_with_points(base_image, points, labels)

        app_state.save_array(mask, "segment_with_click__mask.npy")

        return mask, get_seg_overlay(base_image, mask, points, labels)
    except Exception as e:
        raise gr.Error(f"Segmentation failed: {e}")


def get_selected_mask_and_image(
    source: Literal["text", "click"],
    text_mask: np.ndarray | None,
    click_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, Image.Image | None]:
    mask = text_mask if source == "text" else click_mask
    mask_img = None if mask is None else mask_to_image(mask)
    return mask, mask_img


# =============================================================================
# Stage 2 — texture + tiling
# =============================================================================


def generate_texture_image(
    prompt: str, steps: int, seed: int, app_state: AppState
) -> Image.Image:
    """Generate texture image + geometry (depth or normal) for the current segment."""
    if not prompt.strip():
        raise gr.Error("Please enter a prompt")

    try:
        tex_gen = app_state.get_texture_gen()
        texture_img = tex_gen.generate(prompt, steps=steps, seed=seed)

        app_state.save_image(texture_img, "generate_texture_image__texture.png")

        gr.Info("Texture generation finished")
        return texture_img
    except Exception as e:
        raise gr.Error(f"Texture generation failed: {e}") from e


def generate_texture_geometry(
    image: Image.Image, crop_geometry: bool, app_state: AppState
) -> tuple[np.ndarray, Image.Image]:
    geometry_type = app_state.config.geometry_type
    try:
        geom = app_state.get_geom_estimator()
        if geometry_type == "depth":
            geometry_arr = geom.compute_depth(image, crop=crop_geometry)
            geometry_img = depth_to_image(geometry_arr).convert("RGB")
        else:
            geometry_arr = geom.compute_normal(image, crop=crop_geometry)
            geometry_img = normal_to_image(geometry_arr)

        app_state.save_image(geometry_img, "generate_texture_geometry__geometry.png")
        app_state.save_array(geometry_arr, "generate_texture_geometry__geometry.npy")

        gr.Info("Generated geometry map")
        return geometry_arr, geometry_img
    except Exception as e:
        raise gr.Error(f"Geometry generation failed: {e}") from e


def generate_tiling_and_displacement(
    tileable_image: Image.Image, normal_format: NormalFormat
) -> tuple[Image.Image, Image.Image]:
    try:
        tiled_preview = tile_image(tileable_image, 3, 3)
        displacement_map = tileable_patch_to_displacement(tiled_preview, normal_format)
        displacement_img = displacement_to_image(displacement_map)

        return tiled_preview, displacement_img
    except Exception as e:
        raise gr.Error(f"Tiling generation failed: {e}") from e


def make_tileable(
    geometry_img: Image.Image | None,
    method: TilingMethod,
    steps: int,
    seed: int,
    use_highpass: bool,
    highpass_freq_threshold: int,
    highpass_method: HighpassMethod,
    app_state: AppState,
) -> Image.Image:
    if geometry_img is None:
        raise gr.Error("Create a texture first (Step 2)")

    geometry_img = _prepare_for_tiling(
        geometry_img, use_highpass, highpass_freq_threshold, highpass_method, app_state
    )

    try:
        tiling_gen = app_state.get_tiling_gen(method)
        tileable_patch = tiling_gen.make_tileable(
            geometry_img,
            num_inference_steps=steps,
            seed=seed,
        )

        app_state.save_image(tileable_patch, "make_tileable__patch.png")

        gr.Info("Tiling finished")
        return tileable_patch
    except Exception as e:
        raise gr.Error(f"Tiling failed: {e}") from e


def _prepare_for_tiling(
    geometry_img: Image.Image,
    use_highpass: bool,
    freq_threshold: int,
    method: HighpassMethod,
    app_state: AppState,
) -> Image.Image:
    if not use_highpass:
        return geometry_img

    geometry_type = app_state.config.geometry_type

    if geometry_type == "depth":
        raise AssertionError(
            "High-pass filtering is only supported for normal mode. "
            "Please disable high-pass filter or switch to normal mode."
        )

    # Normal mode
    filtered = apply_high_pass_to_normal_map(
        geometry_img, freq_threshold=freq_threshold, method=method
    )
    return filtered  # type:ignore


# =============================================================================
# Stage 2 — mesh assembly + saved-segments table
# =============================================================================


def generate_mesh_with_textures(
    base_img: Image.Image | None,
    mask: np.ndarray | None,
    tileable_patch: Image.Image | None,
    normal_format: NormalFormat,
    displacement_scale: float,
    displacement_direction: DisplacementDirection,
    tile_repeat: int,
    segments: list[TexturedSegment],
) -> str:
    """Build the final 3D mesh with the in-progress segment + all saved enabled ones."""
    if base_img is None:
        raise gr.Error("Create a base image first (Stage 1)")
    if mask is None:
        raise gr.Error("Select a region first (Step 1)")
    if tileable_patch is None:
        raise gr.Error("Create a tileable texture first (Step 3)")

    try:
        segments_to_apply = [seg for seg in segments if seg.enabled]
        current_seg = TexturedSegment(
            mask=mask.copy(),
            tileable_patch=tileable_patch.copy(),
            displacement_scale=displacement_scale,
            displacement_direction=displacement_direction,
            tile_repeat=tile_repeat,
        )
        segments_to_apply.append(current_seg)

        glb_path = create_tactile_graphic(
            base_img, segments=segments_to_apply or None, normal_format=normal_format
        )

        if not segments_to_apply:
            gr.Info("Mesh generated (no textures)")
            return glb_path

        gr.Info(f"Mesh generated with {len(segments_to_apply)} textured segments")
        return glb_path

    except Exception as e:
        raise gr.Error(f"Mesh generation failed: {e}") from e


def save_segment(
    base_img: Image.Image | None,
    mask: np.ndarray | None,
    tileable_patch: Image.Image | None,
    displacement_scale: float,
    displacement_direction: DisplacementDirection,
    tile_repeat: int,
    segments: list[TexturedSegment],
) -> list[TexturedSegment]:
    if base_img is None:
        raise gr.Error("Create a base image first (Stage 1)")
    if mask is None:
        raise gr.Error("Select a region first (Step 1)")
    if tileable_patch is None:
        raise gr.Error("Create a tileable texture first (Step 3)")

    current_seg = TexturedSegment(
        mask=mask.copy(),
        tileable_patch=tileable_patch.copy(),
        displacement_scale=displacement_scale,
        displacement_direction=displacement_direction,
        tile_repeat=tile_repeat,
    )
    segments.append(current_seg)
    return segments


# =============================================================================
# Stage 3
# =============================================================================


def handle_braille_overlay_click(
    base_img: Image.Image | None,
    box: _Box,
    evt: gr.SelectData,
) -> _Box:
    """Two-click box drawing: first click sets the start corner, second click closes the box."""
    if base_img is None:
        raise gr.Error("Create a base image first (Stage 1)")

    x, y = evt.index
    start_pt = box[0]
    if start_pt is None:
        return (x, y), None

    sx, sy = start_pt
    x1, y1 = min(sx, x), min(sy, y)
    x2, y2 = max(sx, x), max(sy, y)
    if (x2 - x1) < 20 or (y2 - y1) < 10:
        raise gr.Error("Box too small. Click to start again.")

    return (x1, y1), (x2, y2)


def render_custom_braille_overlay(
    base_img: Image.Image | None,
    braille_placements: list[BraillePlacement],
    box: _Box,
    text: str = "",
) -> Image.Image | None:
    """Canvas image with saved braille placements + optional pending draw box."""
    if base_img is None:
        return None
    braille_to_apply = [bp for bp in braille_placements if bp.enabled]
    img = render_braille_on_image(base_img, braille_to_apply, draw_bbox=True)
    if box[0] is not None:
        img = draw_pending_box(img, box[0], box[1], text)
    return img


def save_braille(
    text: str,
    box: _Box,
    braille_placements: list[BraillePlacement],
) -> list | dict:
    if not text.strip():
        raise gr.Error("Enter braille text first")
    if box[1] is None:
        raise gr.Error("Draw a box on the canvas first")

    (x1, y1), (x2, y2) = box
    braille_placements.append(
        BraillePlacement(
            text=text.strip(),
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
            enabled=True,
        )
    )

    gr.Info(f"Saved braille “{text.strip()}”.")
    return braille_placements


def generate_final_mesh(
    base_img: Image.Image | None,
    segments: list[TexturedSegment],
    normal_format: NormalFormat,
    braille_mode: Literal["standard", "custom"],
    standard_braille_text: str,
    plate_size: float,
    flat_top_ratio: float,
    bottom_padding: float,
    braille_placements: list[BraillePlacement],
    dot_height: float,
    flatten_plate: bool,
    plate_thickness: float,
) -> str | dict:
    """Build the final 3D mesh with braille annotations applied."""
    if base_img is None:
        raise gr.Error("Create a base image first (Stage 1)")

    try:
        segments_to_apply = [seg for seg in segments if seg.enabled]
        is_standard = braille_mode == "standard"
        standard_braille_text = (
            standard_braille_text.strip() if standard_braille_text else None
        )

        common_kwargs = dict(
            segments=segments_to_apply or None,
            flatten_plate=flatten_plate,
            plate_thickness=plate_thickness,
            normal_format=normal_format,
        )

        if is_standard:
            glb_path = create_tactile_graphic(
                base_img,
                standard_braille_text=standard_braille_text,
                standard_braille_plate_size=plate_size,
                standard_braille_flat_top_ratio=flat_top_ratio,
                standard_braille_bottom_padding=bottom_padding,
                **common_kwargs,
            )
            num_applied = 1
        else:
            braille_to_apply = [bp for bp in braille_placements if bp.enabled]
            glb_path = create_tactile_graphic(
                base_img,
                braille_placements=braille_to_apply or None,
                braille_dot_height=dot_height,
                **common_kwargs,
            )
            num_applied = len(braille_to_apply)

    except Exception as e:
        raise gr.Error(f"Mesh generation failed: {e}") from e

    gr.Info(
        f"Mesh generated with {len(segments_to_apply)} texture(s) "
        f"and {num_applied} braille annotation(s)"
    )
    return glb_path
