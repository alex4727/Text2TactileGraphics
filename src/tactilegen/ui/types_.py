from dataclasses import dataclass

import gradio as gr

# =============================================================================
# Component dataclasses
# =============================================================================


@dataclass(frozen=True)
class Stage1Section:
    base_img: gr.Image
    prompt_input: gr.Textbox
    model_select: gr.Radio | gr.State
    qwen_settings: gr.Group
    steps_radio: gr.Radio
    seed_num: gr.Number
    style_prefix_input: gr.Textbox
    style_suffix_input: gr.Textbox
    generate_btn: gr.Button

    mesh_output: gr.Model3D
    generate_mesh_btn: gr.Button
    next_stage_btn: gr.Button


@dataclass(frozen=True)
class SegSection:
    """Stage 2, Step 1: region selection."""

    points_state: gr.State  # list[tuple[int, int]]
    labels_state: gr.State  # list[int]
    text_mask_state: gr.State  # np.ndarray
    click_mask_state: gr.State  # np.ndarray
    mask_state: gr.State  # stores one of (text_mask, click_mask) depending on source

    source_radio: gr.Radio

    text_section: gr.Group
    overlay_img: gr.Image
    text_seg_prompt: gr.Textbox
    text_seg_btn: gr.Button

    click_section: gr.Group
    click_seg_img: gr.Image
    subtract_mode_check: gr.Checkbox
    clear_btn: gr.Button

    mask_img: gr.Image
    next_btn: gr.Button


@dataclass(frozen=True)
class TextureSection:
    """Stage 2, Step 2: texture generation."""

    geometry_state: gr.State  # np.ndarray

    texture_img: gr.Image
    texture_prompt: gr.Textbox
    steps_radio: gr.Radio
    seed_num: gr.Number
    crop_check: gr.Checkbox
    gen_texture_btn: gr.Button

    geometry_img: gr.Image
    next_btn: gr.Button


@dataclass(frozen=True)
class TilingSection:
    """Stage 2, Step 3: tileable patch generation."""

    geometry_img: gr.Image
    tiling_method_radio: gr.Radio
    highpass_check: gr.Checkbox
    highpass_freq_slider: gr.Slider
    highpass_method_radio: gr.Radio
    normal_format_radio: gr.Radio
    tile_btn: gr.Button

    tileable_patch_img: gr.Image
    tiled_img: gr.Image
    displacement_img: gr.Image
    next_btn: gr.Button


@dataclass(frozen=True)
class MeshPreviewSection:
    """Stage 2, Step 4: preview + save controls."""

    displacement_scale_slider: gr.Slider
    displacement_direction_radio: gr.Radio
    tile_repeat_slider: gr.Slider
    generate_mesh_btn: gr.Button

    mesh_output: gr.Model3D
    save_btn: gr.Button


@dataclass(frozen=True)
class Stage2Section:
    segments_state: gr.State  # list[Segment]
    add_segment_state: gr.State  # Dummy int value to trigger update
    next_stage_btn: gr.Button
    step_tabs: gr.Tabs
    seg: SegSection
    texture: TextureSection
    tiling: TilingSection
    mesh_preview: MeshPreviewSection


@dataclass(frozen=True)
class BrailleControls:
    braille_placements: gr.State  # list[BraillePlacement]
    braille_box: gr.State  # _Box

    braille_mode_radio: gr.Radio
    flatten_plate_check: gr.Checkbox
    plate_thickness_slider: gr.Slider

    # ----- Standard-mode controls
    standard_braille_settings: gr.Group
    standard_controls: gr.Row
    standard_braille_text_input: gr.Textbox
    standard_braille_preview_img: gr.Image
    standard_overlay_img: gr.Image
    plate_size_slider: gr.Slider
    flat_top_slider: gr.Slider
    bottom_padding_slider: gr.Slider

    # ----- Custom-mode controls
    custom_braille_settings: gr.Group
    custom_controls: gr.Row
    custom_braille_text_input: gr.Textbox  # text for the next saved placement
    custom_braille_preview_img: gr.Image
    custom_overlay_img: gr.Image
    dot_height_slider: gr.Slider
    clear_box_btn: gr.Button
    save_braille_btn: gr.Button


@dataclass(frozen=True)
class BrailleOutput:
    generate_mesh_btn: gr.Button
    mesh_output: gr.Model3D


@dataclass(frozen=True)
class Stage3Section:
    controls: BrailleControls
    output: BrailleOutput
