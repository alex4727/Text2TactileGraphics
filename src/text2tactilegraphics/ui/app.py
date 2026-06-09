"""Text2TactileGraphics Gradio UI — entry point + UI layout."""

import gradio as gr

from text2tactilegraphics import TexturedSegment
from text2tactilegraphics.config import debug_enabled, get_total_gpus
from text2tactilegraphics.generation.utils import mask_to_image
from text2tactilegraphics.geometry.braille import (
    STANDARD_DOT_HEIGHT,
    BraillePlacement,
    render_braille,
    render_standard_braille,
    render_standard_braille_on_image,
)
from text2tactilegraphics.secrets_ import ensure_runtime_secrets, get_gemini_api_key
from text2tactilegraphics.ui.handlers import (
    add_click,
    generate_base_image,
    generate_base_mesh,
    generate_final_mesh,
    generate_mesh_with_textures,
    generate_texture_geometry,
    generate_texture_image,
    generate_tiling_and_displacement,
    get_selected_mask_and_image,
    handle_braille_overlay_click,
    make_tileable,
    proceed_to_stage2,
    proceed_to_step2,
    proceed_to_step3,
    proceed_to_step4,
    render_custom_braille_overlay,
    save_braille,
    save_segment,
    segment_with_click,
    segment_with_text,
)
from text2tactilegraphics.ui.state import AppState, get_sensor_image_paths
from text2tactilegraphics.ui.tables import (
    ColumnSpec,
    cell_column,
    items_table,
    table_cell_image,
)
from text2tactilegraphics.ui.theme import tactile_theme
from text2tactilegraphics.ui.types_ import (
    BrailleControls,
    BrailleOutput,
    MeshPreviewSection,
    SegSection,
    Stage1Section,
    Stage2Section,
    Stage3Section,
    TextureSection,
    TilingSection,
)

CSS = """
.center {
    align-items: center;
    justify-content: center;
}
.px-8px {
    padding-left: 8px;
    padding-right: 8px;
}
"""


# =============================================================================
# Build UI
# =============================================================================


def create_demo() -> gr.Blocks:
    app_state = AppState()

    with gr.Blocks(title="Text-based Tactile Graphics Generation") as blocks:
        gr.Markdown(
            "# Text-based Tactile Graphics Generation\n\n"
            "This is the official demo for the paper “Text-based Tactile Graphics Generation for the Visually Impaired.” "
            "Use this app to generate tactile relief graphics with custom textures and optional Braille annotations."
        )

        with gr.Tabs() as tabs:
            stage1 = _build_stage1_tab(app_state)
            stage2 = _build_stage2_tab(app_state, stage1)
            _build_stage3_tab(stage1, stage2)

        if debug_enabled():
            _build_debug_settings(app_state)

        # --------- Cross-tab navigation

        stage1.next_stage_btn.click(
            fn=proceed_to_stage2,
            inputs=[stage1.base_img],
            outputs=[tabs],
            api_visibility="private",
        )

        stage2.next_stage_btn.click(
            fn=lambda: gr.Tabs(selected="stage3"),
            outputs=[tabs],
            api_visibility="private",
        )

    return blocks


def _build_debug_settings(app_state: AppState) -> None:
    with gr.Accordion("Debug settings", open=False):
        _build_save_settings(app_state)
        _build_runtime_settings(app_state)


def _build_save_settings(app_state: AppState) -> None:
    """Settings row for intermediate-result saving."""
    with gr.Row():
        save_intermediate = gr.Checkbox(
            label="Save intermediate results",
            value=app_state.save_intermediate_results,
            info="Save all intermediate images and data for debugging",
        )
        output_dir_text = gr.Textbox(
            label="Output directory",
            info="Intermediate results will be saved here",
            value=app_state.output_dir,
            scale=2,
        )

    def update_save_settings(save_flag: bool, out_dir: str) -> None:
        app_state.save_intermediate_results = save_flag
        app_state.output_dir = out_dir

    for inp in (save_intermediate, output_dir_text):
        inp.change(
            fn=update_save_settings,
            inputs=[save_intermediate, output_dir_text],
            api_visibility="private",
        )


def _build_runtime_settings(app_state: AppState) -> None:
    """VRAM / MoGe / GPU-assignment settings row."""
    with gr.Row():
        vram_mode = gr.Radio(
            choices=[("48GB", "48gb"), ("80GB", "80gb")],
            value=app_state.config.vram_mode,
            label="Qwen VRAM mode",
            info="Affects weight offloading/data type settings; must be set before first loading Qwen",
            scale=1,
        )
        geometry_type_radio = gr.Radio(
            choices=[("Normal", "normal"), ("Depth", "depth")],
            value=app_state.config.geometry_type,
            label="Geometry type",
            info="Whether to generate a normal or depth map during texture generation",
            scale=1,
        )

    # Clamp the GPU pickers to [0, n_gpus - 1]. When CUDA is unavailable we
    # still expose the inputs (the values are stashed for later) but lock the
    # range to {0} so no invalid index can be entered.
    n_gpus = max(get_total_gpus(), 1)
    max_gpu_id = n_gpus - 1
    gpu_kwargs = dict(
        value=0, precision=0, minimum=0, maximum=max_gpu_id, step=1, scale=1
    )

    with gr.Group():
        gr.Markdown(
            f"**GPU select ({n_gpus} device{'s' if n_gpus != 1 else ''})**",
            padding=True,
        )
        with gr.Row():
            gpu_base_edit = gr.Number(label="Base (if Qwen-Image-Edit)", **gpu_kwargs)
            gpu_moge = gr.Number(label="MoGe", **gpu_kwargs)
            gpu_sam = gr.Number(label="SAM3", **gpu_kwargs)
            gpu_qwen = gr.Number(
                label="Texture (Qwen Image)",
                **{**gpu_kwargs, "value": min(1, max_gpu_id)},
            )
            gpu_tile = gr.Number(
                label="Tileable (Qwen Image / SDXL)",
                **{**gpu_kwargs, "value": min(2, max_gpu_id)},
            )

    def apply_settings(
        vram: str,
        geometry_type: str,
        g_base_edit: float,
        g_moge: float,
        g_sam: float,
        g_qwen: float,
        g_tile: float,
    ) -> None:
        app_state.config.vram_mode = vram
        app_state.config.geometry_type = geometry_type
        app_state.config.gpu_assignments.update(
            {
                "qwen_base_edit": int(g_base_edit),
                "moge2": int(g_moge),
                "sam3": int(g_sam),
                "qwen_texture": int(g_qwen),
                "tile_generator": int(g_tile),
            }
        )

    inputs = [
        vram_mode,
        geometry_type_radio,
        gpu_base_edit,
        gpu_moge,
        gpu_sam,
        gpu_qwen,
        gpu_tile,
    ]
    for inp in inputs:
        inp.change(fn=apply_settings, inputs=inputs, api_visibility="private")


# =============================================================================
# Stage 1: Base Image
# =============================================================================


def _build_stage1_tab(app_state: AppState) -> Stage1Section:
    with gr.Tab("Stage 1: Create base image", id="stage1"):
        gr.Markdown(
            "First, generate or upload a base relief image. Textures will be added to this base in Stage 2."
        )
        with gr.Row():
            with gr.Column(scale=1):
                base_img = gr.Image(
                    label="Base image (generate below, or upload your own)",
                    height=384,
                    type="pil",
                    sources=["upload", "clipboard"],
                    interactive=True,
                )
                prompt_input = gr.Textbox(
                    label="Generation prompt",
                    value="a dolphin with wings",
                    lines=2,
                )
                gr.Examples(
                    examples=[
                        ["a dolphin with wings"],
                        ["a mug with handle"],
                        ["a rose flower"],
                        ["a cat sitting"],
                    ],
                    inputs=[prompt_input],
                )

                gemini_available = get_gemini_api_key(required=False) is not None

                with gr.Accordion("Advanced settings", open=False):
                    if gemini_available:
                        model_select = gr.Radio(
                            choices=[
                                ("Qwen-Image-Edit", "qwen_edit"),
                                ("Nano Banana Pro (Gemini 3 Pro)", "nano_banana_pro"),
                            ],
                            value="qwen_edit",
                            label="Generation model",
                        )
                    else:
                        model_select = gr.State("qwen_edit")

                    with gr.Group() as qwen_settings:
                        gr.Markdown("**Qwen settings**", padding=True)
                        with gr.Row():
                            steps_radio = gr.Radio(
                                choices=[4, 40], value=4, label="Steps"
                            )
                            seed_num = gr.Number(label="Seed", value=42, precision=0)

                    style_prefix_input = gr.Textbox(
                        label="Style prefix",
                        info="Prepended to the prompt to control style",
                        value=app_state.config.base_image_style_prefix,
                        max_lines=3,
                    )
                    style_suffix_input = gr.Textbox(
                        label="Style suffix",
                        info="Appended to the prompt to control style",
                        value=app_state.config.base_image_style_suffix,
                        max_lines=3,
                    )

                generate_btn = gr.Button("Generate base image", variant="secondary")

            with gr.Column(scale=1):
                mesh_output = gr.Model3D(label="Base mesh", height=384)
                generate_mesh_btn = gr.Button(
                    "Preview base mesh (optional)", size="md", variant="secondary"
                )
                next_stage_btn = gr.Button("Next stage →", variant="primary")

    stage1 = Stage1Section(
        base_img=base_img,
        prompt_input=prompt_input,
        model_select=model_select,
        qwen_settings=qwen_settings,
        steps_radio=steps_radio,
        seed_num=seed_num,
        style_prefix_input=style_prefix_input,
        style_suffix_input=style_suffix_input,
        generate_btn=generate_btn,
        mesh_output=mesh_output,
        generate_mesh_btn=generate_mesh_btn,
        next_stage_btn=next_stage_btn,
    )
    _wire_stage1_events(app_state, stage1)
    return stage1


def _wire_stage1_events(app_state: AppState, stage1: Stage1Section) -> None:
    # Show Qwen settings iff Qwen model is selected
    if isinstance(stage1.model_select, gr.Radio):
        stage1.model_select.change(
            fn=lambda m: gr.Group(visible=(m == "qwen_edit")),
            inputs=[stage1.model_select],
            outputs=[stage1.qwen_settings],
            api_visibility="private",
        )

    stage1.generate_btn.click(
        fn=lambda p, m, steps, seed: generate_base_image(p, m, steps, seed, app_state),
        inputs=[
            stage1.prompt_input,
            stage1.model_select,
            stage1.steps_radio,
            stage1.seed_num,
        ],
        outputs=[stage1.base_img],
        api_name="generate_base_image",
    )

    def _set_style_prefix(value: str) -> None:
        app_state.config.base_image_style_prefix = value

    def _set_style_suffix(value: str) -> None:
        app_state.config.base_image_style_suffix = value

    stage1.style_prefix_input.change(
        fn=_set_style_prefix,
        inputs=[stage1.style_prefix_input],
        api_visibility="private",
    )
    stage1.style_suffix_input.change(
        fn=_set_style_suffix,
        inputs=[stage1.style_suffix_input],
        api_visibility="private",
    )

    stage1.base_img.upload(
        fn=lambda: gr.Info("Uploaded base image"), api_visibility="private"
    )

    stage1.generate_mesh_btn.click(
        fn=generate_base_mesh,
        inputs=[stage1.base_img],
        outputs=[stage1.mesh_output],
        api_name="preview_base_mesh",
    )


# =============================================================================
# Stage 2: Segment + Texture + Tiling
# =============================================================================


def _build_stage2_tab(app_state: AppState, stage1: Stage1Section) -> Stage2Section:
    with gr.Tab("Stage 2: Add textures", id="stage2"):
        gr.Markdown(
            "Add custom textures to segments of the base relief. "
            "You may add multiple textured segments; saved segments are displayed in the table below. "
            "When finished, click “Next stage.”"
        )

        segments_state = gr.State([])
        add_segment_state = gr.State(0)

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Saved segments")
                render_segments_table(segments_state)
            with gr.Column():
                next_stage_btn = gr.Button("Next stage →", variant="primary")

        gr.Markdown("---")
        gr.Markdown("## Create new segment")

        with gr.Tabs() as step_tabs:
            with gr.Tab("Step 1: Select region", id="step1"):
                seg = _build_segment_section()
            with gr.Tab("Step 2: Create texture", id="step2"):
                texture = _build_texture_section()
            with gr.Tab("Step 3: Make tileable", id="step3"):
                tiling = _build_tiling_section()
            with gr.Tab("Step 4: Preview & save", id="step4"):
                mesh = _build_segment_preview_section()

    stage2 = Stage2Section(
        segments_state=segments_state,
        add_segment_state=add_segment_state,
        next_stage_btn=next_stage_btn,
        step_tabs=step_tabs,
        seg=seg,
        texture=texture,
        tiling=tiling,
        mesh_preview=mesh,
    )
    _wire_stage2_events(app_state, stage1, stage2)
    return stage2


def render_segments_table(segments_state: gr.State) -> None:
    def render_cells(seg: TexturedSegment) -> None:
        mask_thumb = mask_to_image(seg.mask) if seg.mask is not None else None
        cell_column(2, 64, lambda: table_cell_image(mask_thumb))
        cell_column(2, 64, lambda: table_cell_image(seg.tileable_patch))

    segment_table_cols: list[ColumnSpec] = [("Mask", 2, 64), ("Texture", 2, 64)]

    items_table(
        items_state=segments_state,
        empty_message="*No segments saved*",
        columns=segment_table_cols,
        render_content_cells=render_cells,
    )


def _build_segment_section() -> SegSection:
    points_state = gr.State([])
    labels_state = gr.State([])
    text_mask_state = gr.State()
    click_mask_state = gr.State()
    mask_state = gr.State()

    gr.Markdown(
        "Select a region to be textured, either automatically via text prompt or through clicks."
    )

    with gr.Row():
        with gr.Column():
            source_radio = gr.Radio(
                choices=[
                    ("Text-based", "text"),
                    ("Click-based", "click"),
                ],
                value="text",
                label="Selection mode",
            )

            with gr.Group(elem_id="text-selection-section") as text_section:
                overlay_img = gr.Image(
                    label="Selection overlay", height=384, interactive=False
                )
                text_seg_prompt = gr.Textbox(
                    label="Selection prompt", value="dolphin", scale=3
                )
                text_seg_btn = gr.Button(
                    "Auto-select region",
                    size="md",
                    variant="secondary",
                    scale=1,
                )

            with gr.Group(
                visible=False, elem_id="click-based-selection-section"
            ) as click_section:
                click_seg_img = gr.Image(
                    label="Click to select region",
                    type="pil",
                    height=384,
                    interactive=False,
                )
                subtract_mode_check = gr.Checkbox(
                    label="Subtract mode",
                    value=False,
                    info="Check to remove areas from selection via clicks",
                )
                clear_btn = gr.Button("Clear selection", size="md", variant="stop")

        with gr.Column():
            mask_img = gr.Image(label="Selection mask", height=384, interactive=False)
            next_btn = gr.Button("Next step →", variant="primary")

    return SegSection(
        points_state=points_state,
        labels_state=labels_state,
        text_mask_state=text_mask_state,
        click_mask_state=click_mask_state,
        mask_state=mask_state,
        source_radio=source_radio,
        text_section=text_section,
        overlay_img=overlay_img,
        text_seg_prompt=text_seg_prompt,
        text_seg_btn=text_seg_btn,
        click_section=click_section,
        click_seg_img=click_seg_img,
        subtract_mode_check=subtract_mode_check,
        clear_btn=clear_btn,
        mask_img=mask_img,
        next_btn=next_btn,
    )


def _build_texture_section() -> TextureSection:
    geometry_state = gr.State(None)

    gr.Markdown(
        "Generate a texture or upload your own. "
        "Texture geometry can be produced from an input image (generated or uploaded), or uploaded directly as a normal map."
    )
    with gr.Row():
        with gr.Column():
            texture_img = gr.Image(
                label="Texture image (generate below, or upload your own)",
                height=384,
                type="pil",
                sources=["upload", "clipboard"],
                interactive=True,
            )
            texture_prompt = gr.Textbox(
                label="Texture generation prompt",
                value="an avocado skin",
                lines=2,
            )

            with gr.Accordion("Advanced settings", open=False):
                with gr.Row():
                    steps_radio = gr.Radio(choices=[4, 40], value=4, label="Steps")
                    seed_num = gr.Number(label="Seed", value=42, precision=0)
                crop_check = gr.Checkbox(
                    label="Center crop",
                    info="Crop the geometry map to a center region",
                    value=True,
                )

            gen_texture_btn = gr.Button("Generate texture", variant="secondary")

        with gr.Column():
            geometry_img = gr.Image(
                label="Geometry map (derived from texture image, or upload your own)",
                height=384,
                type="pil",
                sources=["upload", "clipboard"],
                interactive=True,
            )
            sensor_examples = get_sensor_image_paths()
            if sensor_examples:
                gr.Examples(
                    examples=[[p] for p in sensor_examples],
                    inputs=[geometry_img],
                    label="Examples (sensor-captured)",
                    examples_per_page=6,
                )
            next_btn = gr.Button("Next step →", variant="primary")

    return TextureSection(
        geometry_state=geometry_state,
        texture_img=texture_img,
        texture_prompt=texture_prompt,
        steps_radio=steps_radio,
        seed_num=seed_num,
        crop_check=crop_check,
        gen_texture_btn=gen_texture_btn,
        geometry_img=geometry_img,
        next_btn=next_btn,
    )


def _build_tiling_section() -> TilingSection:
    gr.Markdown(
        "Smooth the seams of the texture created in Step 2 to make it tileable. "
        "If the texture is already tileable, you may skip this step."
    )
    with gr.Row():
        with gr.Column():
            geometry_img = gr.Image(
                label="Geometry map (from Step 2)",
                height=192,
                interactive=False,
                type="pil",
            )

            with gr.Accordion("Advanced settings", open=False):
                tiling_method_radio = gr.Radio(
                    choices=[
                        ("Intra-tile inpainting", "intra_tile_inpainting"),
                        ("Inter-tile inpainting", "inter_tile_inpainting"),
                        ("Tiled Diffusion", "tiled_diffusion"),
                    ],
                    value="intra_tile_inpainting",
                    label="Tiling method",
                )
                highpass_check = gr.Checkbox(
                    value=True,
                    label="Enable high-pass filter",
                    info="Remove low-frequency gradients from normal map before tiling (recommended)",
                )
                highpass_freq_slider = gr.Slider(
                    minimum=5,
                    maximum=200,
                    value=120,
                    step=5,
                    precision=0,
                    label="Frequency threshold",
                    info="Larger value = less aggressive filtering. Typical values: 5-50 for texture, 100-200 for gentle filtering.",
                )
                highpass_method_radio = gr.Radio(
                    choices=[
                        ("Per channel (recommended)", "per_channel"),
                        ("Height integration", "height_integration"),
                    ],
                    value="per_channel",
                    label="Filtering method",
                    info="Per channel: Fast and simple, works for most cases. Height integration: More geometrically accurate but slower.",
                )
                normal_format_radio = gr.Radio(
                    choices=[
                        ("OpenGL (Y-up)", "opengl"),
                        ("DirectX (Y-down)", "directx"),
                    ],
                    value="opengl",
                    label="Normal map convention",
                    info="OpenGL: Green channel = up. DirectX: Green channel = down. Try both if texture looks inverted.",
                )

            tile_btn = gr.Button("Make tileable", variant="secondary")

        with gr.Column():
            tileable_patch_img = gr.Image(
                label="Tileable patch", height=192, type="pil", interactive=False
            )
            tiled_img = gr.Image(
                label="3×3 tiled", height=192, type="pil", interactive=False
            )
            displacement_img = gr.Image(
                label="Displacement map",
                height=192,
                interactive=False,
                type="pil",
            )
            next_btn = gr.Button("Next step →", variant="primary")

    return TilingSection(
        geometry_img=geometry_img,
        tiling_method_radio=tiling_method_radio,
        highpass_check=highpass_check,
        highpass_freq_slider=highpass_freq_slider,
        highpass_method_radio=highpass_method_radio,
        normal_format_radio=normal_format_radio,
        tile_btn=tile_btn,
        tileable_patch_img=tileable_patch_img,
        tiled_img=tiled_img,
        displacement_img=displacement_img,
        next_btn=next_btn,
    )


def _build_segment_preview_section() -> MeshPreviewSection:
    gr.Markdown("Preview the textured segment and save if satisfied.")
    with gr.Row():
        with gr.Column():
            with gr.Accordion("Advanced settings", open=False):
                displacement_scale_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=0.1,
                    label="Displacement scale (mm)",
                    info="Strength of surface texture detail",
                )
                displacement_direction_radio = gr.Radio(
                    choices=[("Normal", "normal"), ("Z", "z")],
                    value="normal",
                    label="Displacement direction",
                    info="Normal: natural; Z: vertical",
                )
                tile_repeat_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    precision=0,
                    label="Tile repeat (N×N)",
                    info="Number of tiles for texture; higher = finer texture",
                )
            generate_mesh_btn = gr.Button("Preview mesh", variant="secondary")

        with gr.Column():
            mesh_output = gr.Model3D(label="Mesh with textured segment", height=384)
            save_btn = gr.Button("Save segment ✓", variant="primary", size="lg")

    return MeshPreviewSection(
        displacement_scale_slider=displacement_scale_slider,
        displacement_direction_radio=displacement_direction_radio,
        tile_repeat_slider=tile_repeat_slider,
        generate_mesh_btn=generate_mesh_btn,
        mesh_output=mesh_output,
        save_btn=save_btn,
    )


def _wire_stage2_events(
    app_state: AppState, stage1: Stage1Section, stage2: Stage2Section
) -> None:
    seg = stage2.seg
    texture = stage2.texture
    tiling = stage2.tiling
    mesh = stage2.mesh_preview

    # ------ segmentation
    seg.text_seg_btn.click(
        fn=lambda img, text: segment_with_text(img, text, app_state=app_state),
        inputs=[stage1.base_img, seg.text_seg_prompt],
        outputs=[seg.text_mask_state, seg.overlay_img],
        api_visibility="private",
    )

    seg.click_seg_img.select(
        fn=add_click,
        inputs=[
            seg.points_state,
            seg.labels_state,
            seg.subtract_mode_check,
        ],
        outputs=[seg.points_state, seg.labels_state],
        api_visibility="private",
    )

    # Re-run click-based segmentation when click input changes
    seg.points_state.change(
        fn=lambda img, points, labels: (
            (None, img)
            if not points or not labels
            else segment_with_click(img, points, labels, app_state=app_state)
        ),
        inputs=[stage1.base_img, seg.points_state, seg.labels_state],
        outputs=[seg.click_mask_state, seg.click_seg_img],
        api_visibility="private",
    )

    seg.clear_btn.click(
        fn=lambda: ([], []),
        outputs=[seg.points_state, seg.labels_state],
        api_visibility="private",
    )

    # Update mask image & visible sections when source changes
    seg.source_radio.change(
        fn=lambda src: (
            gr.Group(visible=src == "text"),
            gr.Group(visible=src == "click"),
        ),
        inputs=[seg.source_radio],
        outputs=[seg.text_section, seg.click_section],
        api_visibility="private",
    )
    # Update mask_state when its dependencies change
    mask_state_deps = [seg.source_radio, seg.text_mask_state, seg.click_mask_state]
    for dep in mask_state_deps:
        dep.change(
            fn=get_selected_mask_and_image,
            inputs=mask_state_deps,
            outputs=[seg.mask_state, seg.mask_img],
            api_visibility="private",
        )

    # ------ texture
    texture.gen_texture_btn.click(
        fn=lambda p, s, sd: generate_texture_image(p, s, sd, app_state),
        inputs=[texture.texture_prompt, texture.steps_radio, texture.seed_num],
        outputs=[texture.texture_img],
        api_name="generate_texture_image",
    )

    texture.texture_img.upload(
        fn=lambda: gr.Info("Uploaded texture image"), api_visibility="private"
    )

    # Regenerate geometry when texture image changes
    texture.texture_img.change(
        fn=lambda img, crop: (
            (None, None)
            if img is None
            else generate_texture_geometry(img, crop, app_state)
        ),
        inputs=[texture.texture_img, texture.crop_check],
        outputs=[texture.geometry_state, texture.geometry_img],
        api_name="generate_texture_geometry",
    )

    # Mirror geometry map into Step 3 when it changes
    texture.geometry_img.change(
        fn=lambda img: (img, img),
        inputs=[texture.geometry_img],
        outputs=[tiling.geometry_img, tiling.tileable_patch_img],
        api_visibility="private",
    )

    # Derive the 3×3 tiled preview + displacement map whenever the tileable
    # patch changes.
    tiling.tileable_patch_img.change(
        fn=lambda img, fmt: (
            (None, None) if img is None else generate_tiling_and_displacement(img, fmt)
        ),
        inputs=[tiling.tileable_patch_img, tiling.normal_format_radio],
        outputs=[tiling.tiled_img, tiling.displacement_img],
        api_visibility="private",
    )

    tiling.tile_btn.click(
        fn=lambda img, method, hp, freq, hpm: make_tileable(
            geometry_img=img,
            method=method,
            steps=100 if method == "tiled_diffusion" else 10,
            seed=42,
            use_highpass=hp,
            highpass_freq_threshold=int(freq),
            highpass_method=hpm,
            app_state=app_state,
        ),
        inputs=[
            tiling.geometry_img,
            tiling.tiling_method_radio,
            tiling.highpass_check,
            tiling.highpass_freq_slider,
            tiling.highpass_method_radio,
        ],
        outputs=[tiling.tileable_patch_img],
        api_name="make_tileable",
    )

    # ------ mesh preview
    mesh.generate_mesh_btn.click(
        fn=lambda img, mask, patch, normal, ds_mm, dd, scale_mm, segs: (
            generate_mesh_with_textures(
                img, mask, patch, normal, ds_mm / 1000, dd, scale_mm, segs
            )
        ),
        inputs=[
            stage1.base_img,
            seg.mask_state,
            tiling.tileable_patch_img,
            tiling.normal_format_radio,
            mesh.displacement_scale_slider,
            mesh.displacement_direction_radio,
            mesh.tile_repeat_slider,
            stage2.segments_state,
        ],
        outputs=[mesh.mesh_output],
        api_name="preview_textured_mesh",
    )

    # ------ save segment + refresh table
    mesh.save_btn.click(
        fn=lambda img, mask, patch, ds_mm, dd, scale, segs, add_seg: (
            save_segment(img, mask, patch, ds_mm / 1000, dd, int(scale), segs),
            add_seg + 1,
        ),
        inputs=[
            stage1.base_img,
            seg.mask_state,
            tiling.tileable_patch_img,
            mesh.displacement_scale_slider,
            mesh.displacement_direction_radio,
            mesh.tile_repeat_slider,
            stage2.segments_state,
            stage2.add_segment_state,
        ],
        outputs=[stage2.segments_state, stage2.add_segment_state],
        api_visibility="private",
    )
    # ------ Step-to-step navigation
    seg.next_btn.click(
        fn=proceed_to_step2,
        inputs=[seg.mask_img],
        outputs=[stage2.step_tabs],
        api_visibility="private",
    )
    texture.next_btn.click(
        fn=proceed_to_step3,
        inputs=[texture.geometry_img],
        outputs=[stage2.step_tabs],
        api_visibility="private",
    )
    tiling.next_btn.click(
        fn=proceed_to_step4,
        inputs=[tiling.tileable_patch_img],
        outputs=[stage2.step_tabs],
        api_visibility="private",
    )

    # Successfully saving a segment modifies add_segment_state to trigger reset,
    # so reset components don't display error message if save_segment fails
    stage2.add_segment_state.change(
        fn=lambda img: (
            [],
            [],
            None,
            None,
            img,
            img,
            None,
            None,
            gr.Tabs(selected="step1"),
        ),
        inputs=[stage1.base_img],
        outputs=[
            seg.points_state,
            seg.labels_state,
            seg.text_mask_state,
            seg.click_mask_state,
            seg.overlay_img,
            seg.click_seg_img,
            texture.texture_img,
            mesh.mesh_output,
            stage2.step_tabs,
        ],
        api_visibility="private",
    )

    # Reset Stage 2 state when Stage 1 image changes
    stage1.base_img.change(
        fn=lambda img: ([], [], [], None, None, img, img, None),
        inputs=[stage1.base_img],
        outputs=[
            stage2.segments_state,
            seg.points_state,
            seg.labels_state,
            seg.text_mask_state,
            seg.click_mask_state,
            seg.overlay_img,
            seg.click_seg_img,
            mesh.mesh_output,
        ],
        api_visibility="private",
    )


# =============================================================================
# Stage 3: Braille
# =============================================================================


def _build_stage3_tab(stage1: Stage1Section, stage2: Stage2Section) -> None:
    with gr.Tab("Stage 3: Braille & finish", id="stage3"):
        gr.Markdown(
            "Add braille annotations (optional). "
            "You may add a single standard annotation or multiple custom annotations. "
            "When finished, click “Generate final mesh.”"
        )
        with gr.Row():
            with gr.Column():
                controls = _build_braille_controls()
            with gr.Column():
                output = _build_braille_output()

        stage3 = Stage3Section(controls=controls, output=output)
        _wire_stage3_events(stage1, stage2, stage3)


def _build_braille_controls() -> BrailleControls:
    braille_placements = gr.State([])
    braille_box = gr.State((None, None))

    # ----- Mode
    braille_mode_radio = gr.Radio(
        choices=[
            ("Standard (fixed location)", "standard"),
            ("Custom location", "custom"),
        ],
        value="standard",
        label="Annotation mode",
    )

    # ----- Standard-mode text input (visible when mode == "standard")
    with gr.Row() as standard_controls:
        with gr.Column():
            standard_braille_text_input = gr.Textbox(label="Text", value="dolphin")
            standard_braille_preview_img = gr.Image(label="Braille preview", height=96)
            standard_overlay_img = gr.Image(
                label="Plate preview", height=384, interactive=False
            )

    # ----- Custom-mode draw-box workflow (visible when mode == "custom")
    with gr.Row(visible=False) as custom_controls:
        with gr.Column():
            gr.Markdown("## Saved braille annotations")
            render_braille_table(braille_placements)

            gr.Markdown("---")
            gr.Markdown("## New braille annotation")

            braille_text_input = gr.Textbox(label="Text", value="dolphin")
            braille_preview_img = gr.Image(label="Braille preview", height=96)

            with gr.Group():
                canvas_img = gr.Image(
                    label="Specify location (click once to set start corner, again to set end corner)",
                    height=384,
                )
                clear_box_btn = gr.Button("Clear selection", size="md", variant="stop")

            save_braille_btn = gr.Button("Save braille ✓", variant="secondary")

    with gr.Accordion("Advanced settings", open=False):
        flatten_plate_check = gr.Checkbox(
            value=True,
            label="Flatten plate background (recommended)",
            info="Flatten plate to uniform height and close mesh for 3D printing",
        )
        plate_thickness_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=4,
            step=0.1,
            label="Plate thickness (mm)",
            info="Thickness of closed plate bottom",
        )

        with gr.Group() as standard_braille_settings:
            gr.Markdown("**Standard mode settings**", padding=True)
            plate_size_slider = gr.Slider(
                minimum=5,
                maximum=50,
                value=12,
                step=1,
                label="Plate size (cm)",
                info="Side length of square plate, affects relative size of braille",
            )
            bottom_padding_slider = gr.Slider(
                minimum=-0.2,
                maximum=3,
                value=0.1,
                step=0.1,
                label="Bottom padding (cm)",
                info="Distance from plate’s bottom edge to braille",
            )
            flat_top_radio = gr.Slider(
                minimum=0.0,
                maximum=0.5,
                value=0.3,
                step=0.01,
                label="Braille dot flat-top ratio",
                info="Fraction of braille dot radius that is flattened",
            )

        with gr.Group(visible=False) as custom_braille_settings:
            gr.Markdown("**Custom mode settings**", padding=True)
            dot_height_slider = gr.Slider(
                minimum=0,
                maximum=1.5,
                value=STANDARD_DOT_HEIGHT * 1000,
                step=0.01,
                label="Braille dot height (mm)",
            )

    return BrailleControls(
        braille_placements=braille_placements,
        braille_box=braille_box,
        braille_mode_radio=braille_mode_radio,
        flatten_plate_check=flatten_plate_check,
        plate_thickness_slider=plate_thickness_slider,
        # standard mode
        standard_braille_settings=standard_braille_settings,
        standard_controls=standard_controls,
        standard_braille_text_input=standard_braille_text_input,
        standard_braille_preview_img=standard_braille_preview_img,
        standard_overlay_img=standard_overlay_img,
        plate_size_slider=plate_size_slider,
        flat_top_slider=flat_top_radio,
        bottom_padding_slider=bottom_padding_slider,
        # custom mode
        custom_braille_settings=custom_braille_settings,
        custom_controls=custom_controls,
        custom_braille_text_input=braille_text_input,
        custom_braille_preview_img=braille_preview_img,
        dot_height_slider=dot_height_slider,
        custom_overlay_img=canvas_img,
        save_braille_btn=save_braille_btn,
        clear_box_btn=clear_box_btn,
    )


def _build_braille_output() -> BrailleOutput:
    """Stage 3 right column: 3D mesh output + final-mesh trigger."""
    with gr.Column(scale=2):
        mesh_output = gr.Model3D(label="Final mesh", height=512)
        generate_mesh_btn = gr.Button(
            "Generate final mesh", variant="primary", size="lg"
        )

    return BrailleOutput(
        generate_mesh_btn=generate_mesh_btn,
        mesh_output=mesh_output,
    )


def render_braille_table(braille_placements: gr.State) -> None:
    def render_cells(bp: BraillePlacement) -> None:
        preview = render_braille(bp.text, width=394, height=96)
        cell_column(3, 64, lambda: table_cell_image(preview))
        cell_column(2, 64, lambda: gr.Markdown(bp.text, elem_classes="px-8px"))

    braille_table_cols: list[ColumnSpec] = [("Braille", 3, 64), ("Text", 2, 64)]

    items_table(
        items_state=braille_placements,
        empty_message="*No annotations saved*",
        columns=braille_table_cols,
        render_content_cells=render_cells,
    )


def _wire_stage3_events(
    stage1: Stage1Section, stage2: Stage2Section, stage3: Stage3Section
) -> None:
    controls = stage3.controls
    output = stage3.output

    controls.standard_braille_text_input.change(
        fn=lambda text, ps_cm: (
            None
            if text is None
            else render_standard_braille(
                text.strip(), plate_size=ps_cm / 100, width=512, height=128
            )
        ),
        inputs=[controls.standard_braille_text_input, controls.plate_size_slider],
        outputs=[controls.standard_braille_preview_img],
        api_visibility="private",
    )

    # Update standard overlay when its inputs change
    for dep in (
        stage1.base_img,
        controls.standard_braille_text_input,
        controls.plate_size_slider,
    ):
        dep.change(
            fn=lambda img, t, ps_cm, bp_cm: (
                None
                if img is None
                else render_standard_braille_on_image(
                    img, t, plate_size=ps_cm / 100, bottom_padding=bp_cm / 100
                )
            ),
            inputs=[
                stage1.base_img,
                controls.standard_braille_text_input,
                controls.plate_size_slider,
                controls.bottom_padding_slider,
            ],
            outputs=[controls.standard_overlay_img],
            api_visibility="private",
        )

    controls.custom_braille_text_input.change(
        fn=lambda text: (
            None
            if text is None
            else render_braille(text.strip(), width=512, height=128)
        ),
        inputs=[controls.custom_braille_text_input],
        outputs=[controls.custom_braille_preview_img],
        api_visibility="private",
    )

    # ------ interactive box drawing
    controls.custom_overlay_img.select(
        fn=handle_braille_overlay_click,
        inputs=[stage1.base_img, controls.braille_box],
        outputs=[controls.braille_box],
        api_visibility="private",
    )

    controls.clear_box_btn.click(
        fn=lambda: (None, None),
        outputs=[controls.braille_box],
        api_visibility="private",
    )

    # Box resets on save
    controls.save_braille_btn.click(
        fn=lambda t, b, p: (save_braille(t, b, p), (None, None)),
        inputs=[
            controls.custom_braille_text_input,
            controls.braille_box,
            controls.braille_placements,
        ],
        outputs=[controls.braille_placements, controls.braille_box],
        api_visibility="private",
    )

    # Update canvas when braille placements change
    for dep in (controls.braille_placements, controls.braille_box):
        dep.change(
            fn=render_custom_braille_overlay,
            inputs=[
                stage1.base_img,
                controls.braille_placements,
                controls.braille_box,
                controls.custom_braille_text_input,
            ],
            outputs=[controls.custom_overlay_img],
            api_visibility="private",
        )

    # ------ mode toggle
    controls.braille_mode_radio.change(
        fn=lambda mode: (
            gr.Group(visible=mode == "standard"),
            gr.Row(visible=mode == "standard"),
            gr.Group(visible=mode == "custom"),
            gr.Row(visible=mode == "custom"),
        ),
        inputs=[controls.braille_mode_radio],
        outputs=[
            controls.standard_braille_settings,
            controls.standard_controls,
            controls.custom_braille_settings,
            controls.custom_controls,
        ],
        api_visibility="private",
    )

    # ------ final mesh generation
    output.generate_mesh_btn.click(
        fn=lambda img,
        segs,
        normal,
        bm,
        text,
        ps_cm,
        ft,
        bp_cm,
        brailles,
        dh_mm,
        fp,
        pt_mm: (
            generate_final_mesh(
                img,
                segs,
                normal,
                bm,
                text,
                ps_cm / 100,
                ft,
                bp_cm / 100,
                brailles,
                dh_mm / 1000,
                fp,
                pt_mm / 1000,
            )
        ),
        inputs=[
            stage1.base_img,
            stage2.segments_state,
            stage2.tiling.normal_format_radio,
            controls.braille_mode_radio,
            controls.standard_braille_text_input,
            controls.plate_size_slider,
            controls.flat_top_slider,
            controls.bottom_padding_slider,
            controls.braille_placements,
            controls.dot_height_slider,
            controls.flatten_plate_check,
            controls.plate_thickness_slider,
        ],
        outputs=[output.mesh_output],
        api_name="generate_final_mesh",
    )

    # Reset Stage 3 state when Stage 1 image changes
    stage1.base_img.change(
        fn=lambda img: (img, [], (None, None), None),
        inputs=[stage1.base_img],
        outputs=[
            controls.custom_overlay_img,
            controls.braille_placements,
            controls.braille_box,
            output.mesh_output,
        ],
        api_visibility="private",
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    ensure_runtime_secrets(hf=True, gemini=False)
    demo = create_demo()
    demo.launch(theme=tactile_theme, css=CSS)
