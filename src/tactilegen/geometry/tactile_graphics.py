"""End-to-end image → tactile graphic orchestration."""

import logging
import os
import tempfile
from uuid import uuid4

import trimesh
from PIL import Image
from scipy.ndimage import zoom

from tactilegen.generation.segmentation import SegmentationEngine
from tactilegen.generation.texture_generation import GeometryEstimator
from tactilegen.geometry.displacement import (
    DisplacementDirection,
    NormalFormat,
    apply_braille_displacements,
    apply_segment_displacements,
    apply_standard_braille_displacement,
)
from tactilegen.geometry.utils import (
    depth2mesh,
    flatten_and_close_plate,
    pv2trimesh,
)

logger = logging.getLogger(__name__)


def create_tactile_graphic(
    image: Image.Image,
    output_path: str | None = None,
    *,
    geom_estimator: GeometryEstimator | None = None,
    seg_engine: SegmentationEngine | None = None,
    segments: list | None = None,
    displacement_scale: float = 0.005,
    displacement_direction: DisplacementDirection = "normal",
    braille_placements: list | None = None,
    braille_dot_height: float = 0.005,
    standard_braille_text: str | None = None,
    standard_braille_plate_size: float = 0.12,
    standard_braille_flat_top_ratio: float = 0.3,
    standard_braille_bottom_padding: float = 0.005,
    flatten_plate: bool = True,
    plate_thickness: float = 0.002,
    plate_segmentation_prompt: str = "a white marble plate background",
    plate_segmentation_confidence: float = 0.3,
    target_resolution: int = 1024,
    normal_format: NormalFormat = "opengl",
) -> str:
    """Convert an image to a 2.5D tactile graphic and export as GLB.

    Pipeline:

    1. Image depth estimation → base mesh.
    2. Add per-segment textures (if ``segments`` given).
    3. Plate flattening + closing (if `flatten_plate=True`).
    4. Add braille (if ``standard_braille_text`` and/or ``braille_placements`` given).

    Returns ``glb_file_path``.
    """
    if geom_estimator is None:
        geom_estimator = GeometryEstimator()
    if seg_engine is None:
        seg_engine = SegmentationEngine()

    # Base depth to mesh
    depth = geom_estimator.compute_depth(image)
    if depth.shape != (target_resolution, target_resolution):
        depth = zoom(depth, target_resolution / depth.shape[0], order=1)
    mesh_pv = depth2mesh(depth, smooth=True, decimate=True)
    mesh = pv2trimesh(mesh_pv)

    if segments:
        logger.info("=== Applying displacements from %d segments ===", len(segments))
        mesh = apply_segment_displacements(
            mesh=mesh,
            segments=segments,
            scale=displacement_scale,
            direction=displacement_direction,
            target_resolution=target_resolution,
            normal_format=normal_format,
        )

    if flatten_plate:
        mesh = _flatten_plate(
            mesh,
            image,
            seg_engine,
            target_resolution=target_resolution,
            plate_thickness=plate_thickness,
            plate_segmentation_prompt=plate_segmentation_prompt,
            plate_segmentation_confidence=plate_segmentation_confidence,
        )

    mesh = _apply_braille(
        mesh,
        image,
        target_resolution=target_resolution,
        braille_placements=braille_placements,
        braille_dot_height=braille_dot_height,
        standard_braille_text=standard_braille_text,
        standard_braille_plate_size=standard_braille_plate_size,
        standard_braille_flat_top_ratio=standard_braille_flat_top_ratio,
        standard_braille_bottom_padding=standard_braille_bottom_padding,
    )

    logger.info("Exporting to GLB file...")
    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(), f"tactilegen_mesh_{uuid4()}.glb"
        )
    mesh.export(output_path)
    logger.info("Export done.")

    return output_path


def _flatten_plate(
    mesh: trimesh.Trimesh,
    image: Image.Image,
    seg_engine: SegmentationEngine,
    *,
    target_resolution: int,
    plate_thickness: float,
    plate_segmentation_prompt: str,
    plate_segmentation_confidence: float,
) -> trimesh.Trimesh:
    """SAM3-segment the plate, flatten it, and close the mesh."""
    logger.info("=== Flattening plate and closing mesh ===")

    plate_segments = seg_engine.segment_with_text(
        image,
        plate_segmentation_prompt,
        confidence_threshold=plate_segmentation_confidence,
    )
    if not plate_segments:
        logger.warning(
            "Could not segment plate region, skipping flattening", stacklevel=2
        )
        return mesh

    plate_mask, score = plate_segments[0]
    logger.info("Found plate region with confidence %.3f", score)
    if plate_mask.shape != (target_resolution, target_resolution):
        # Nearest-neighbor + threshold to keep the mask boolean.
        plate_mask = (
            zoom(plate_mask, target_resolution / plate_mask.shape[0], order=0) > 0.5
        )

    mesh = flatten_and_close_plate(
        mesh=mesh,
        plate_mask=plate_mask,
        plate_thickness=plate_thickness,
        foreground_z_threshold=0.003,
    )
    logger.info("=== Plate flattened, closed with bottom ===")
    return mesh


def _apply_braille(
    mesh: trimesh.Trimesh,
    image: Image.Image,
    *,
    target_resolution: int,
    braille_placements: list | None,
    braille_dot_height: float,
    standard_braille_text: str | None,
    standard_braille_plate_size: float,
    standard_braille_flat_top_ratio: float,
    standard_braille_bottom_padding: float,
) -> trimesh.Trimesh:
    """Apply braille displacement (standard and/or bounding-box)."""
    if standard_braille_text:
        logger.info("=== Applying standard braille: %r ===", standard_braille_text)
        mesh = apply_standard_braille_displacement(
            mesh=mesh,
            text=standard_braille_text,
            plate_size=standard_braille_plate_size,
            target_resolution=target_resolution,
            flat_top_ratio=standard_braille_flat_top_ratio,
            bottom_padding=standard_braille_bottom_padding,
        )

    if braille_placements:
        logger.info(
            "=== Applying braille from %d placements ===", len(braille_placements)
        )
        mesh = apply_braille_displacements(
            mesh=mesh,
            placements=braille_placements,
            image_size=(image.width, image.height),
            target_resolution=target_resolution,
            dot_height=braille_dot_height,
        )

    return mesh
