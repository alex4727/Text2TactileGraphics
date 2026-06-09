from pathlib import Path
from unittest.mock import Mock, create_autospec

import numpy as np
import pytest
import trimesh
from PIL import Image
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from text2tactilegraphics.generation.segmentation import SegmentationEngine
from text2tactilegraphics.generation.texture_generation import GeometryEstimator
from text2tactilegraphics.geometry.tactile_graphics import create_tactile_graphic

_DATA_DIR = Path(__file__).parent / "data"


def _geom_estimator() -> Mock:
    geom = create_autospec(GeometryEstimator)
    geom.compute_depth.return_value = np.load(_DATA_DIR / "dolphin_depth.npy")
    return geom


def _seg_engine() -> Mock:
    seg = create_autospec(SegmentationEngine)
    seg.segment_with_text.return_value = [
        (np.load(_DATA_DIR / "dolphin_plate_mask.npy"), 1)
    ]
    return seg


@pytest.mark.slow
class TestCreateTactileGraphic:
    def test_create_tactile_graphic(
        self,
        tmp_path: Path,
        ndarrays_regression: NDArraysRegressionFixture,
    ):
        glb_path = create_tactile_graphic(
            Image.open(_DATA_DIR / "dolphin.png"),
            output_path=str(tmp_path / "dolphin.glb"),
            geom_estimator=_geom_estimator(),
            seg_engine=_seg_engine(),
            standard_braille_text="the quick brown fox",
            standard_braille_plate_size=0.12,
            plate_thickness=0.004,
        )
        mesh = trimesh.load_mesh(glb_path)
        ndarrays_regression.check(
            {"vertices": np.asarray(mesh.vertices, dtype=np.float32)}
        )
