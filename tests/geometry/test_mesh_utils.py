from pathlib import Path

import numpy as np
import pytest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from tactilegen.geometry.utils import depth2mesh, pv2trimesh

_DATA_DIR = Path(__file__).parent / "data"


class TestPv2Trimesh:
    def test_converts_simple_mesh(self):
        pv = pytest.importorskip("pyvista")
        plane = pv.Plane()
        mesh = pv2trimesh(plane)
        assert mesh.vertices.shape[1] == 3
        assert mesh.faces.shape[1] == 3
        assert len(mesh.faces) > 0


class TestDepth2Mesh:
    def test_returns_polydata_with_points(self):
        depth = np.zeros((16, 16), dtype=np.float32)
        # Small bump in the center
        depth[6:10, 6:10] = 1.0
        mesh = depth2mesh(
            depth, width=0.1, max_depth=0.01, smooth=False, decimate=False
        )
        assert mesh.n_points > 0

    @pytest.mark.slow
    def test_depth2mesh(
        self,
        tmp_path: Path,
        ndarrays_regression: NDArraysRegressionFixture,
    ):
        depth = np.load(_DATA_DIR / "dolphin_depth.npy")
        mesh_pv = depth2mesh(depth, smooth=True, decimate=True)
        mesh = pv2trimesh(mesh_pv)

        mesh.export(str(tmp_path / "dolphin_mesh.glb"))
        ndarrays_regression.check(
            {"vertices": np.asarray(mesh.vertices, dtype=np.float32)}
        )
