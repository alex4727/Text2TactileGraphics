import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from text2tactilegraphics.config import Config
from text2tactilegraphics.generation.models import LoraManager, ModelManager

# =============================================================================
# Test ModelManager
# =============================================================================


@pytest.fixture
def mm():
    m = ModelManager()
    yield m
    m.unload_all_models()


class TestModelManager:
    @pytest.mark.slow
    def test_qwen_base_edit(self, mm: ModelManager):
        _ = mm.qwen_base_edit

    @pytest.mark.slow
    def test_qwen_texture(self, mm: ModelManager):
        _ = mm.qwen_texture

    @pytest.mark.slow
    def test_qwen_tiling(self, mm: ModelManager):
        _ = mm.qwen_tiling

    @pytest.mark.slow
    def test_sdxl_tiling(self, mm: ModelManager):
        _ = mm.sdxl_tiling

    @pytest.mark.slow
    def test_sam3_text(self, mm: ModelManager):
        _ = mm.sam3_text

    @pytest.mark.slow
    def test_sam3_tracker(self, mm: ModelManager):
        _ = mm.sam3_tracker

    @pytest.mark.slow
    def test_moge2(self, mm: ModelManager):
        _ = mm.moge2

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GENAI_API_KEY"),
        reason="Gemini API key not set",
    )
    @pytest.mark.slow
    def test_genai(self, mm: ModelManager):
        _ = mm.genai

    @pytest.mark.slow
    def test_cached_property_returns_same_object(self, mm: ModelManager):
        first = mm.moge2
        second = mm.moge2
        assert first is second

    @pytest.mark.slow
    def test_unload_model_clears_cached_property(self, mm: ModelManager):
        first = mm.moge2
        mm.unload_model("moge2")
        assert "moge2" not in mm.__dict__
        second = mm.moge2  # triggers a fresh load
        assert first is not second

    def test_unload_unknown_name_raises(self, mm: ModelManager):
        with pytest.raises(ValueError, match="not a recognized model"):
            mm.unload_model("nothing_loaded_with_this_name")

    def test_unload_non_model_attribute_raises(self, mm: ModelManager):
        with pytest.raises(ValueError, match="not a recognized model"):
            mm.unload_model("config")

    def test_unload_valid_name_when_not_loaded_is_noop(self, mm: ModelManager):
        assert "moge2" not in mm.__dict__
        mm.unload_model("moge2")
        assert "moge2" not in mm.__dict__

    @pytest.mark.slow
    def test_unload_all_models_clears_every_loaded_model(self, mm: ModelManager):
        _ = mm.moge2
        _ = mm.sam3_text
        assert {"moge2", "sam3_text"} <= set(mm.__dict__)

        mm.unload_all_models()

        assert not (mm._model_names() & set(mm.__dict__))
        assert "config" in mm.__dict__

    def test_unload_all_models_when_nothing_loaded_is_noop(self, mm: ModelManager):
        assert not (mm._model_names() & set(mm.__dict__))
        mm.unload_all_models()


# =============================================================================
# Test LoraManager
# =============================================================================


def _make_lora_manager(
    ckpt_paths: dict[str, str] | None = None, pipeline: Mock | None = None
) -> LoraManager:
    config = Config()
    config.ckpt_paths = ckpt_paths or {}
    return LoraManager(pipeline=pipeline or _make_fake_pipeline(), config=config)


def _loaded_paths(pipe: Mock) -> list[str]:
    """Paths passed to `pipe.load_lora(target, path)`, in call order."""
    return [c.args[1] for c in pipe.load_lora.call_args_list]


def _make_fake_pipeline() -> Mock:
    pipe = Mock(spec=["load_lora", "clear_lora", "dit"])
    pipe.dit = object()
    return pipe


def _ckpt_map(tmp_path: Path, *keys: str) -> dict[str, str]:
    """Create empty `<key>.safetensors` files under `tmp_path` and return a
    `{key: path}` map suitable for stubbing `config.ckpt_paths`."""
    out: dict[str, str] = {}
    for key in keys:
        p = tmp_path / f"{key}.safetensors"
        p.write_bytes(b"")
        out[key] = str(p)
    return out


class TestLoraManager:
    def test_resolve_returns_path_when_exists(self, tmp_path: Path):
        p = tmp_path / "lora.safetensors"
        p.write_bytes(b"")  # exists, even if empty
        m = _make_lora_manager(ckpt_paths={"my_lora": str(p)})
        assert m.resolve("my_lora") == str(p)

    def test_resolve_missing_raises(self, tmp_path: Path):
        m = _make_lora_manager(ckpt_paths={"my_lora": str(tmp_path / "nope.bin")})
        with pytest.raises(FileNotFoundError, match="my_lora"):
            m.resolve("my_lora")

    def test_resolve_unregistered_key_raises(self):
        m = _make_lora_manager()
        with pytest.raises(KeyError, match="missing_lora"):
            m.resolve("missing_lora")

    def test_resolve_none_key(self):
        assert _make_lora_manager().resolve(None) is None
        assert _make_lora_manager().resolve("") is None

    def test_apply_loads_in_order(self, tmp_path: Path):
        ckpts = _ckpt_map(tmp_path, "a", "b")
        pipe = _make_fake_pipeline()
        m = _make_lora_manager(ckpt_paths=ckpts, pipeline=pipe)
        m.apply(["a", "b"])
        assert _loaded_paths(pipe) == [ckpts["a"], ckpts["b"]]
        assert m.loaded_paths == (ckpts["a"], ckpts["b"])
        pipe.clear_lora.assert_not_called()  # nothing was loaded before

    def test_apply_skips_none(self, tmp_path: Path):
        ckpts = _ckpt_map(tmp_path, "a")
        pipe = _make_fake_pipeline()
        m = _make_lora_manager(ckpt_paths=ckpts, pipeline=pipe)
        m.apply(["a", None])
        assert _loaded_paths(pipe) == [ckpts["a"]]
        assert m.loaded_paths == (ckpts["a"],)

    def test_apply_noop_if_same(self, tmp_path: Path):
        ckpts = _ckpt_map(tmp_path, "x")
        pipe = _make_fake_pipeline()
        m = _make_lora_manager(ckpt_paths=ckpts, pipeline=pipe)
        m.apply(["x"])
        pipe.load_lora.reset_mock()
        # Second call with identical desired set should not reload.
        m.apply(["x"])
        pipe.load_lora.assert_not_called()
        pipe.clear_lora.assert_not_called()

    def test_apply_reloads_on_change(self, tmp_path: Path):
        ckpts = _ckpt_map(tmp_path, "x", "y")
        pipe = _make_fake_pipeline()
        m = _make_lora_manager(ckpt_paths=ckpts, pipeline=pipe)
        m.apply(["x"])
        m.apply(["y"])
        pipe.clear_lora.assert_called_once()
        assert _loaded_paths(pipe) == [ckpts["x"], ckpts["y"]]
        assert m.loaded_paths == (ckpts["y"],)

    def test_apply_to_empty_clears(self, tmp_path: Path):
        ckpts = _ckpt_map(tmp_path, "x")
        pipe = _make_fake_pipeline()
        m = _make_lora_manager(ckpt_paths=ckpts, pipeline=pipe)
        m.apply(["x"])
        m.apply([None])
        pipe.clear_lora.assert_called_once()
        assert m.loaded_paths == ()
