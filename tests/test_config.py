import pytest

from text2tactilegraphics.config import BASE_IMAGE_STEPS, INFERENCE_STEPS, Config


@pytest.fixture
def cfg():
    return Config()


class TestConfig:
    def test_get_device_returns_string(self, cfg: Config):
        device = cfg.get_device("qwen_base_edit")
        assert isinstance(device, str)
        assert device.startswith("cuda:") or device == "cpu"

    def test_get_device_for_unknown_model_defaults_to_zero(self, cfg: Config):
        dev = cfg.get_device("nonexistent_model")
        assert dev.endswith("0") or dev == "cpu"

    def test_get_texture_config_valid(self, cfg: Config):
        assert set(INFERENCE_STEPS) == {4, 40}
        for steps in INFERENCE_STEPS:
            texture_cfg = cfg.get_texture_config(steps)
            assert "distill_lora" in texture_cfg
            assert "cfg_scale" in texture_cfg

    def test_get_texture_config_invalid(self, cfg: Config):
        with pytest.raises(ValueError):
            cfg.get_texture_config(99999)

    def test_get_texture_config_returns_copy(self, cfg: Config):
        texture_cfg = cfg.get_texture_config(4)
        texture_cfg["cfg_scale"] = 999.0
        # Should not pollute the global dict
        assert INFERENCE_STEPS[4]["cfg_scale"] != 999.0

    def test_get_base_image_config_valid(self, cfg: Config):
        for steps in BASE_IMAGE_STEPS:
            texture_cfg = cfg.get_base_image_config(steps)
            assert "edit_lora" in texture_cfg
            assert "cfg_scale" in texture_cfg

    def test_get_base_image_config_invalid(self, cfg: Config):
        with pytest.raises(ValueError):
            cfg.get_base_image_config(99999)

    def test_get_qwen_vram_config_replaces_cuda(self, cfg: Config):
        texture_cfg = cfg.get_qwen_vram_config("cuda:3")
        # Any device key that was "cuda" should now be the passed device string
        for key in (
            "offload_device",
            "onload_device",
            "preparing_device",
            "computation_device",
        ):
            assert texture_cfg[key] in ("cuda:3", "cpu")  # depends on vram_mode
