"""Unit tests for `text2tactilegraphics.ui.app`. These tests do NOT launch a Gradio server."""

import gradio as gr
import pytest

from text2tactilegraphics.ui.app import create_demo

# =============================================================================
# create_demo smoke
# =============================================================================


class TestCreateDemo:
    @pytest.fixture(scope="class")
    def demo(self) -> gr.Blocks:
        return create_demo()

    def test_title_set(self, demo: gr.Blocks):
        assert "tactile" in demo.title.lower()

    def test_has_components(self, demo: gr.Blocks):
        # Sanity: the demo should assemble many components without errors. The
        # exact number isn't pinned (it changes as the UI evolves), just that
        # we ended up with a non-trivial graph.
        assert len(demo.blocks) > 100


class TestDebugPanelGate:
    """The Debug-settings panel is gated by the `TEXT2TACTILEGRAPHICS_DEBUG` env var."""

    @staticmethod
    def _has_debug_panel(demo: gr.Blocks) -> bool:
        return any(
            isinstance(b, gr.Accordion) and b.label == "Debug settings"
            for b in demo.blocks.values()
        )

    def test_hidden_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TEXT2TACTILEGRAPHICS_DEBUG", raising=False)
        assert self._has_debug_panel(create_demo()) is False

    def test_shown_when_enabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEXT2TACTILEGRAPHICS_DEBUG", "1")
        assert self._has_debug_panel(create_demo()) is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", ""])
    def test_falsy_values_keep_it_hidden(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ):
        monkeypatch.setenv("TEXT2TACTILEGRAPHICS_DEBUG", value)
        assert self._has_debug_panel(create_demo()) is False


# =============================================================================
# Stage 1
# =============================================================================


class TestGenerationModelRadio:
    """The Stage 1 'Generation model' radio is gated on the presence of a
    Gemini API key. When no key is available, Nano Banana Pro can't run, so
    the picker is replaced by a non-rendering `gr.State("qwen_edit")` carrier.
    """

    @staticmethod
    def _find_model_radio(demo: gr.Blocks) -> gr.Radio | None:
        for b in demo.blocks.values():
            if isinstance(b, gr.Radio) and b.label == "Generation model":
                return b
        return None

    @staticmethod
    def _clear_keys(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GENAI_API_KEY", raising=False)

    def test_radio_shown_when_gemini_key_set(self, monkeypatch: pytest.MonkeyPatch):
        self._clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
        radio = self._find_model_radio(create_demo())
        assert radio is not None, "expected Generation-model radio to be present"
        values = [v for _, v in radio.choices]
        assert "qwen_edit" in values
        assert "nano_banana_pro" in values

    def test_radio_shown_when_genai_alias_set(self, monkeypatch: pytest.MonkeyPatch):
        """`GENAI_API_KEY` is an accepted alias for `GEMINI_API_KEY`."""
        self._clear_keys(monkeypatch)
        monkeypatch.setenv("GENAI_API_KEY", "fake-key-for-test")
        assert self._find_model_radio(create_demo()) is not None

    def test_radio_hidden_when_no_key(self, monkeypatch: pytest.MonkeyPatch):
        self._clear_keys(monkeypatch)
        assert self._find_model_radio(create_demo()) is None

    def test_empty_key_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch):
        """An empty string in the env var doesn't count as a valid key."""
        self._clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "")
        assert self._find_model_radio(create_demo()) is None
