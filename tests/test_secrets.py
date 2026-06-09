import os

import pytest

from text2tactilegraphics.secrets_ import SecretSpec, resolve_secret

_KEY = "MY_KEY"
_KEY_ALIAS = "MY_KEY_ALIAS"


def _spec_for(*aliases) -> SecretSpec:
    return SecretSpec(
        label=aliases[0],
        env_vars=tuple(aliases),
        instructions="(test only — no real instructions)",
    )


@pytest.fixture
def spec():
    return _spec_for(_KEY, _KEY_ALIAS)


@pytest.fixture(autouse=True)
def _clean_env(spec: SecretSpec, monkeypatch):
    """Snapshot+restore the process env around each test."""
    snapshot = dict(os.environ)
    for name in spec.env_vars:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


# =============================================================================
# resolve_secret — happy paths reading from env
# =============================================================================


class TestResolveSecretFromEnv:
    def test_picks_up_first_env(self, spec: SecretSpec, monkeypatch):
        monkeypatch.setenv(_KEY, "value-1")
        assert resolve_secret(spec, required=True) == "value-1"

    def test_caches_into_missing_aliases(self, spec: SecretSpec, monkeypatch):
        """If only one alias is set, the others get `setdefault`-ed."""
        monkeypatch.setenv(_KEY, "value-1")
        resolve_secret(spec, required=True)
        assert os.environ[_KEY] == "value-1"
        assert os.environ[_KEY_ALIAS] == "value-1"

    def test_preserves_distinct_pre_set_aliases(self, spec: SecretSpec, monkeypatch):
        """`setdefault` must NOT overwrite an alias the caller deliberately set."""
        monkeypatch.setenv(_KEY, "primary")
        monkeypatch.setenv(_KEY_ALIAS, "secondary")
        out = resolve_secret(spec, required=True)
        # `_first_env` returned the first hit ("primary").
        assert out == "primary"
        # But the explicitly-set alias is NOT overwritten.
        assert os.environ[_KEY] == "primary"
        assert os.environ[_KEY_ALIAS] == "secondary"

    def test_picks_up_second_alias_when_first_missing(
        self, spec: SecretSpec, monkeypatch
    ):
        monkeypatch.setenv(_KEY_ALIAS, "from-alias")
        out = resolve_secret(spec, required=True)
        assert out == "from-alias"
        assert os.environ[_KEY] == "from-alias"  # backfilled
        assert os.environ[_KEY_ALIAS] == "from-alias"


# =============================================================================
# resolve_secret — not-set paths
# =============================================================================


class TestResolveSecretMissing:
    def test_required_false_returns_none(self, spec: SecretSpec):
        assert resolve_secret(spec, required=False) is None

    def test_required_false_does_not_touch_env(self, spec: SecretSpec):
        resolve_secret(spec, required=False)
        assert _KEY not in os.environ
        assert _KEY_ALIAS not in os.environ

    def test_required_true_non_tty_raises(self, spec: SecretSpec, monkeypatch):
        """Non-interactive caller must have the env set; we raise instead of hanging."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with pytest.raises(RuntimeError, match="not a TTY"):
            resolve_secret(spec, required=True)


# =============================================================================
# resolve_secret — interactive prompt path
# =============================================================================


class TestResolveSecretPrompt:
    def test_prompts_then_caches_into_all_aliases(self, spec: SecretSpec, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt: "prompted-value")
        out = resolve_secret(spec, required=True)
        assert out == "prompted-value"
        # Both aliases should hold the prompted value (explicit assign).
        assert os.environ[_KEY] == "prompted-value"
        assert os.environ[_KEY_ALIAS] == "prompted-value"

    def test_empty_prompt_raises(self, spec: SecretSpec, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt: "")
        with pytest.raises(RuntimeError, match="prompt was empty"):
            resolve_secret(spec, required=True)

    def test_whitespace_only_prompt_raises(self, spec: SecretSpec, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt: "   \n\t  ")
        with pytest.raises(RuntimeError, match="prompt was empty"):
            resolve_secret(spec, required=True)

    def test_prompt_value_is_stripped(self, spec: SecretSpec, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt: "  padded  \n")
        assert resolve_secret(spec, required=True) == "padded"
