"""Runtime resolution of API keys and tokens."""

import getpass
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class SecretSpec:
    label: str  # canonical env var name shown to the user
    env_vars: tuple[str, ...]  # all env vars this secret may be read from / cached into
    instructions: str


_GEMINI = SecretSpec(
    label="GEMINI_API_KEY",
    env_vars=("GEMINI_API_KEY", "GENAI_API_KEY"),
    instructions=(
        "Get one at https://aistudio.google.com/apikey, then export it in your "
        "shell (e.g., add `export GEMINI_API_KEY=...` to ~/.bashrc)."
    ),
)

_HF = SecretSpec(
    label="HF_TOKEN",
    env_vars=("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    instructions=(
        "Create one at https://huggingface.co/settings/tokens, then export it "
        "in your shell (e.g., add `export HF_TOKEN=...` to ~/.bashrc)."
    ),
)


def _first_env(names: tuple[str, ...]) -> str | None:
    """Return the first non-empty value among the given env-var names."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def ensure_runtime_secrets(*, gemini: bool = False, hf: bool = True) -> None:
    if hf:
        get_hf_token()
    if gemini:
        get_gemini_api_key()


def get_hf_token(*, required: bool = True) -> str | None:
    """Return the HuggingFace Hub token (env or prompt)."""
    return resolve_secret(_HF, required=required)


def get_gemini_api_key(*, required: bool = True) -> str | None:
    """Return the Gemini / GenAI API key (env or prompt)."""
    return resolve_secret(_GEMINI, required=required)


def resolve_secret(spec: SecretSpec, *, required: bool) -> str | None:
    """Resolve ``spec`` from env (preferred) or by prompting the TTY.

    Behavior:
      * Found in env → cache the value into any of ``spec.env_vars`` that aren't
        already set (preserves caller-provided distinct values).
      * Missing and ``required=False`` → return None.
      * Missing and ``required=True`` → prompt; on success, set all env vars
        to the prompted value (overwriting) so downstream libraries see it.
    """
    value = _first_env(spec.env_vars)
    if value is not None:
        for name in spec.env_vars:
            os.environ.setdefault(name, value)
        return value

    if not required:
        return None

    value = _prompt_for_secret(spec.label, spec.instructions)
    for name in spec.env_vars:
        os.environ[name] = value
    return value


def _prompt_for_secret(label: str, instructions: str) -> str:
    """Prompt for a secret on the TTY; raise if non-interactive or empty."""
    if not sys.stdin.isatty():
        raise RuntimeError(f"{label} not set and stdin is not a TTY.")

    print(f"\n{label} is not set. {instructions}")
    value = getpass.getpass(f"Enter {label} (input hidden): ").strip()
    if not value:
        raise RuntimeError(f"{label} prompt was empty; aborting.")
    return value
