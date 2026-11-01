"""Load the optional Together API key used only for prior regeneration."""

from __future__ import annotations

import os
from pathlib import Path


def load_api_key() -> str:
    """Read ``TOGETHER_API_KEY`` or the local, git-ignored ``apikey.txt``."""
    key = os.environ.get("TOGETHER_API_KEY", "").strip()
    if key:
        return key

    key_file = Path(__file__).resolve().parent / "apikey.txt"
    if key_file.is_file():
        key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(
            "No Together API key was found. Set TOGETHER_API_KEY or place the key "
            "in prior_generation/apikey.txt. The key is unnecessary when using the "
            "pre-generated priors supplied with the paper."
        )
    return key
