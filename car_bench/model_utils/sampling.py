"""Provider-specific sampling settings for benchmark-controlled LLM roles."""

from __future__ import annotations

from typing import Any


def evaluation_sampling_parameters(model: str, provider: str) -> dict[str, Any]:
    """Use provider defaults for Gemini 3.x and the legacy fixed setting otherwise."""

    normalized_model = model.removeprefix("gemini/")
    if provider.lower() == "gemini" and normalized_model.startswith("gemini-3"):
        return {}
    return {"temperature": 0.0}
