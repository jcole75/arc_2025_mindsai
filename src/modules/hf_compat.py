"""Compatibility utilities for Hugging Face models used in ARC solution."""

from __future__ import annotations

import logging


__all__ = ["apply_t5gemma_config_patch"]

logger = logging.getLogger(__name__)
_PATCHED_FLAG = "_arc_num_hidden_layers_patched"


def apply_t5gemma_config_patch(source: str | None = None) -> bool:
    """Ensure T5Gemma configs expose ``num_hidden_layers`` for newer HF releases."""
    try:
        from transformers.models.t5gemma.configuration_t5gemma import T5GemmaConfig  # type: ignore
    except ImportError:
        if source:
            logger.debug("T5GemmaConfig not available; skipping patch (%s)", source)
        else:
            logger.debug("T5GemmaConfig not available; skipping patch")
        return False

    if getattr(T5GemmaConfig, _PATCHED_FLAG, False):
        return False

    original_getattribute = T5GemmaConfig.__getattribute__

    def _patched_getattribute(self, key):
        if key == "num_hidden_layers":
            try:
                decoder = original_getattribute(self, "decoder")
            except AttributeError:
                decoder = None
            except Exception:
                decoder = None
            if decoder is not None and hasattr(decoder, "num_hidden_layers"):
                return decoder.num_hidden_layers
        return original_getattribute(self, key)

    T5GemmaConfig.__getattribute__ = _patched_getattribute  # type: ignore[assignment]
    setattr(T5GemmaConfig, _PATCHED_FLAG, True)

    if source:
        logger.info("Applied T5GemmaConfig num_hidden_layers compatibility patch (%s)", source)
    else:
        logger.info("Applied T5GemmaConfig num_hidden_layers compatibility patch")
    return True
