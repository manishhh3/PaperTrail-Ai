"""Utility functions for PaperTrail AI.

Provides:
- simple logger configuration
- safe model loading with caching
- a tiny helper for sanitizing text
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a basic format if not already configured."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

@lru_cache(maxsize=1)
def load_model(model_name: str) -> Any:
    """Load and cache a Hugging Face sequence-classification model.

    Failures raise an actionable error so the caller can report the unavailable model
    instead of silently treating a failed load as a successful empty result.
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=True,
        )
        logger.info(f"Model '{model_name}' loaded successfully.")
        return model, tokenizer
    except Exception as exc:
        logger.exception("Failed to load Hugging Face model '%s'.", model_name)
        raise RuntimeError(
            f"Could not load model '{model_name}'. Check your internet connection "
            "for the first download, available disk space, and the local Hugging Face "
            f"cache. Original error: {exc}"
        ) from exc

def safe_read_text(file_path: str) -> str:
    """Read a text file safely, returning an empty string on error."""
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return ""
