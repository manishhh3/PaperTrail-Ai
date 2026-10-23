"""NLI engine for PaperTrail AI.

Provides a wrapper around a Hugging Face NLI model. If loading or inference fails,
predictions are unavailable and the caller must treat the result as requiring review.
"""

import logging
from typing import Optional, Tuple

from .utils import load_model

logger = logging.getLogger(__name__)

# Default model name – a small, zero‑shot NLI model that works on CPU.
DEFAULT_NLI_MODEL = "facebook/bart-large-mnli"

def _load_nli() -> Tuple[object, object]:
    """Load the NLI model and tokenizer, cached via utils.load_model.

    Returns ``(model, tokenizer)`` or raises an actionable loading error.
    """
    return load_model(DEFAULT_NLI_MODEL)

_nli_cache = None
_nli_load_attempted = False
_nli_load_error: Optional[str] = None

def _ensure_model():
    global _nli_cache, _nli_load_attempted, _nli_load_error
    if _nli_cache is None and not _nli_load_attempted:
        _nli_load_attempted = True
        try:
            _nli_cache = _load_nli()
            _nli_load_error = None
        except RuntimeError as exc:
            _nli_load_error = str(exc)
            logger.error("NLI model unavailable: %s", exc)
    return _nli_cache

def get_nli_model_error() -> Optional[str]:
    """Return the model loading error, if the one-time load attempt failed."""
    return _nli_load_error

def predict_nli(premise: str, hypothesis: str) -> Tuple[Optional[str], Optional[float]]:
    """Run NLI inference.

    Parameters
    ----------
    premise: str
        Evidence passage (treated as the premise).
    hypothesis: str
        Claim (treated as the hypothesis).

    Returns
    -------
    tuple(label, score)
        ``label`` is one of ``ENTAILMENT``, ``CONTRADICTION``, ``NEUTRAL`` (uppercase) or
        ``None`` if the model is unavailable. ``score`` is the confidence (probability) of
        the chosen label, also ``None`` on failure.
    """
    if not premise.strip() or not hypothesis.strip():
        return None, None

    model_tokenizer = _ensure_model()
    if model_tokenizer is None:
        logger.warning(
            "NLI model unavailable; returning None predictions. %s",
            _nli_load_error or "",
        )
        return None, None
    model, tokenizer = model_tokenizer
    try:
        import torch

        model.eval()
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
            if not torch.isfinite(logits).all():
                logger.error("NLI model returned non-finite logits.")
                return None, None
            probabilities = torch.softmax(logits, dim=-1)
        best_idx = int(probabilities.argmax().item())
        id2label = getattr(model.config, "id2label", {})
        raw_label = id2label.get(best_idx, id2label.get(str(best_idx)))
        if raw_label is None:
            logger.error("NLI model config does not define label index %s.", best_idx)
            return None, None
        label_text = str(raw_label).upper()
        label = next(
            (
                known
                for known in ("ENTAILMENT", "CONTRADICTION", "NEUTRAL")
                if known in label_text
            ),
            None,
        )
        if label is None:
            logger.error("Unrecognized NLI model label: %s", raw_label)
            return None, None
        score = float(probabilities[best_idx].item())
        if not 0.0 <= score <= 1.0:
            logger.error("NLI confidence is outside the probability range: %s", score)
            return None, None
        return label, score
    except Exception:
        logger.exception("NLI inference failed.")
        return None, None
