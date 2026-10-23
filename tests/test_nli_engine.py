import torch
import pytest
import transformers

from src import nli_engine
from src.utils import load_model


class _Tokenizer:
    model_max_length = 1024

    def __call__(self, premise, hypothesis, **kwargs):
        assert premise and hypothesis
        assert kwargs["truncation"] is True
        return {}


class _Model:
    config = type(
        "Config",
        (),
        {"id2label": {"0": "contradiction", "1": "neutral", "2": "entailment"}},
    )()

    def eval(self):
        return self

    def __call__(self, **inputs):
        return type("Output", (), {"logits": torch.tensor([[0.0, 0.0, 5.0]])})()


def test_predict_nli_uses_model_label_configuration(monkeypatch):
    monkeypatch.setattr(nli_engine, "_nli_cache", (_Model(), _Tokenizer()))

    label, score = nli_engine.predict_nli("Evidence supports the claim.", "The claim is supported.")

    assert label == "ENTAILMENT"
    assert score is not None and score > 0.9


def test_predict_nli_does_not_run_on_empty_text(monkeypatch):
    monkeypatch.setattr(
        nli_engine,
        "_ensure_model",
        lambda: (_ for _ in ()).throw(AssertionError("Model should not load")),
    )

    assert nli_engine.predict_nli(" ", "claim") == (None, None)


def test_model_load_failure_is_exposed_and_predictions_remain_unavailable(monkeypatch):
    monkeypatch.setattr(nli_engine, "_nli_cache", None)
    monkeypatch.setattr(nli_engine, "_nli_load_attempted", False)
    monkeypatch.setattr(nli_engine, "_nli_load_error", None)
    monkeypatch.setattr(
        nli_engine,
        "_load_nli",
        lambda: (_ for _ in ()).throw(RuntimeError("check model download")),
    )

    assert nli_engine.predict_nli("evidence", "claim") == (None, None)
    assert nli_engine.get_nli_model_error() == "check model download"


def test_model_loader_raises_actionable_error_when_weights_are_unavailable(monkeypatch):
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda model_name: object(),
    )

    def fail_loading_weights(model_name, **kwargs):
        raise OSError("offline test failure")

    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        fail_loading_weights,
    )
    load_model.cache_clear()

    with pytest.raises(
        RuntimeError,
        match="Check your internet connection.*Original error: offline test failure",
    ):
        load_model("test-model")
