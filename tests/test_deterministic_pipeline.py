import json
from pathlib import Path

from src.claim_auditor import audit_claims
from src.claim_extractor import extract_claims
from src.evidence_retriever import retrieve_evidence
from src.preprocessing import preprocess


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "deterministic_audit.json"


def test_synthetic_fixture_runs_pipeline_with_explicit_nli_stub(monkeypatch):
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr("src.evidence_retriever._load_embedding_model", lambda: None)

    sentences = preprocess([(1, fixture["page_text"])])
    claims = extract_claims(sentences)
    evidences = retrieve_evidence(claims, sentences)

    def deterministic_nli(premise, hypothesis):
        if "accuracy" in hypothesis.lower():
            return "ENTAILMENT", 0.95
        if "error reduction" in hypothesis.lower():
            return "CONTRADICTION", 0.92
        if "latency improvement" in hypothesis.lower():
            return "NEUTRAL", 0.9
        return "NEUTRAL", 0.52

    monkeypatch.setattr("src.claim_auditor.predict_nli", deterministic_nli)
    results = audit_claims(claims, evidences)
    results_by_key = {
        key: next(
            result
            for result in results
            if key in result.claim.text.lower()
        ).status
        for key in fixture["expected_statuses"]
    }

    assert results_by_key == fixture["expected_statuses"]
    assert all(
        evidence.text != result.claim.text
        for result in results
        for evidence in result.evidence
    )
