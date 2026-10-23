from src.claim_auditor import audit_claims
from src.schemas import Claim, Evidence


def _claim(claim_id=1):
    return Claim(
        claim_id=claim_id,
        text=f"Claim {claim_id}",
        page_number=1,
        source_sentence_id=claim_id,
    )


def _evidence(claim_id, evidence_id):
    return Evidence(
        evidence_id=evidence_id,
        claim_id=claim_id,
        text=f"Evidence {evidence_id}",
        page_number=2,
        similarity=0.9,
    )


def test_claim_statuses_respect_confidence_and_conflicting_evidence(monkeypatch):
    predictions = {
        "Evidence 1": ("ENTAILMENT", 0.91),
        "Evidence 2": ("CONTRADICTION", 0.92),
        "Evidence 3": ("NEUTRAL", 0.88),
        "Evidence 4": ("NEUTRAL", 0.51),
        "Evidence 5": ("ENTAILMENT", 0.90),
        "Evidence 6": ("CONTRADICTION", 0.83),
    }
    monkeypatch.setattr(
        "src.claim_auditor.predict_nli",
        lambda premise, hypothesis: predictions[premise],
    )

    claims = [_claim(index) for index in range(1, 6)]
    evidences = [
        _evidence(1, 1),
        _evidence(2, 2),
        _evidence(3, 3),
        _evidence(4, 4),
        _evidence(5, 5),
        _evidence(5, 6),
    ]

    results = audit_claims(claims, evidences)

    assert [result.status for result in results] == [
        "SUPPORTED",
        "CONTRADICTED",
        "INSUFFICIENT_EVIDENCE",
        "HUMAN_REVIEW_REQUIRED",
        "HUMAN_REVIEW_REQUIRED",
    ]
    assert "not proof" in results[0].explanation
    assert all(evidence.nli_score is not None for evidence in evidences)


def test_unavailable_nli_requires_manual_review(monkeypatch):
    monkeypatch.setattr("src.claim_auditor.predict_nli", lambda premise, hypothesis: (None, None))

    result = audit_claims([_claim()], [_evidence(1, 1)])[0]

    assert result.status == "HUMAN_REVIEW_REQUIRED"


def test_no_relevant_evidence_is_insufficient_without_nli(monkeypatch):
    monkeypatch.setattr(
        "src.claim_auditor.predict_nli",
        lambda premise, hypothesis: (_ for _ in ()).throw(AssertionError("NLI should not run")),
    )
    evidence = _evidence(1, 1)
    evidence.similarity = 0.1

    result = audit_claims([_claim()], [evidence])[0]

    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert not result.evidence


def test_mixed_neutral_and_contradiction_predictions_require_review(monkeypatch):
    predictions = {
        "Evidence 1": ("NEUTRAL", 0.99),
        "Evidence 2": ("CONTRADICTION", 0.81),
    }
    monkeypatch.setattr(
        "src.claim_auditor.predict_nli",
        lambda premise, hypothesis: predictions[premise],
    )

    results = audit_claims(
        [_claim()],
        [_evidence(1, 1), _evidence(1, 2)],
    )

    assert results[0].status == "HUMAN_REVIEW_REQUIRED"
    assert "mixed NLI labels" in results[0].explanation
