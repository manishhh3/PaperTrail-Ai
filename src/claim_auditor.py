"""Claim auditing module.

Combines extracted claims, retrieved evidence, and optional NLI predictions to assign a
conservative status to each claim.
"""

import logging
from typing import List, Dict

from .schemas import Claim, Evidence, AuditResult
from .nli_engine import predict_nli

logger = logging.getLogger(__name__)

# Confidence thresholds for interpreting model predictions.
SIMILARITY_THRESHOLD = 0.4  # minimum similarity to consider evidence relevant
ENTAILMENT_CONF = 0.6
CONTRADICTION_CONF = 0.6

def _group_evidence_by_claim(evidences: List[Evidence]) -> Dict[int, List[Evidence]]:
    grouped: Dict[int, List[Evidence]] = {}
    for ev in evidences:
        grouped.setdefault(ev.claim_id, []).append(ev)
    return grouped

def audit_claims(claims: List[Claim], evidences: List[Evidence]) -> List[AuditResult]:
    """Assign a status to each claim.

    Evidence similarity selects passages for review; it is not itself proof of support.
    Confident entailment or contradiction predictions determine a status only when the
    relevant evidence does not contain a confident prediction in the opposite direction.
    Neutral predictions indicate insufficient evidence, while unavailable or uncertain
    predictions require manual review.
    """
    audit_results: List[AuditResult] = []
    ev_by_claim = _group_evidence_by_claim(evidences)

    for claim in claims:
        relevant = [ev for ev in ev_by_claim.get(claim.claim_id, []) if ev.similarity >= SIMILARITY_THRESHOLD]
        if not relevant:
            audit_results.append(
                AuditResult(
                    claim=claim,
                    status="INSUFFICIENT_EVIDENCE",
                    evidence=[],
                    explanation="No retrieved passage reached the similarity threshold.",
                )
            )
            continue

        # Run NLI on each relevant evidence
        predictions = []
        for ev in relevant:
            label, score = predict_nli(ev.text, claim.text)
            if label is not None and score is not None:
                ev.nli_label = label
                ev.nli_score = score
                predictions.append((label, score, ev))
        if not predictions:
            audit_results.append(
                AuditResult(
                    claim=claim,
                    status="HUMAN_REVIEW_REQUIRED",
                    evidence=relevant,
                    explanation="Evidence was retrieved, but NLI predictions were unavailable; manual review is required.",
                )
            )
            continue

        observed_labels = {label for label, _, _ in predictions}
        if len(observed_labels) > 1:
            labels = ", ".join(sorted(observed_labels))
            audit_results.append(
                AuditResult(
                    claim=claim,
                    status="HUMAN_REVIEW_REQUIRED",
                    evidence=relevant,
                    explanation=(
                        "Retrieved passages produced mixed NLI labels "
                        f"({labels}); manual review is required."
                    ),
                )
            )
            continue

        confident_entailments = [
            item for item in predictions
            if item[0] == "ENTAILMENT" and item[1] >= ENTAILMENT_CONF
        ]
        confident_contradictions = [
            item for item in predictions
            if item[0] == "CONTRADICTION" and item[1] >= CONTRADICTION_CONF
        ]
        if confident_entailments and confident_contradictions:
            status = "HUMAN_REVIEW_REQUIRED"
            explanation = (
                "Retrieved passages contain confident supporting and contradicting "
                "predictions; manual review is required."
            )
        elif confident_entailments:
            _, best_score, best_ev = max(confident_entailments, key=lambda item: item[1])
            status = "SUPPORTED"
            explanation = (
                f"Evidence (ID {best_ev.evidence_id}) was classified as entailing the "
                f"claim (model confidence {best_score:.2f}); this is not proof of truth."
            )
        elif confident_contradictions:
            _, best_score, best_ev = max(confident_contradictions, key=lambda item: item[1])
            status = "CONTRADICTED"
            explanation = (
                f"Evidence (ID {best_ev.evidence_id}) was classified as contradicting the "
                f"claim (model confidence {best_score:.2f}); this is not proof of falsehood."
            )
        elif all(
            label == "NEUTRAL" and score >= ENTAILMENT_CONF
            for label, score, _ in predictions
        ):
            status = "INSUFFICIENT_EVIDENCE"
            explanation = (
                "The retrieved passages did not establish support or contradiction "
                "according to the NLI model."
            )
        else:
            status = "HUMAN_REVIEW_REQUIRED"
            best_label, best_score, _ = max(predictions, key=lambda item: item[1])
            explanation = (
                f"NLI predictions were uncertain (strongest label={best_label}, "
                f"model confidence {best_score:.2f}); manual review is required."
            )

        audit_results.append(
            AuditResult(
                claim=claim,
                status=status,
                evidence=relevant,
                explanation=explanation,
            )
        )
    return audit_results
