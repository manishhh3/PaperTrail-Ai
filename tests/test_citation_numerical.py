from src.citation_analyzer import extract_citations
from src.numerical_auditor import check_numerical_consistency
from src.schemas import Claim, Evidence


def test_extract_citations_detects_and_deduplicates_markers():
    citations = extract_citations(
        "Prior work [1, 2] and [1, 2] (Smith et al., 2020) are mentioned."
    )

    assert citations == ["[1, 2]", "(Smith et al., 2020)"]


def test_numeric_check_only_flags_matching_metric_and_unit():
    claim = Claim(
        claim_id=1,
        text="Our accuracy was 95%.",
        page_number=1,
        source_sentence_id=1,
    )
    comparable = Evidence(
        evidence_id=1,
        claim_id=1,
        text="The measured accuracy was 80%.",
        page_number=2,
        similarity=0.8,
    )
    different_unit = Evidence(
        evidence_id=2,
        claim_id=1,
        text="The measured accuracy was 80 milliseconds.",
        page_number=3,
        similarity=0.7,
    )

    warnings = check_numerical_consistency([claim], [comparable, different_unit])

    assert len(warnings) == 1
    assert warnings[0][0] == 1
    assert warnings[0][1].startswith("Review:")
    assert "not automatically an error" in warnings[0][1]


def test_numeric_check_ignores_unmatched_metrics_and_small_differences():
    accuracy_claim = Claim(
        claim_id=1,
        text="The accuracy reached 95%.",
        page_number=1,
        source_sentence_id=1,
    )
    unrelated = Evidence(
        evidence_id=1,
        claim_id=1,
        text="Training loss was 2.0.",
        page_number=2,
        similarity=0.8,
    )
    close_accuracy = Evidence(
        evidence_id=2,
        claim_id=1,
        text="Measured accuracy was 90%.",
        page_number=3,
        similarity=0.7,
    )

    assert check_numerical_consistency([accuracy_claim], [unrelated, close_accuracy]) == []
