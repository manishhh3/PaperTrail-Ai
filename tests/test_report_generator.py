import csv

import fitz

from src.report_generator import export_to_csv, export_to_pdf
from src.schemas import AuditResult, Claim, Evidence


def test_csv_and_pdf_exports_contain_auditable_content(tmp_path):
    claim = Claim(
        claim_id=1,
        text="Accuracy is < 95% & measured on test set.",
        page_number=1,
        source_sentence_id=1,
    )
    evidence = Evidence(
        evidence_id=1,
        claim_id=1,
        text="The measured accuracy was 90% & independently reported.",
        page_number=2,
        similarity=0.8,
        nli_label="ENTAILMENT",
        nli_score=0.85,
    )
    result = AuditResult(
        claim=claim,
        status="SUPPORTED",
        evidence=[evidence],
        explanation="Evidence supports the claim; review remains appropriate.",
    )
    csv_path = tmp_path / "report.csv"
    pdf_path = tmp_path / "report.pdf"

    export_to_csv([result], str(csv_path))
    export_to_pdf([result], str(pdf_path))

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]["claim_text"] == claim.text
    assert rows[0]["evidence_text"] == evidence.text
    assert rows[0]["nli_label"] == "ENTAILMENT"

    document = fitz.open(pdf_path)
    pdf_text = " ".join(
        " ".join(page.get_text().split()) for page in document
    )
    document.close()
    assert "PaperTrail AI Audit Report" in pdf_text
    assert "Accuracy is < 95% & measured on test set." in pdf_text
    assert "The measured accuracy was 90% & independently reported." in pdf_text
