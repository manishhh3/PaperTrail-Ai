"""Exercise the pipeline on a clearly labeled, synthetic demonstration PDF."""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.pdf_parser import extract_pages
from src.preprocessing import preprocess
from src.claim_extractor import extract_claims
from src.evidence_retriever import retrieve_evidence
from src.claim_auditor import audit_claims
from src.report_generator import export_to_csv, export_to_pdf


def build_demo_pdf(pdf_path: Path) -> None:
    if pdf_path.exists():
        return

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    lines = [
        "We propose a new algorithm that achieves 95% accuracy on the benchmark.",
        "In our experiments, we observe that the loss decreases rapidly across epochs.",
        "However, the reported results contradict prior work showing only 80% accuracy.",
    ]
    y = 720
    for line in lines:
        c.drawString(72, y, line)
        y -= 24
    c.save()
    print(f"Created demo PDF at {pdf_path}")


if __name__ == "__main__":
    pdf_path = Path("demo.pdf")
    build_demo_pdf(pdf_path)

    print("Synthetic demonstration only; this is not a published research paper.")
    pages = extract_pages(str(pdf_path))
    print("Pages extracted:", pages)

    sentences = preprocess(pages)
    print("Sentences:", [s.text for s in sentences])

    claims = extract_claims(sentences)
    print("Claims:", [c.text for c in claims])

    evidences = retrieve_evidence(claims, sentences)
    print("Evidence retrieved:", [(e.claim_id, e.text[:30], e.similarity) for e in evidences])

    audit_results = audit_claims(claims, evidences)
    print("Audit results:")
    for r in audit_results:
        print(r.claim.claim_id, r.status, r.explanation)

    csv_path = export_to_csv(audit_results, "demo_report.csv")
    pdf_path_out = export_to_pdf(audit_results, "demo_report.pdf")
    print("Reports written:", csv_path, pdf_path_out)
