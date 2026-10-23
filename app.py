"""PaperTrail AI — premium research-audit Streamlit application."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from src.pdf_parser import OCRUnavailableError, extract_pages
from src.preprocessing import preprocess
from src.claim_extractor import extract_claims
from src.evidence_retriever import retrieve_evidence
from src.claim_auditor import audit_claims
from src.report_generator import export_to_csv, export_to_pdf
from src.numerical_auditor import check_numerical_consistency
from src.citation_analyzer import extract_citations
from src.nli_engine import get_nli_model_error

APP_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = APP_ROOT / "uploaded_pdfs"
RECENT_AUDITS_PATH = APP_ROOT / ".papertrail_recent.json"
logger = logging.getLogger(__name__)


def _dark_theme() -> None:
    st.set_page_config(layout="wide", page_title="PaperTrail AI", page_icon="🧪")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [data-testid="stAppViewContainer"], .stApp {
            font-family: 'Inter', sans-serif;
            background:
                radial-gradient(circle at top left, rgba(139, 92, 246, 0.22), transparent 28%),
                radial-gradient(circle at bottom right, rgba(34, 211, 238, 0.14), transparent 30%),
                linear-gradient(180deg, #060d17 0%, #0c1524 30%, #0a111c 100%);
            color: #edf6ff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .glass-card {
            background: rgba(15, 23, 35, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            box-shadow: 0 24px 60px rgba(2, 6, 23, 0.5), inset 0 1px 0 rgba(255,255,255,0.08);
            padding: 1.2rem 1.3rem;
            transform: perspective(1200px) rotateX(0deg);
        }
        .hero-panel {
            background: linear-gradient(135deg, rgba(91, 80, 190, 0.25), rgba(18, 28, 52, 0.85));
            border: 1px solid rgba(139, 92, 246, 0.42);
            border-radius: 24px;
            box-shadow: 0 30px 70px rgba(91, 80, 190, 0.14);
            padding: 1.5rem;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(11, 18, 31, 0.85), rgba(18, 27, 42, 0.78));
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 18px 36px rgba(15,23,42,.28);
        }
        [data-testid="stMetricValue"] {
            color: #f7fbff;
            font-size: 1.7rem;
            font-weight: 700;
        }
        [data-testid="stMetricLabel"] {
            color: #9db2cc;
        }
        [data-testid="stSidebar"] {
            background: rgba(8, 13, 22, 0.92);
            border-right: 1px solid rgba(148,163,184,0.18);
        }
        .stButton > button {
            background: linear-gradient(90deg, #8b5cf6 0%, #3b82f6 100%);
            color: white;
            border: none;
            border-radius: 12px;
            box-shadow: 0 18px 28px rgba(59,130,246,0.18);
            font-weight: 700;
            padding: 0.7rem 1rem;
        }
        .stDownloadButton > button {
            background: rgba(15, 23, 42, 0.9);
            color: white;
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 12px;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(15, 23, 35, 0.8);
            border: 1px dashed rgba(59,130,246,0.5);
            border-radius: 18px;
        }
        .stDataFrame {
            border: 1px solid rgba(148,163,184,0.12);
            border-radius: 14px;
            overflow: hidden;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(15,23,35,0.78);
            border: 1px solid rgba(148,163,184,0.12);
            border-radius: 12px;
            color: #dfe9f5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_storage() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True, parents=True)


def _save_uploaded_pdf(file_name: str, content: bytes) -> Path:
    """Save an upload without replacing existing files; reuse identical session uploads."""
    _ensure_storage()
    digest = hashlib.sha256(content).hexdigest()
    previous = st.session_state.get("_uploaded_pdf")
    if isinstance(previous, dict) and previous.get("sha256") == digest:
        previous_path = previous.get("path")
        if isinstance(previous_path, str):
            saved_path = Path(previous_path)
            if saved_path.parent == UPLOAD_DIR and saved_path.is_file():
                return saved_path

    original_name = Path(file_name).name or "uploaded.pdf"
    suffix = Path(original_name).suffix or ".pdf"
    stem = Path(original_name).stem or "uploaded"
    for index in range(10_000):
        candidate_name = (
            original_name if index == 0 else f"{stem} ({index}){suffix}"
        )
        target = UPLOAD_DIR / candidate_name
        try:
            with target.open("xb") as output:
                output.write(content)
        except FileExistsError:
            continue
        st.session_state["_uploaded_pdf"] = {
            "sha256": digest,
            "path": str(target),
        }
        return target

    raise OSError("Could not reserve a unique filename for the uploaded PDF.")


def _read_recent_audits() -> List[Dict[str, Any]]:
    if not RECENT_AUDITS_PATH.exists():
        return []
    try:
        with RECENT_AUDITS_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read recent audit history.", exc_info=True)
        return []


def _write_recent_audits(entries: List[Dict[str, Any]]) -> None:
    with RECENT_AUDITS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(entries[:5], fh, indent=2)


def _load_demo_paper() -> None:
    st.info("This is a synthetic demonstration document, not a published research paper.")
    demo_pdf = APP_ROOT / "demo.pdf"
    if not demo_pdf.exists():
        from run_demo import build_demo_pdf
        build_demo_pdf(demo_pdf)
    if demo_pdf.exists():
        result = _process_pdf(demo_pdf)
        if result is not None:
            st.session_state["audit_results"] = result["audit_results"]
            st.session_state["claims"] = result["claims"]
            st.session_state["evidences"] = result["evidences"]
            st.session_state["sentences"] = result["sentences"]
            st.session_state["citations"] = result["citations"]
            st.session_state["numerical_warnings"] = result["numerical_warnings"]
            st.session_state["file_name"] = demo_pdf.name
            _write_recent_audits([
                {"label": "Demo paper", "filename": demo_pdf.name, "mtime": str(demo_pdf.stat().st_mtime)}
            ] + _read_recent_audits())
            st.success("Demo paper loaded into the workspace.")


def _process_pdf(file_path: Path | str, use_ocr: bool = False) -> Dict[str, Any] | None:
    path = str(file_path)
    try:
        pages = extract_pages(path, use_ocr=use_ocr)
    except OCRUnavailableError as exc:
        st.error(str(exc))
        return None
    if not pages:
        st.error("No readable pages were extracted from the uploaded PDF.")
        return None

    sentences = preprocess(pages)
    if not sentences:
        if not any(text.strip() for _, text in pages):
            if use_ocr:
                st.error("OCR did not recover any text from the PDF pages.")
            else:
                st.error(
                    "PDF pages were found, but none contain selectable text. This file "
                    "may be scanned. Enable OCR and install Tesseract OCR, or use a "
                    "text-searchable PDF."
                )
        else:
            st.error("The document contains no usable sentences after text extraction.")
        return None

    doc_text = "\n".join(page for _, page in pages)
    citations = extract_citations(doc_text)
    claims = extract_claims(sentences)
    if not claims:
        st.warning("The document did not yield any strong scientific claim candidates.")
        return {
            "audit_results": [],
            "claims": [],
            "evidences": [],
            "sentences": sentences,
            "pages": pages,
            "citations": citations,
            "numerical_warnings": [],
        }

    evidences = retrieve_evidence(claims, sentences)
    audit_results = audit_claims(claims, evidences)
    model_error = get_nli_model_error()
    if model_error:
        st.warning(
            "The NLI model could not be loaded. Claims with retrieved passages are "
            "marked for manual review. Check model download, disk space, and cache "
            f"availability. Details: {model_error}"
        )
    numerical_warnings = check_numerical_consistency(claims, evidences)
    return {
        "audit_results": audit_results,
        "claims": claims,
        "evidences": evidences,
        "sentences": sentences,
        "pages": pages,
        "citations": citations,
        "numerical_warnings": numerical_warnings,
        "nli_model_error": model_error,
    }


def _compute_summary(results: List[Any]) -> Dict[str, int]:
    counts = {
        "SUPPORTED": 0,
        "CONTRADICTED": 0,
        "INSUFFICIENT_EVIDENCE": 0,
        "HUMAN_REVIEW_REQUIRED": 0,
    }
    for entry in results:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    return counts


def _report_payloads(results: List[Any]) -> Tuple[bytes, bytes]:
    """Generate downloadable report bytes without writing reports into the project."""
    with tempfile.TemporaryDirectory(prefix="papertrail-report-") as directory:
        csv_path = Path(directory) / "audit_report.csv"
        pdf_path = Path(directory) / "audit_report.pdf"
        export_to_csv(results, str(csv_path))
        export_to_pdf(results, str(pdf_path))
        return csv_path.read_bytes(), pdf_path.read_bytes()


def _overview_page() -> None:
    st.markdown('<div class="hero-panel">', unsafe_allow_html=True)
    def _go_to_workspace():
        st.session_state["page"] = "Workspace"
    st.title("PaperTrail AI")
    st.subheader("Every research claim deserves evidence.")
    st.write(
        "PaperTrail AI extracts candidate research claims, retrieves passages from the same "
        "paper, and uses an NLI model to identify support, contradiction, or uncertainty. "
        "It does not independently verify scientific truth or cited sources."
    )
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.button("Start New Audit", width="stretch", on_click=_go_to_workspace)

    with col_b:
        if st.button("Try Demo Paper", width="stretch"):
            _load_demo_paper()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    recent = _read_recent_audits()
    if recent:
        st.caption("Recent audits")
        for item in recent[:3]:
            st.markdown(f"- {item.get('label', 'Audit')}: {item.get('filename', 'unknown')}")
    else:
        st.info("No local audits yet. Upload a paper or run the demo to generate the first audit.")

    st.markdown("---")
    st.subheader("Verification pipeline")
    cols = st.columns(4)
    steps = [
        ("1. Extraction", "Parse each PDF page and recover raw scientific text."),
        ("2. Preprocessing", "Normalize text, preserve structure, and segment into sentence units."),
        ("3. Claim detection", "Surface verifiable scientific claims using NLP heuristics and quantitative cues."),
        ("4. Audit", "Rank passages, assess NLI support, and classify claims for review."),
    ]
    for i, (title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"<div class='glass-card'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Synthetic example (illustrative only)")
    st.markdown(
        """
        <div class='glass-card'>
            <b>Example claim:</b> “The method achieves 95% accuracy on the benchmark.”<br><br>
            This static example is not an audit result and is not evidence that the claim is true.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "This platform evaluates how strongly a claim is supported by the evidence contained in the document. "
        "It is a research-review assistant, not an independent scientific authority."
    )


def _workspace_page() -> None:
    _ensure_storage()
    st.header("Research Workspace")
    st.write("Upload a PDF to run claim extraction, evidence retrieval, and audit classification.")
    use_ocr = st.checkbox(
        "Use OCR for scanned pages",
        help=(
            "Requires Tesseract OCR with English language data installed. On Windows, "
            "install with `winget install --id UB-Mannheim.TesseractOCR -e`, restart "
            "the app, and verify with `& \"$env:ProgramFiles\\Tesseract-OCR\\tesseract.exe\" "
            "--list-langs`."
        ),
    )

    uploaded = st.file_uploader("Upload a PDF research paper", type=["pdf"], key="pdf_upload")
    if uploaded is not None:
        file_name = Path(uploaded.name).name or "uploaded.pdf"
        try:
            target = _save_uploaded_pdf(file_name, uploaded.getvalue())
        except OSError as exc:
            st.error(f"Could not save the uploaded PDF: {exc}")
            return
        file_name = target.name
        st.session_state["file_name"] = file_name
        st.session_state.pop("audit_results", None)
        st.success(f"Saved {file_name} to the local workspace.")

        with st.spinner("Running the audit pipeline…"):
            result = _process_pdf(target, use_ocr=use_ocr)

        if result is not None:
            st.session_state["audit_results"] = result["audit_results"]
            st.session_state["claims"] = result["claims"]
            st.session_state["evidences"] = result["evidences"]
            st.session_state["sentences"] = result["sentences"]
            st.session_state["citations"] = result["citations"]
            st.session_state["numerical_warnings"] = result["numerical_warnings"]
            _write_recent_audits([
                {"label": uploaded.name, "filename": file_name, "mtime": str(target.stat().st_mtime)}
            ] + _read_recent_audits())
            st.success("The analysis is complete. Review the audit dashboard and claim explorer.")

    st.markdown("---")
    if st.button("Load demo paper", width="stretch"):
        _load_demo_paper()

    if "file_name" in st.session_state:
        st.caption(f"Current file: {st.session_state['file_name']}")


def _dashboard_page() -> None:
    if "audit_results" not in st.session_state:
        st.warning("No audit data is available yet. Upload a PDF or load the demo paper.")
        return

    results = st.session_state["audit_results"]
    summary = _compute_summary(results)
    total = len(results)
    citations = st.session_state.get("citations", [])
    numerical_warnings = st.session_state.get("numerical_warnings", [])

    st.header("Audit Results Dashboard")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Total claims", total)
    metric_cols[1].metric("Supported", summary.get("SUPPORTED", 0))
    metric_cols[2].metric("Contradicted", summary.get("CONTRADICTED", 0))
    metric_cols[3].metric("Insufficient", summary.get("INSUFFICIENT_EVIDENCE", 0))
    metric_cols[4].metric("Needs review", summary.get("HUMAN_REVIEW_REQUIRED", 0))

    if total == 0:
        st.info("The extraction step did not identify any actionable scientific claims in this document.")
        return

    coverage = sum(1 for item in results if item.evidence) / total
    st.metric("Evidence coverage", f"{coverage * 100:.1f}%")

    status_chart = pd.DataFrame({
        "Status": ["Supported", "Contradicted", "Insufficient", "Needs review"],
        "Count": [
            summary.get("SUPPORTED", 0),
            summary.get("CONTRADICTED", 0),
            summary.get("INSUFFICIENT_EVIDENCE", 0),
            summary.get("HUMAN_REVIEW_REQUIRED", 0),
        ],
    })
    left, right = st.columns([1.6, 1])
    with left:
        st.bar_chart(status_chart.set_index("Status"), height=220)
    with right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write(f"**Citations found:** {len(citations)}")
        st.write(f"**Numerical warnings:** {len(numerical_warnings)}")
        if citations:
            st.write("Recent citation markers:")
            for item in citations[:5]:
                st.caption(item)
        else:
            st.caption("No explicit citation markers were detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "Citation checks identify formatting markers only; cited sources and their claims "
        "are not externally verified."
    )
    if numerical_warnings:
        st.subheader("Numerical review prompts")
        for claim_id, warning in numerical_warnings:
            st.write(f"Claim {claim_id}: {warning}")
    st.caption(
        "Numerical comparisons are limited to matching metric names and units. Differences "
        "are prompts for review, not confirmed errors."
    )
    rows = []
    for result in results:
        rows.append(
            {
                "Claim ID": result.claim.claim_id,
                "Page": result.claim.page_number,
                "Status": result.status,
                "Evidence": len(result.evidence),
                "Explanation": result.explanation,
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

    priority_rows = []
    for result in results:
        priority_rows.append({
            "Claim ID": result.claim.claim_id,
            "Title": result.claim.text[:80],
            "Status": result.status,
            "Evidence count": len(result.evidence),
        })
    st.subheader("Priority review queue")
    st.dataframe(priority_rows, width="stretch", hide_index=True)

    csv_bytes, pdf_bytes = _report_payloads(results)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download CSV",
            csv_bytes,
            file_name="audit_report.csv",
            mime="text/csv",
            on_click="ignore",
        )
    with col_b:
        st.download_button(
            "Download PDF",
            pdf_bytes,
            file_name="audit_report.pdf",
            mime="application/pdf",
            on_click="ignore",
        )


def _claim_explorer_page() -> None:
    if "audit_results" not in st.session_state:
        st.warning("Run an audit first to populate claims.")
        return

    st.header("Claim Explorer")
    results = st.session_state["audit_results"]
    if not results:
        st.info("No claim candidates were extracted from this document.")
        return
    options = {f"Claim {r.claim.claim_id}: {r.claim.text[:90]}...": r.claim.claim_id for r in results}
    selected_label = st.selectbox("Select claim", list(options.keys()))
    claim_id = options[selected_label]
    result = next(r for r in results if r.claim.claim_id == claim_id)

    st.subheader(f"Claim {result.claim.claim_id} — Page {result.claim.page_number}")
    st.write(result.claim.text)
    st.write(f"**Status:** {result.status}")
    st.write(result.explanation)
    st.caption(
        "NLI confidence is the model's confidence in its label, not a probability that "
        "the scientific claim is true."
    )

    if result.evidence:
        ev_rows = []
        for ev in result.evidence:
            ev_rows.append(
                {
                    "Evidence ID": ev.evidence_id,
                    "Page": ev.page_number,
                    "Similarity": round(ev.similarity, 3),
                    "NLI": ev.nli_label or "-",
                    "Score": round(ev.nli_score, 3) if ev.nli_score is not None else "-",
                    "Text": ev.text,
                }
            )
        st.dataframe(ev_rows, width="stretch", hide_index=True)
    else:
        st.info("No evidence was retrieved for this claim.")


def _evidence_inspector_page() -> None:
    if "audit_results" not in st.session_state:
        st.warning("No evidence available. Run an audit first.")
        return

    st.header("Evidence Inspector")
    evidences = []
    for result in st.session_state["audit_results"]:
        for ev in result.evidence:
            evidences.append({
                "Evidence ID": ev.evidence_id,
                "Claim ID": result.claim.claim_id,
                "Page": ev.page_number,
                "Text": ev.text,
                "Similarity": ev.similarity,
                "NLI": ev.nli_label,
                "Score": ev.nli_score,
            })

    if not evidences:
        st.info("No retrieved evidence is available for inspection.")
        return

    selected_id = st.selectbox("Choose evidence to inspect", [item["Evidence ID"] for item in evidences])
    selected = next(item for item in evidences if item["Evidence ID"] == selected_id)
    st.subheader(f"Evidence {selected['Evidence ID']} — Page {selected['Page']}")
    st.write(selected["Text"])
    st.write(f"**Claim ID:** {selected['Claim ID']}")
    st.write(f"**Similarity:** {selected['Similarity']:.3f}")
    st.write(f"**NLI:** {selected['NLI'] or 'N/A'}")
    score = selected["Score"]
    if isinstance(score, float):
        st.write(f"**NLI score:** {score:.3f}")
    else:
        st.write("**NLI score:** N/A")


def _export_report_page() -> None:
    if "audit_results" not in st.session_state:
        st.warning("No document has been audited yet.")
        return

    st.header("Export Report")
    results = st.session_state["audit_results"]
    csv_bytes, pdf_bytes = _report_payloads(results)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download CSV",
            csv_bytes,
            file_name="audit_report.csv",
            mime="text/csv",
            on_click="ignore",
        )
    with col_b:
        st.download_button(
            "Download PDF",
            pdf_bytes,
            file_name="audit_report.pdf",
            mime="application/pdf",
            on_click="ignore",
        )


def main() -> None:
    _dark_theme()
    nav = st.sidebar.radio(
        "Navigation",
        ["Overview", "Workspace", "Dashboard", "Claim Explorer", "Evidence Inspector", "Export Report"],
        key="page",
    )
    if nav == "Overview":
        _overview_page()
    elif nav == "Workspace":
        _workspace_page()
    elif nav == "Dashboard":
        _dashboard_page()
    elif nav == "Claim Explorer":
        _claim_explorer_page()
    elif nav == "Evidence Inspector":
        _evidence_inspector_page()
    elif nav == "Export Report":
        _export_report_page()


if __name__ == "__main__":
    main()
