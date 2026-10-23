"""Report generation utilities.

Provides functions to export audit results to CSV and PDF (using ReportLab).
"""

import csv
from html import escape
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

from .schemas import AuditResult

logger = logging.getLogger(__name__)

def export_to_csv(audit_results: List[AuditResult], output_path: str) -> str:
    """Write audit results to a CSV file.

    Columns: claim_id, claim_text, page, status, evidence_text, evidence_page,
    similarity, nli_label, nli_score, explanation
    """
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "claim_id",
                "claim_text",
                "claim_page",
                "status",
                "evidence_id",
                "evidence_text",
                "evidence_page",
                "similarity",
                "nli_label",
                "nli_score",
                "explanation",
            ])
            for result in audit_results:
                if result.evidence:
                    for ev in result.evidence:
                        writer.writerow([
                            result.claim.claim_id,
                            result.claim.text,
                            result.claim.page_number,
                            result.status,
                            ev.evidence_id,
                            ev.text,
                            ev.page_number,
                            f"{ev.similarity:.3f}",
                            ev.nli_label or "",
                            f"{ev.nli_score:.3f}" if ev.nli_score is not None else "",
                            result.explanation,
                        ])
                else:
                    writer.writerow([
                        result.claim.claim_id,
                        result.claim.text,
                        result.claim.page_number,
                        result.status,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        result.explanation,
                    ])
        logger.info(f"CSV report written to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to write CSV report: {e}")
        raise

def export_to_pdf(audit_results: List[AuditResult], output_path: str) -> str:
    """Create a PDF report using ReportLab.

    The PDF contains a header, a table per claim, and a summary section.
    """
    try:
        doc = SimpleDocTemplate(output_path, pagesize=LETTER)
        styles = getSampleStyleSheet()
        elements = []

        # Header
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph("<b>PaperTrail AI Audit Report</b>", styles["Title"]))
        elements.append(Paragraph(f"Generated: {escape(now)}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Summary table
        summary_data = [["Total Claims", len(audit_results)]]
        status_counts = {}
        for r in audit_results:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1
        for status, cnt in status_counts.items():
            summary_data.append([status.replace("_", " "), cnt])
        summary_table = Table(summary_data, hAlign="LEFT")
        summary_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ])
        )
        elements.append(summary_table)
        elements.append(Spacer(1, 12))

        # Detailed claim sections
        for result in audit_results:
            elements.append(
                Paragraph(
                    f"<b>Claim {result.claim.claim_id} "
                    f"(Page {result.claim.page_number})</b>",
                    styles["Heading2"],
                )
            )
            elements.append(Paragraph(escape(result.claim.text), styles["BodyText"]))
            elements.append(
                Paragraph(f"<i>Status:</i> {escape(result.status)}", styles["Italic"])
            )
            elements.append(
                Paragraph(
                    f"Explanation: {escape(result.explanation)}", styles["Normal"]
                )
            )
            if result.evidence:
                ev_data = [["Evidence ID", "Page", "Similarity", "NLI", "Text"]]
                for ev in result.evidence:
                    ev_data.append([
                        str(ev.evidence_id),
                        str(ev.page_number),
                        f"{ev.similarity:.3f}",
                        (
                            f"{escape(ev.nli_label or '')} ({ev.nli_score:.2f})"
                            if ev.nli_score is not None
                            else escape(ev.nli_label or "")
                        ),
                        Paragraph(escape(ev.text), styles["BodyText"]),
                    ])
                ev_table = Table(ev_data, colWidths=[42, 32, 50, 68, 276])
                ev_table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ])
                )
                elements.append(ev_table)
                elements.append(Spacer(1, 12))
            elements.append(Spacer(1, 12))

        doc.build(elements)
        logger.info(f"PDF report written to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to write PDF report: {e}")
        raise
