import app
from src.pdf_parser import OCRUnavailableError


def test_uploaded_files_do_not_overwrite_existing_files(monkeypatch, tmp_path):
    state = {}
    monkeypatch.setattr(app, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(app.st, "session_state", state)
    existing = tmp_path / "paper.pdf"
    existing.write_bytes(b"existing paper")

    saved = app._save_uploaded_pdf("paper.pdf", b"new paper")
    repeated = app._save_uploaded_pdf("paper.pdf", b"new paper")

    assert existing.read_bytes() == b"existing paper"
    assert saved.name == "paper (1).pdf"
    assert repeated == saved


def test_scanned_pdf_message_explains_ocr_limitation(monkeypatch):
    errors = []
    monkeypatch.setattr(
        app, "extract_pages", lambda path, use_ocr=False: [(1, ""), (2, "")]
    )
    monkeypatch.setattr(app.st, "error", errors.append)

    result = app._process_pdf("scanned.pdf")

    assert result is None
    assert any("Enable OCR" in message for message in errors)


def test_citations_are_retained_when_no_claims_are_detected(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        app,
        "extract_pages",
        lambda path, use_ocr=False: [(1, "See prior work [3].")],
    )
    monkeypatch.setattr(app.st, "warning", warnings.append)

    result = app._process_pdf("paper.pdf")

    assert result is not None
    assert result["citations"] == ["[3]"]
    assert warnings


def test_ocr_setup_error_is_shown_in_the_app(monkeypatch):
    errors = []

    def unavailable(path, use_ocr=False):
        raise OCRUnavailableError(
            "Install Tesseract with `winget install --id UB-Mannheim.TesseractOCR -e`."
        )

    monkeypatch.setattr(app, "extract_pages", unavailable)
    monkeypatch.setattr(app.st, "error", errors.append)

    assert app._process_pdf("scanned.pdf", use_ocr=True) is None
    assert any("winget install --id UB-Mannheim.TesseractOCR -e" in message for message in errors)
