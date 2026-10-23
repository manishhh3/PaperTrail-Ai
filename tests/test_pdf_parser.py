import fitz
import pytest

from src.pdf_parser import OCRUnavailableError, extract_pages


def test_extract_pages_nonexistent(tmp_path):
    result = extract_pages(str(tmp_path / "nosuch.pdf"))
    assert result == []


def test_extract_pages_preserves_page_numbers_and_text(tmp_path):
    pdf_path = tmp_path / "text.pdf"
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "First page text.")
    document.new_page()
    document.save(pdf_path)
    document.close()

    assert extract_pages(str(pdf_path)) == [
        (1, "First page text.\n"),
        (2, ""),
    ]


def test_extract_pages_handles_malformed_pdf(tmp_path):
    pdf_path = tmp_path / "malformed.pdf"
    pdf_path.write_bytes(b"not a PDF")

    assert extract_pages(str(pdf_path)) == []


def test_ocr_requested_runs_on_pages_without_selectable_text(tmp_path, monkeypatch):
    pdf_path = tmp_path / "image-only.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    class FakeTesseract:
        @staticmethod
        def image_to_string(image, lang):
            assert image.size[0] > 0
            assert lang == "eng"
            return "Recognized text from scanned page."

    monkeypatch.setattr("src.pdf_parser._ocr_engine", lambda language: FakeTesseract)

    assert extract_pages(str(pdf_path), use_ocr=True) == [
        (1, "Recognized text from scanned page.")
    ]


def test_ocr_missing_executable_is_reported(tmp_path, monkeypatch):
    pdf_path = tmp_path / "image-only.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    def missing_engine(language):
        raise OCRUnavailableError("Install Tesseract OCR.")

    monkeypatch.setattr("src.pdf_parser._ocr_engine", missing_engine)

    with pytest.raises(OCRUnavailableError, match="Install Tesseract OCR"):
        extract_pages(str(pdf_path), use_ocr=True)


def test_ocr_dpi_is_validated(tmp_path):
    with pytest.raises(ValueError, match="between 72 and 600"):
        extract_pages(str(tmp_path / "does-not-need-to-exist.pdf"), use_ocr=True, ocr_dpi=20)
