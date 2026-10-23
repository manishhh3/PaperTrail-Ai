"""PDF parsing utilities.

Provides safe extraction of page text while preserving page numbers.
Handles corrupted or empty PDFs gracefully.
"""

import fitz
import logging
import os
import shutil
from typing import List, Tuple

logger = logging.getLogger(__name__)

_OCR_INSTALL_INSTRUCTIONS = (
    "Install Tesseract on Windows with `winget install --id UB-Mannheim.TesseractOCR "
    "-e`, then verify `& \"$env:ProgramFiles\\Tesseract-OCR\\tesseract.exe\" --version` "
    "and `& \"$env:ProgramFiles\\Tesseract-OCR\\tesseract.exe\" --list-langs` "
    "(which must include `eng`). Restart Streamlit after installation."
)


class OCRUnavailableError(RuntimeError):
    """Raised when OCR was requested but its executable or language data is missing."""


def _ocr_engine(language: str):
    try:
        import pytesseract

        if not shutil.which("tesseract"):
            install_roots = [
                os.environ.get("ProgramFiles"),
                os.environ.get("ProgramFiles(x86)"),
            ]
            executable = next(
                (
                    os.path.join(root, "Tesseract-OCR", "tesseract.exe")
                    for root in install_roots
                    if root
                    and os.path.isfile(
                        os.path.join(root, "Tesseract-OCR", "tesseract.exe")
                    )
                ),
                None,
            )
            if executable:
                pytesseract.pytesseract.tesseract_cmd = executable
        pytesseract.get_tesseract_version()
        available_languages = pytesseract.get_languages(config="")
    except ImportError as exc:
        raise OCRUnavailableError(
            "OCR requires the pytesseract Python package and the Tesseract OCR "
            f"executable. Install project requirements and {_OCR_INSTALL_INSTRUCTIONS}"
        ) from exc
    except Exception as exc:
        raise OCRUnavailableError(
            "Tesseract OCR was requested but its executable could not be run. "
            f"{_OCR_INSTALL_INSTRUCTIONS} "
            f"Details: {exc}"
        ) from exc

    requested = set(language.split("+"))
    missing = requested - set(available_languages)
    if missing:
        raise OCRUnavailableError(
            f"Tesseract language data is missing for: {', '.join(sorted(missing))}. "
            "Install the matching traineddata files and verify them with "
            "`tesseract --list-langs`, which must include `eng`."
        )
    return pytesseract


def extract_pages(
    pdf_path: str,
    use_ocr: bool = False,
    ocr_language: str = "eng",
    ocr_dpi: int = 220,
) -> List[Tuple[int, str]]:
    """Extract text from each page of a PDF.

    Returns a list of ``(page_number, text)`` tuples where ``page_number`` is 1-indexed.
    Empty pages yield an empty string. Errors are caught and logged; the function
    returns an empty list on failure. If ``use_ocr`` is set, Tesseract OCR is run on
    pages where text extraction returns no selectable text.
    """
    if ocr_dpi < 72 or ocr_dpi > 600:
        raise ValueError("OCR DPI must be between 72 and 600.")

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        logger.exception("Failed to open PDF %s.", pdf_path)
        return []

    try:
        pytesseract = _ocr_engine(ocr_language) if use_ocr else None
        pages: List[Tuple[int, str]] = []
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if use_ocr and not text.strip():
                    pixmap = page.get_pixmap(dpi=ocr_dpi, alpha=False)
                    from PIL import Image

                    image = Image.frombytes(
                        "RGB",
                        (pixmap.width, pixmap.height),
                        pixmap.samples,
                    )
                    text = pytesseract.image_to_string(image, lang=ocr_language)
                    if not text.strip():
                        logger.warning(
                            "OCR returned no text for PDF page %s; the page may be "
                            "blank or unreadable.",
                            page_num + 1,
                        )
            except OCRUnavailableError:
                raise
            except Exception as exc:
                logger.exception("Could not extract page %s.", page_num + 1)
                if use_ocr:
                    raise OCRUnavailableError(
                        f"OCR failed on PDF page {page_num + 1}: {exc}"
                    ) from exc
                text = ""
            pages.append((page_num + 1, text))
        return pages
    finally:
        doc.close()
