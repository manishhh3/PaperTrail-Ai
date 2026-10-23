"""Preprocessing utilities for PaperTrail AI.

- Normalizes whitespace.
- Splits extracted page text into sentences while preserving page numbers.
- Returns a list of :class:`~src.schemas.Sentence` objects.
"""

import re
from typing import List, Tuple

from .schemas import Sentence


def _clean_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n+", text)
    return "\n\n".join(
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in paragraphs
        if paragraph.strip()
    )


def preprocess(pages: List[Tuple[int, str]]) -> List[Sentence]:
    """Convert raw page text into a flat list of sentences.

    Pages are split into paragraphs first, then sentence fragments. Paragraph breaks are
    treated as natural separators while still keeping all page-number metadata intact.
    """
    sentences: List[Sentence] = []
    sentence_counter = 0

    for page_number, raw_text in pages:
        if raw_text is None:
            continue

        cleaned = _clean_whitespace(raw_text)
        if not cleaned:
            continue

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", cleaned) if p.strip()]
        if not paragraphs:
            paragraphs = [cleaned]

        for paragraph in paragraphs:
            fragments = re.split(r"(?<=[.!?])\s+(?=[\w\"'(])", paragraph)
            for fragment in fragments:
                txt = fragment.strip()
                if not txt:
                    continue
                sentence_counter += 1
                sentences.append(
                    Sentence(
                        page_number=page_number,
                        text=txt,
                        sentence_id=sentence_counter,
                    )
                )

    return sentences
