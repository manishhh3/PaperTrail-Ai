"""Simple claim extraction for scientific papers.

We use a lightweight heuristic based on a whitelist of reporting verbs and the presence of
numeric results or the phrase "we". This is deliberately conservative – only sentences that
match the pattern become candidates.
"""

import re
from typing import List

from .schemas import Claim, Sentence

# Common scientific reporting verbs (lower-case)
_REPORTING_VERBS = [
    "show",
    "demonstrate",
    "prove",
    "achieve",
    "improve",
    "propose",
    "introduce",
    "present",
    "evaluate",
    "find",
    "observe",
    "report",
    "indicate",
    "reveal",
    "confirm",
    "outperform",
    "surpass",
    "exceed",
    "reduce",
    "increase",
    "decrease",
    "boost",
    "attain",
]

_VERB_PATTERN = re.compile(r"\b(?:" + "|".join(_REPORTING_VERBS) + r")\b", re.IGNORECASE)
_TRAILING_AUTHOR_NOTE_PATTERN = re.compile(
    r"\s*[\u2217*\u2020\u2021]+\s*(?:equal contribution|corresponding author)\.?\s*$",
    re.IGNORECASE,
)

_STRUCTURAL_PATTERNS = (
    re.compile(
        r"\bwe\s+(?:present|report|summarize)\s+(?:these|the|our)\s+results?\s+"
        r"in\s+(?:table|fig(?:ure)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:source\s+)?code\b.*\b(?:is|are)\s+available\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bto\s+evaluate\b.*\bwe\s+(?:performed|conducted|carried\s+out)\s+"
        r"experiments\b|\bto\s+evaluate\b.*\bwe\s+"
        r"(?:varied|measured|evaluated|compared)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwe\s+(?:inspect|examine|visualize)\b.*\b(?:appendix|supplement)\b",
        re.IGNORECASE,
    ),
)


def _looks_like_claim(text: str) -> bool:
    lowered = text.lower()
    if any(pattern.search(text) for pattern in _STRUCTURAL_PATTERNS):
        return False
    has_numeric = bool(re.search(r"\d", text))
    has_research_signal = any(phrase in lowered for phrase in [
        "we ", "our ", "results", "accuracy", "performance", "outperforms",
        "baseline", "experiment", "benchmark", "improvement", "loss", "score"
    ])
    return _VERB_PATTERN.search(text) and (has_numeric or has_research_signal)


def extract_claims(sentences: List[Sentence]) -> List[Claim]:
    """Identify candidate scientific claims.

    Parameters
    ----------
    sentences: List[Sentence]
        Pre-processed sentences from the document.

    Returns
    -------
    List[Claim]
        Candidate claims with minimal metadata.
    """
    claims: List[Claim] = []
    claim_id = 0
    for s in sentences:
        claim_text = _TRAILING_AUTHOR_NOTE_PATTERN.sub("", s.text).strip()
        if _looks_like_claim(claim_text):
            claim_id += 1
            claims.append(
                Claim(
                    claim_id=claim_id,
                    text=claim_text,
                    page_number=s.page_number,
                    source_sentence_id=s.sentence_id,
                )
            )
    return claims
