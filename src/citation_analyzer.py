"""Simple citation analysis for PaperTrail AI.

We look for typical in‑text citation patterns such as ``[1]`` or ``(Smith et al., 2020)``.
The function returns a list of strings representing the raw citation markers found in the
document. No external lookup is performed.
"""

import re
from typing import List

def extract_citations(text: str) -> List[str]:
    """Return a list of citation markers found in *text*.

    Parameters
    ----------
    text: str
        Full document text (concatenated pages).
    """
    # Pattern for numeric brackets e.g., [12]
    bracket_pattern = r"\[\s*\d+(?:,\s*\d+)*\s*\]"
    # Pattern for author‑year parentheses e.g., (Smith et al., 2020)
    author_year_pattern = r"\(\s*[A-Z][A-Za-z]+(?: et al\.)?,\s*\d{4}\s*\)"
    matches = re.findall(bracket_pattern, text)
    matches += re.findall(author_year_pattern, text)
    # Deduplicate while preserving order
    seen = set()
    citations = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            citations.append(m)
    return citations
