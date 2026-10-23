from typing import List, Optional
from pydantic import BaseModel, Field

class PageText(BaseModel):
    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(..., description="Raw extracted text of the page")

class Sentence(BaseModel):
    page_number: int = Field(..., description="Page where the sentence originates")
    text: str = Field(..., description="Sentence text")
    sentence_id: int = Field(..., description="Unique id within the document")

class Claim(BaseModel):
    claim_id: int = Field(..., description="Unique claim identifier")
    text: str = Field(..., description="Extracted claim sentence")
    page_number: int = Field(..., description="Page where claim was found")
    source_sentence_id: int = Field(..., description="Sentence id of the original sentence")

class Evidence(BaseModel):
    evidence_id: int = Field(..., description="Unique evidence identifier")
    claim_id: int = Field(..., description="Associated claim identifier")
    text: str = Field(..., description="Evidence passage text")
    page_number: int = Field(..., description="Page where evidence occurs")
    similarity: float = Field(..., description="Semantic similarity score (0‑1)")
    nli_label: Optional[str] = Field(None, description="Entailment / Contradiction / Neutral")
    nli_score: Optional[float] = Field(None, description="Confidence of NLI prediction")

class AuditResult(BaseModel):
    claim: Claim
    status: str = Field(..., description="One of SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, HUMAN_REVIEW_REQUIRED")
    evidence: List[Evidence] = Field(default_factory=list)
    explanation: str = Field(..., description="Human‑readable justification for the status")
