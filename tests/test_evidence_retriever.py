from src.evidence_retriever import retrieve_evidence
from src.schemas import Claim, Sentence


def test_tfidf_retrieval_excludes_claim_source_and_preserves_page(monkeypatch):
    monkeypatch.setattr("src.evidence_retriever._load_embedding_model", lambda: None)
    sentences = [
        Sentence(
            sentence_id=1,
            page_number=1,
            text="Our model achieves 95% accuracy on the benchmark.",
        ),
        Sentence(
            sentence_id=2,
            page_number=3,
            text="The benchmark accuracy for the model was measured at 90%.",
        ),
        Sentence(
            sentence_id=3,
            page_number=5,
            text="Training used a single graphics processor.",
        ),
    ]
    claims = [
        Claim(
            claim_id=1,
            text=sentences[0].text,
            page_number=1,
            source_sentence_id=1,
        )
    ]

    evidences = retrieve_evidence(claims, sentences, top_k=3)

    assert evidences
    assert all(evidence.text != sentences[0].text for evidence in evidences)
    assert evidences[0].page_number == 3
    assert evidences[0].similarity > 0


def test_retrieval_handles_empty_inputs(monkeypatch):
    monkeypatch.setattr("src.evidence_retriever._load_embedding_model", lambda: None)

    assert retrieve_evidence([], []) == []
    assert retrieve_evidence(
        [Claim(claim_id=1, text=" ", page_number=1, source_sentence_id=1)],
        [Sentence(sentence_id=1, page_number=1, text=" ")],
    ) == []


def test_retrieval_handles_text_without_indexable_terms(monkeypatch):
    monkeypatch.setattr("src.evidence_retriever._load_embedding_model", lambda: None)
    claim = Claim(claim_id=1, text="!!!", page_number=1, source_sentence_id=1)
    sentence = Sentence(sentence_id=2, page_number=1, text="...")

    assert retrieve_evidence([claim], [sentence]) == []
