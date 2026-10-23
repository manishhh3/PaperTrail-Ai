"""Evidence retrieval utilities.

If the ``sentence-transformers`` library is available we embed each sentence and the claims
and perform a cosine‑similarity search. When the library cannot be imported we fall back
to a simple lexical TF‑IDF‑like overlap using ``sklearn``'s ``TfidfVectorizer``.
"""

import logging
from functools import lru_cache
from typing import List

from .schemas import Claim, Evidence, Sentence

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_embedding_model():
    """Load the optional embedding model on first use, not during app startup."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:
        logger.warning("SentenceTransformer unavailable (%s); using TF-IDF.", exc)
        return None
    logger.info("SentenceTransformer model loaded for evidence retrieval.")
    return model

def retrieve_evidence(
    claims: List[Claim],
    sentences: List[Sentence],
    top_k: int = 3,
) -> List[Evidence]:
    """Retrieve top‑k most similar sentences for each claim.

    Returns a flat list of :class:`Evidence` objects (one per claim‑sentence pair).
    """
    if not claims or not sentences or top_k <= 0:
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    claim_texts = [claim.text for claim in claims]
    sentence_texts = [sentence.text for sentence in sentences]
    if not any(text.strip() for text in claim_texts + sentence_texts):
        return []

    evidences: List[Evidence] = []
    evidence_counter = 0
    model = _load_embedding_model()
    use_embeddings = model is not None
    if use_embeddings:
        try:
            claim_vectors = model.encode(
                claim_texts,
                convert_to_tensor=False,
                show_progress_bar=False,
            )
            sentence_vectors = model.encode(
                sentence_texts,
                convert_to_tensor=False,
                show_progress_bar=False,
            )
            similarities = cosine_similarity(claim_vectors, sentence_vectors)
        except Exception as exc:
            logger.warning("Embedding retrieval failed (%s); using TF-IDF.", exc)
            use_embeddings = False

    if not use_embeddings:
        vectorizer = TfidfVectorizer()
        corpus = claim_texts + sentence_texts
        try:
            vectors = vectorizer.fit_transform(corpus)
        except ValueError as exc:
            logger.warning("TF-IDF could not build a vocabulary: %s", exc)
            return []
        claim_vectors = vectors[: len(claims)]
        sentence_vectors = vectors[len(claims) :]
        similarities = cosine_similarity(claim_vectors, sentence_vectors)

    for claim_index, claim in enumerate(claims):
        candidates = [
            index
            for index, sentence in enumerate(sentences)
            if sentence.sentence_id != claim.source_sentence_id
        ]
        ranked = sorted(
            candidates,
            key=lambda index: similarities[claim_index, index],
            reverse=True,
        )[:top_k]
        for index in ranked:
            evidence_counter += 1
            evidences.append(
                Evidence(
                    evidence_id=evidence_counter,
                    claim_id=claim.claim_id,
                    text=sentences[index].text,
                    page_number=sentences[index].page_number,
                    similarity=float(similarities[claim_index, index]),
                )
            )
    return evidences
