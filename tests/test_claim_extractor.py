from src.claim_extractor import extract_claims
from src.schemas import Sentence


def test_filters_paper_structure_and_method_sentences():
    sentences = [
        Sentence(
            page_number=1,
            sentence_id=index,
            text=text,
        )
        for index, text in enumerate(
            [
                "We present these results in Table 3.",
                "The code we used to train our models is available at the project site.",
                "To evaluate generalization, we performed experiments on a parsing task.",
                "To evaluate model components, we varied our base model and measured performance.",
                "We inspect attention distributions and discuss examples in the appendix.",
            ],
            start=1,
        )
    ]

    assert extract_claims(sentences) == []


def test_keeps_a_quantitative_finding_that_references_a_table():
    sentence = Sentence(
        page_number=8,
        sentence_id=1,
        text="We report that the Transformer achieves 28.4 BLEU on WMT 2014 (Table 2).",
    )

    claims = extract_claims([sentence])

    assert len(claims) == 1
    assert claims[0].page_number == 8


def test_removes_trailing_author_note_from_claim_text():
    sentence = Sentence(
        page_number=1,
        sentence_id=1,
        text="We show that the model generalizes well to parsing. \u2217Equal contribution.",
    )

    claims = extract_claims([sentence])

    assert len(claims) == 1
    assert claims[0].text == "We show that the model generalizes well to parsing."
