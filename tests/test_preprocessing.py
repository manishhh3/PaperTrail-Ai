from src.preprocessing import preprocess


def test_preprocess_empty():
    pages = []
    sentences = preprocess(pages)
    assert sentences == []

def test_preprocess_basic():
    pages = [(1, "This is a sentence. And another one!\n\nNew paragraph.")]
    sentences = preprocess(pages)
    # Should split into three sentences ("This is a sentence.", "And another one!", "New paragraph.")
    texts = [s.text for s in sentences]
    assert "This is a sentence." in texts
    assert "And another one!" in texts
    assert "New paragraph." in texts
    # Page numbers should be 1 for all
    assert all(s.page_number == 1 for s in sentences)


def test_preprocess_preserves_scientific_notation_negation_and_unicode():
    pages = [
        (
            4,
            "We do not observe a change at 1.2e-3. "
            "Évaluation: accuracy is 95%.",
        )
    ]

    sentences = preprocess(pages)

    assert [sentence.text for sentence in sentences] == [
        "We do not observe a change at 1.2e-3.",
        "Évaluation: accuracy is 95%.",
    ]
    assert all(sentence.page_number == 4 for sentence in sentences)


def test_preprocess_normalizes_whitespace_and_keeps_paragraph_boundaries():
    sentences = preprocess([(2, "  First   sentence.\n\nSecond\t sentence! ")])

    assert [sentence.text for sentence in sentences] == [
        "First sentence.",
        "Second sentence!",
    ]
