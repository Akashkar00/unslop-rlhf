import pytest

from unslop.metrics import distinct_n, slop_hits, style_summary


def test_slop_hits_counts_phrases_and_not_just_but():
    text = ("Let's delve into this rich tapestry. It's not just a tool, but a way of life. "
            "Moreover, it is vibrant.")
    assert slop_hits(text) == 5  # delve, tapestry, not-just-but, moreover, vibrant


def test_slop_hits_respects_word_boundaries():
    assert slop_hits("The pivotally placed sofa.") == 0


def test_distinct_n():
    assert distinct_n(["a b a b"], 2) == pytest.approx(2 / 3)
    assert distinct_n([""], 2) == 0.0


def test_style_summary():
    s = style_summary(["Moreover, this is fine — really.", "- a list\n- item"])
    assert s["n"] == 2
    assert s["list_or_header_share"] == 0.5
    # 6 + 5 whitespace tokens; the em dash counts as one.
    assert s["slop_per_1k_words"] == pytest.approx(1000 / 11)
    assert s["em_dash_per_1k_words"] == pytest.approx(1000 / 11)
