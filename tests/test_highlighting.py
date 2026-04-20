from src.dashboard.highlighting import render_highlighted_transcript


def _segs(*words):
    return [{"text": w, "start_time": float(i), "end_time": float(i + 1), "confidence": 1.0}
            for i, w in enumerate(words)]


VOCAB = frozenset({"urgent", "wire transfer", "bank account", "verify"})


def test_empty_transcript_returns_empty():
    assert render_highlighted_transcript([], [], VOCAB) == ""


def test_plain_word_no_highlight():
    html = render_highlighted_transcript(_segs("hello", "there"), [], VOCAB)
    assert "hello" in html
    assert "<mark" not in html


def test_model_only_keyword_gets_yellow():
    html = render_highlighted_transcript(_segs("please", "confirm"), ["confirm"], VOCAB)
    assert 'class="kw-model"' in html
    assert "confirm" in html


def test_vocab_only_single_word_gets_blue():
    html = render_highlighted_transcript(_segs("please", "verify"), [], VOCAB)
    assert 'class="kw-vocab"' in html
    assert "verify" in html


def test_both_keyword_and_vocab_gets_red():
    html = render_highlighted_transcript(_segs("please", "urgent"), ["urgent"], VOCAB)
    assert 'class="kw-both"' in html
    assert "urgent" in html


def test_case_insensitive_matching():
    html = render_highlighted_transcript(_segs("URGENT", "call"), ["urgent"], VOCAB)
    assert 'class="kw-both"' in html


def test_multi_word_vocab_phrase():
    html = render_highlighted_transcript(
        _segs("please", "wire", "transfer", "now"), [], VOCAB
    )
    assert 'class="kw-vocab"' in html
    assert "wire transfer" in html


def test_multi_word_phrase_with_keyword_overlap():
    html = render_highlighted_transcript(
        _segs("wire", "transfer", "funds"), ["wire transfer"], VOCAB
    )
    assert 'class="kw-both"' in html
