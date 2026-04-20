from typing import List, Dict, FrozenSet


CSS = """
<style>
.kw-both  { background: #ff4b4b; color: white; border-radius: 3px; padding: 1px 3px; }
.kw-model { background: #ffd700; color: black; border-radius: 3px; padding: 1px 3px; }
.kw-vocab { border-bottom: 2px solid #4b9dff; padding-bottom: 1px; }
</style>
"""


def render_highlighted_transcript(
    segments: List[Dict],
    top_keywords: List[str],
    vocabulary: FrozenSet[str],
) -> str:
    if not segments:
        return ""

    keywords = {kw.lower() for kw in top_keywords}
    vocab = {v.lower() for v in vocabulary}

    # Collect max phrase length from vocab for sliding window
    max_phrase_len = max((len(p.split()) for p in vocab), default=1)

    # Tokenize segments into (word_lower, original_word) pairs
    tokens = [(s["text"].strip().lower(), s["text"].strip()) for s in segments]

    # Build list of (span_start, span_end, class) for multi-word matches first
    n = len(tokens)
    covered = [False] * n  # prevent double-tagging
    spans: List[tuple] = []  # (start_idx, end_idx_exclusive, css_class)

    # Pass 1: multi-word vocab phrases (longest match first)
    for length in range(max_phrase_len, 1, -1):
        for i in range(n - length + 1):
            if any(covered[i:i + length]):
                continue
            phrase = " ".join(t[0] for t in tokens[i:i + length])
            if phrase in vocab:
                in_kw = phrase in keywords
                css = "kw-both" if in_kw else "kw-vocab"
                spans.append((i, i + length, css))
                for j in range(i, i + length):
                    covered[j] = True

    # Pass 2: single-word matches
    for i, (word_lower, _) in enumerate(tokens):
        if covered[i]:
            continue
        in_kw = word_lower in keywords
        in_vocab = word_lower in vocab
        if in_kw and in_vocab:
            spans.append((i, i + 1, "kw-both"))
            covered[i] = True
        elif in_kw:
            spans.append((i, i + 1, "kw-model"))
            covered[i] = True
        elif in_vocab:
            spans.append((i, i + 1, "kw-vocab"))
            covered[i] = True

    # Build span lookup: token_idx -> (end_idx, css_class)
    span_starts = {s[0]: s for s in spans}

    # Render HTML
    parts = [CSS]
    i = 0
    while i < n:
        if i in span_starts:
            start, end, css = span_starts[i]
            text = " ".join(t[1] for t in tokens[start:end])
            parts.append(f'<mark class="{css}">{text}</mark>')
            i = end
        else:
            parts.append(tokens[i][1])
            i += 1
        if i < n:
            parts.append(" ")

    return "".join(parts)
