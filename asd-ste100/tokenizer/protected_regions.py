"""Conservative additional literal regions shared by encoding and inference guards."""
import re
from quality_gates import code_spans, HTML, QUOTED
from text_utils import Token, tokenize

EXTRA = re.compile(r'(?<!\w)(?:C\+\+|C#|\.NET)(?!\w)|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$')
BACKTICKS = re.compile(r'`+')


def regions(text):
    spans = [m.span() for m in re.finditer(r'\b[A-Za-z][A-Za-z0-9+.-]*://[^\s<>`]+', text)]
    spans += code_spans(text) + [m.span() for pattern in (HTML, QUOTED, EXTRA) for m in pattern.finditer(text)]
    cursor = 0
    while (match := BACKTICKS.search(text, cursor)) is not None:
        marker = match.group()
        close = re.search(r'(?<!`)' + re.escape(marker) + r'(?!`)', text[match.end():])
        end = match.end() + close.end() if close else len(text)
        spans.append((match.start(), end))
        cursor = end
    # Preserve paired Markdown emphasis as a unit; unpaired marks are rejected by
    # the output guard if detokenization would change their spelling/spacing.
    spans += [m.span() for m in re.finditer(r'\*\*[^\n]+?\*\*|__[^\n]+?__', text)]
    spans += [(t.start, t.end) for t in tokenize(text) if t.protected]
    merged = []
    for a, b in sorted(spans):
        if merged and a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def tokenize_protected(text):
    result, cursor = [], 0
    for a, b in regions(text):
        result.extend(Token(t.text, t.start + cursor, t.end + cursor, t.protected)
                      for t in tokenize(text[cursor:a]))
        result.append(Token(text[a:b], a, b, True))
        cursor = b
    result.extend(Token(t.text, t.start + cursor, t.end + cursor, t.protected)
                  for t in tokenize(text[cursor:]))
    return result
