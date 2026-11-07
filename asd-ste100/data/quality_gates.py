"""Conservative deterministic data-quality gates, separate from the STE checker.

Hard exclusions target literal-copy violations and specific damaged-target patterns.
Context-dependent fragments are review flags, never automatic exclusions.
"""
from __future__ import annotations

from collections import Counter
import re

INLINE = re.compile(r'```[\s\S]*?```|`[^`\n]*`')
HTML = re.compile(r"""</?[A-Za-z][\w:-]*(?:\s+[A-Za-z_:][\w:.-]*(?:\s*=\s*(?:"[^"\n]*"|'[^'\n]*'|[^\s<>"'=]+))?)*\s*/?>""")
CALL = re.compile(r'\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*\(')
VB = re.compile(r'\bIf\s+[^\n]*?\bThen\b[^\n]*?\bEnd\s+If\b', re.I)
MATH_VARIABLE = re.compile(r'\b([A-Za-z])\s*(?=[≤≥≠=^]|[<>]\s*(?:-?\d|[A-Za-z]\b)|[+*/-]\s*(?:-?\d|[A-Za-z]\b))')
QUOTED = re.compile(r'“[^”\n]*”|‘[^’\n]*’|«[^»\n]*»|"(?:\\.|[^"\\\n])*"|(?<!\w)\x27(?:\\.|[^\x27\\\n])*\x27(?!\w)')
CONTRACTION = re.compile(r"\b(?:it|he|she|that|there|here|what|who|where|how|when|why|let)['’]s\b|\b[A-Za-z]+['’](?:re|ve|ll|d|m)\b|\b[A-Za-z]+n['’]t\b", re.I)
TRUNCATED = re.compile(r'(?:\.{3}|…)?\s*[\[(]\s*(?:truncated|text truncated|cut off)\s*[\])]\s*[.!]?\s*$', re.I)
EMPTY_MARKUP = re.compile(r'`[ \t]*`\s+(?:HTML\s+|XML\s+)?(?:tags?|elements?)\b', re.I)
DANGLING_STEP = re.compile(r'(?:^|[.!?]\s+|\n)\s*Step\s+\d+[A-Za-z]?\s*[.:]?\s*$', re.I)
COPULA = re.compile(r'(?:^|[.!?]\s+)((?:the|a|an|this|that|these|those|your|our|its|their)\b[^.!?\n]*?\b(?:is|are|was|were|will be|must be|should be))\s*[.:]?\s*$', re.I)


def overlap(start, end, spans):
    return any(start < b and end > a for a, b in spans)


def mask(text, spans):
    chars = list(text)
    for a, b in spans:
        chars[a:b] = ' ' * (b - a)
    return ''.join(chars)


def code_spans(text):
    """Recognize inline code, one-line VB blocks and balanced no-space call syntax.

    This intentionally is not a complete parser for arbitrary unfenced programs.
    Prose parentheses such as 'function (optional)' are not call syntax.
    """
    spans = [m.span() for m in INLINE.finditer(text)]
    spans += [m.span() for m in VB.finditer(text) if not overlap(*m.span(), spans)]
    for match in CALL.finditer(text):
        if overlap(*match.span(), spans):
            continue
        depth, quote, escaped, end = 1, None, False, match.end()
        while end < len(text) and depth:
            char = text[end]
            if quote:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in ('"', "'"):
                quote = char
            elif char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            end += 1
        if not depth:
            spans.append((match.start(), end))
    return sorted(spans)


def literal_signatures(text):
    code = code_spans(text)
    tags = [m.span() for m in HTML.finditer(text) if not overlap(*m.span(), code)]
    pod = [m.span() for m in re.finditer(r'\b[BLCEFSIXZ]<[^<>\n]*>', text)]
    prose = mask(text, code + tags + pod)
    return {'code': Counter(text[a:b] for a, b in code),
            'markup': Counter(text[a:b] for a, b in tags),
            'math_variables': Counter(m.group(1) for m in MATH_VARIABLE.finditer(prose))}


def unsafe_changes(source, target):
    a, b = literal_signatures(source), literal_signatures(target)
    issues = []
    for key, rule in [('code', 'code_literal_changed'), ('markup', 'markup_changed'),
                      ('math_variables', 'math_variable_changed')]:
        if a[key] != b[key]:
            issues.append({'rule': rule, 'action': 'exclude',
                           'source_only': list((a[key] - b[key]).elements()),
                           'target_only': list((b[key] - a[key]).elements())})
    return issues


def bad_merge_marker(source, target, noise):
    if noise != 'merge_sentences':
        return []
    # Only reject a known synthetic transformation of a numeric list marker.
    for match in re.finditer(r'(?<![\w.])(\d{1,3})\.\s+(?=[A-Za-z])', target):
        start, end = match.span()
        prefix = target[:start]
        marker_context = re.search(r'(?:^|\n)\s*(?:[*#>-]+\s*)?(?:(?:Step|Part|Section|Item)\s+)?$', prefix, re.I) or re.search(r'\b(?:Step|Part|Section|Item)\s+$', prefix, re.I)
        tail = target[end:]
        possible_tails = (tail, tail[:1].lower() + tail[1:])
        if marker_context and any(source == target[:start] + match.group(1) + ' and ' + tail for tail in possible_tails):
            return [{'rule': 'list_marker_merged', 'action': 'exclude', 'marker': match.group()}]
    return []


def target_issues(text):
    issues = []
    code = code_spans(text)
    quoted = [m.span() for m in QUOTED.finditer(text) if not overlap(*m.span(), code)]
    # Contractions used as quoted examples or inside code are data, not target prose.
    prose = mask(text, code + quoted)
    for match in CONTRACTION.finditer(prose):
        issues.append({'rule': 'target_contraction', 'action': 'exclude', 'text': text[match.start():match.end()]})
    if TRUNCATED.search(prose):
        issues.append({'rule': 'explicit_truncation', 'action': 'exclude'})
    if EMPTY_MARKUP.search(text):
        issues.append({'rule': 'empty_markup_reference', 'action': 'exclude'})
    # Replace code/quotes with nonempty nouns, so a valid clause ending in code
    # does not look as though it ended at its copula after masking.
    complete = text
    for a, b in sorted(code + quoted, reverse=True):
        complete = complete[:a] + 'LITERAL' + complete[b:]
    match = COPULA.search(complete)
    if match and not re.search(r'\b(?:what|where|who|which|why|how|whether|if)\b|^the\s+(?:more|less|fewer|[A-Za-z]+er)\b|\bas\s+(?:it|they|he|she|this|that)\b', match.group(1), re.I):
        multiple = len(re.findall(r'\b(?:is|are|was|were|be)\b', match.group(1), re.I)) > 1
        issues.append({'rule': 'possible_unfinished_copula' if multiple else 'unfinished_copula',
                       'action': 'review' if multiple else 'exclude', 'text': match.group(1)})
    if DANGLING_STEP.search(complete):
        issues.append({'rule': 'dangling_step_heading', 'action': 'review'})
    if re.search(r'`[ \t]*`', text) and not EMPTY_MARKUP.search(text):
        issues.append({'rule': 'empty_inline_literal', 'action': 'review'})
    if re.search(r'(?:\.{3}|…)\s*$', prose):
        issues.append({'rule': 'trailing_ellipsis', 'action': 'review'})
    return issues


def inspect_pair(row):
    return (unsafe_changes(row['src'], row['ste'])
            + bad_merge_marker(row['src'], row['ste'], row.get('noise'))
            + target_issues(row['ste']))
