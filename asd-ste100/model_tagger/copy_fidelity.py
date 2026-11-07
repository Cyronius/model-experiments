"""Independent exact-surface copy counts; output tokenization may merge adjacent names."""
from collections import Counter
import re


def counts(text,slots):
    surfaces=sorted({s['surface'] for s in slots},key=lambda x:(-len(x),x))
    if not surfaces:return Counter()
    if any(not s for s in surfaces):raise ValueError('empty copy-slot surface')
    alternatives=[]
    for surface in surfaces:
        left=r'(?<!\w)' if re.match(r'\w',surface[0]) else ''
        right=r'(?!\w)' if re.match(r'\w',surface[-1]) else ''
        alternatives.append(left+re.escape(surface)+right)
    return Counter(m.group() for m in re.finditer('|'.join(alternatives),text))


def preserved(text,slots):
    return counts(text,slots)==Counter({s['surface']:s['count'] for s in slots})
