"""Explicit phrase equivalences reviewed for this software-writing experiment.

Apply to prose token sequences only, never inside protected literals. These are
not a synonym dictionary or a general semantic-equivalence classifier.
"""
PHRASES = {
    ('in','order','to'):('to',),
    ('prior','to'):('before',),
    ('subsequent','to'):('after',),
    ('in','the','event','that'):('if',),
    ('due','to','the','fact','that'):('because',),
    ('at','this','point','in','time'):('at','this','time'),
}
ORDERED=sorted(PHRASES,key=lambda x:(-len(x),x))


def canonical_words(words):
    result=[];i=0
    while i<len(words):
        for source in ORDERED:
            if tuple(words[i:i+len(source)])==source:
                result.extend(PHRASES[source]);i+=len(source);break
        else:
            result.append(words[i]);i+=1
    return result
