import json
from corpus import ranges, sections, passage, divide, cite, digest


def fixture(text, unit_key="unit-a", pointer=None):
    unit={"source_key":"file-a","sha256":digest(text),"json_pointer":pointer,"source_context":{"evidence":"hearsay"}}
    source={"sha256":digest(text),"path":"raw/a.md","snapshot":"snapshot.txt","layer":"raw"}
    return unit,source


def test_every_character_survives_hard_splits():
    text="x"*8000+"\n\n| col | value |\n|---|---|\n"+"| a | 4 |\n"*1000
    chunks=list(ranges(text))
    assert "".join(text[a:b] for a,b in chunks)==text
    assert max(b-a for a,b in chunks)<=3200


def test_original_headings_are_citable_but_code_comments_are_not_sections():
    text="# Qwen3 workload\n\n```python\n# not a heading\n```\n\n## Linux only\n\nKernel ash took 9 ms."
    unit,source=fixture(text)
    start=text.index("Kernel")
    case=passage("unit-a",unit,text,start,len(text))
    assert "# Qwen3 workload" in case['text']
    assert all(text[a:b]!="# not a heading" for a,b,_ in sections(text))
    record=cite({"statement":"For the Qwen3 workload on Linux, ash took 9 ms.","quote":"Kernel ash took 9 ms.","conditions":["Qwen3 workload","Linux only"]},case,unit,source,text,"run")
    assert record['citation_valid']
    assert record['review_status']=='proposed' and not record['semantic_support_verified']


def test_reassembled_context_cannot_create_an_original_quote():
    text="# Title\n\n"+"omitted "*600+"\n\nTarget assertion."
    unit,source=fixture(text)
    start=text.index('Target assertion.')
    case=passage('unit-a',unit,text,start,len(text))
    record=cite({'statement':'x','quote':'# Title\n\nTarget assertion.','conditions':[]},case,unit,source,text,'run')
    assert not record['citation_valid']


def test_condition_must_have_an_allowed_original_span():
    text="Kernel ash took 9 ms."
    unit,source=fixture(text);case=passage('unit-a',unit,text,0,len(text))
    record=cite({'statement':'ash always works','quote':text,'conditions':['target_passage']},case,unit,source,text,'run')
    assert not record['citation_valid']


def test_json_coordinates_refer_to_decoded_field_not_raw_serialization():
    value='Kernel "ash" took\n9 ms.'
    raw=json.dumps([{'body':value}])
    assert value not in raw
    unit,source=fixture(value,pointer='/0/body');source['sha256']=digest(raw)
    case=passage('unit-a',unit,value,0,len(value))
    record=cite({'statement':'ash took 9 ms','quote':value,'conditions':[]},case,unit,source,value,'run')
    assert record['citation_valid'] and record['json_pointer']=='/0/body'
    assert record['source_sha256']==digest(raw) and record['source_spans'][0]['char_end']==len(value)


def test_identical_text_in_different_files_retains_both_origins():
    text='Kernel ash took 9 ms.'
    unit,source=fixture(text)
    a=passage('unit-a',unit,text,0,len(text));b=passage('unit-b',unit,text,0,len(text))
    claim={'statement':'ash took 9 ms','quote':text,'conditions':[]}
    assert cite(claim,a,unit,source,text,'run')['id']!=cite(claim,b,unit,source,text,'run')['id']


def test_failed_outputs_can_split_without_losing_target_text():
    text=('Claim 1.\nClaim 2.\n'*100)
    unit,_=fixture(text);parent=passage('unit-a',unit,text,0,len(text))
    children=divide(parent,unit,text)
    assert len(children)==2
    assert ''.join(text[c['source_start']:c['source_end']] for c in children)==text
    assert all(c['parent']==parent['id'] for c in children)
    parent['depth']=3
    assert not divide(parent,unit,text)
