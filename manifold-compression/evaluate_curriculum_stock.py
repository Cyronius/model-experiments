"""Uncompressed source baseline on the frozen new test, never a training target."""
import json
import torch
from curriculum_healing import OUT, SOURCE, assess
from local_projection_probe import load_source, save
from scale_data import load_tokenizer
from healing_data import chat_ids
from audit_rotation import cache_check


def main():
    torch.set_num_threads(4)
    data=json.loads((OUT/'data.json').read_text());tok=load_tokenizer(SOURCE)
    model=load_source().eval()
    count=sum(p.numel() for p in model.parameters());assert count==1881825088
    ids=torch.tensor([chat_ids(tok,data['test'][0]['messages'],generation=True)[:64]])
    gate=cache_check(model,ids);assert gate['numerical_pass']
    save({'parameters':count,'cache':gate,'purpose':'held-out correctness reference only; no outputs used for training or selection'},OUT/'stock-gates.json')
    result=assess(model,tok,data['test'],OUT/'stock-test.json');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
