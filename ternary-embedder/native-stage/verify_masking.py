"""Hardware checks for sequence isolation, partial batches and padding masks."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from fused_encoder import FusedEncoder,ROOT
from bench_encoder import pack
from diagnose import TOKENIZER
from transformers import AutoTokenizer


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True)
    texts=['The cat sat on the mat.','Quantum particles behave probabilistically.',
           'A brief sentence.','Retrieval should preserve semantic similarity across different wordings.']
    ids,mask=pack(tokenizer,texts,32)
    engine=FusedEncoder(32)
    try:
        original=engine(ids,mask)
        perm=torch.tensor([3,1,0,2])
        reordered=engine(ids[perm],mask[perm])
        torch.testing.assert_close(reordered,original[perm],atol=2e-6,rtol=2e-5)
        partial=engine(ids[:2],mask[:2])
        torch.testing.assert_close(partial,original[:2],atol=2e-6,rtol=2e-5)
        changed=ids.clone();changed[~mask]=1234
        padding=engine(changed,mask)
        cos=F.cosine_similarity(padding,original)
        print(json.dumps({'padding_cosines':cos.tolist(),
                          'padding_max_abs_error':float((padding-original).abs().max())}),flush=True)
        assert float(cos.min())>.9999
        rejected=False
        try:
            engine(ids,torch.zeros_like(mask))
        except ValueError:
            rejected=True
        assert rejected
        report={'batch_permutation_max_abs_error':float((reordered-original[perm]).abs().max()),
                'partial_batch_max_abs_error':float((partial-original[:2]).abs().max()),
                'changed_masked_token_ids_min_cosine':float(cos.min()),
                'empty_sequences_rejected':rejected,'finite':bool(torch.isfinite(original).all()),
                'max_unit_norm_error':float((original.norm(dim=-1)-1).abs().max())}
        (ROOT/'masking-result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:
        engine.close()


if __name__=='__main__':
    main()
