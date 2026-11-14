"""Offline text-to-vector CLI. Run under `dibs run npu -- ...`."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from fused_encoder import FusedEncoder,ROOT,CHECKPOINT

from transformers import AutoTokenizer


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('texts',nargs='+')
    p.add_argument('--tokenizer',type=Path,default=ROOT.parent/'tokenizer')
    p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    p.add_argument('--cap',type=int,choices=(32,128),default=128,help='Truncation cap, including special tokens')
    p.add_argument('--output',type=Path,help='Save float32 vectors as .npy; otherwise print JSON')
    args=p.parse_args()
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    engine=FusedEncoder(args.cap,checkpoint=args.checkpoint)
    vectors=[]
    try:
        start=time.perf_counter()
        for offset in range(0,len(args.texts),128//args.cap):
            enc=tokenizer(args.texts[offset:offset+128//args.cap],padding='max_length',
                          truncation=True,max_length=args.cap,return_tensors='pt')
            vectors.append(engine(enc['input_ids'],enc['attention_mask'].bool()).numpy())
        elapsed=time.perf_counter()-start
        result=np.concatenate(vectors)
        if args.output:
            np.save(args.output,result)
        else:
            print(json.dumps(result.tolist()))
        print(json.dumps({'texts':len(result),'dimensions':result.shape[1],
                          'seconds_excluding_model_load':elapsed,'cap':args.cap}),file=sys.stderr)
    finally:
        engine.close()


if __name__=='__main__':
    main()
