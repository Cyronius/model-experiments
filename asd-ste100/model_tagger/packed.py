"""Compact memory-mapped examples and vectorized training batches."""
from pathlib import Path
import numpy as np
import torch
from tagger import Tags, RULES, TRANSFORMS, source_units, checker_features

KINDS = ('original', 'messy', 'rule', 'local_rule', 'reviewed', 'doc_teacher')   # append only: packed data stores the index
INDEX_COLUMNS = ('offset_i32', 'pieces', 'units', 'mode', 'terminal', 'kind')


def pack(row, labels, casing, terminal, tags, lexicon):
    n, m = len(row['source_ids']), len(row['source_spans'])
    units = source_units(row)
    transforms = []
    for i in range(m):
        transforms.append(sum(1 << t for t, name in enumerate(TRANSFORMS) if tags.transformed(units, i, name) is not None))
    features = checker_features(row['src'], row, lexicon, row['mode']).to(torch.int32)
    bits = (features * (2 ** torch.arange(len(RULES)))[None,:]).sum(1).tolist()
    arrays = [row['source_ids'], row['source_case'], row['source_slot_refs'],
              [s[0] for s in row['source_spans']], [s[1] for s in row['source_spans']],
              [u[0] for u in units], [u[1] for u in units], labels, casing, bits, transforms]
    data = np.concatenate([np.asarray(a, dtype='<i4') for a in arrays])
    assert len(data) == 4*n + 7*m + 3
    return data, (n, m, row['mode_id'], int(terminal), KINDS.index(row['kind']))


def unpack(data, index):
    offset, n, m, mode, terminal, kind = map(int,index)
    cursor = offset
    result = {}
    for key, size in [('ids',n),('cases',n),('refs',n),('starts',m),('ends',m),
                      ('unit_ids',m),('unit_cases',m),('labels',m+1),('label_cases',m+1),
                      ('feature_bits',n+1),('transform_bits',m)]:
        result[key] = data[cursor:cursor+size]
        cursor += size
    result.update(mode=mode, terminal=terminal, kind=kind)
    return result


class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, directory, split):
        self.directory, self.split = str(directory), split
        self.index = self.data = None
        self.length = len(np.load(Path(directory)/f'{split}.index.npy',mmap_mode='r'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.index is None:
            self.index = np.load(Path(self.directory)/f'{self.split}.index.npy',mmap_mode='r')
            self.data = np.memmap(Path(self.directory)/f'{self.split}.data.i32',dtype='<i4',mode='r')
        return unpack(self.data,self.index[index])

    def __getstate__(self):
        return dict(self.__dict__,index=None,data=None)


class Batches:
    def __init__(self,index,batch_size,seed,epoch,shuffle=True,start=0,local_cap=None):
        rng=np.random.default_rng(seed+epoch)
        lengths=index[:,1]+1
        eligible=np.ones(len(index),dtype=bool)
        if local_cap is not None:
            if local_cap < 1:raise ValueError('local cap must be positive')
            local=np.flatnonzero(index[:,5]==KINDS.index('local_rule'))
            if len(local)>local_cap:
                eligible[local]=False
                eligible[rng.choice(local,local_cap,replace=False)]=True
        self.batches=[]
        for low,high in ((0,16),(16,32),(32,64),(64,129),(129,257)):
            positions=np.flatnonzero(eligible&(lengths>low)&(lengths<=high))
            if shuffle:rng.shuffle(positions)
            self.batches.extend(positions[i:i+batch_size].tolist() for i in range(0,len(positions),batch_size))
        if shuffle:rng.shuffle(self.batches)
        self.start=start

    def __len__(self):return len(self.batches)-self.start
    def __iter__(self):return iter(self.batches[self.start:])


class Collator:
    def __init__(self,tokenizer):
        tags=Tags(tokenizer)
        self.vocab,self.tag_size,self.transform_offset=tags.vocab,tags.size,tags.transform_offset
        self.pad=tokenizer.output_ids['PAD']
        self.ordinary=torch.tensor([i not in tokenizer.control_ids|tokenizer.slot_ids for i in range(self.vocab)])
        self.slot_ids=torch.tensor(sorted(tokenizer.slot_ids))

    def __call__(self,rows):
        batch=len(rows);pieces=max(len(r['ids']) for r in rows)+1;spans=max(len(r['unit_ids']) for r in rows)+1
        ids=torch.full((batch,pieces),self.pad,dtype=torch.long)
        cases=torch.zeros_like(ids);refs=torch.zeros_like(ids);feature_bits=torch.zeros_like(ids)
        padding=torch.ones_like(ids,dtype=torch.bool)
        starts=torch.zeros((batch,spans),dtype=torch.long);ends=torch.ones_like(starts)
        actions=torch.zeros_like(starts,dtype=torch.bool);unit_ids=torch.full_like(starts,-1)
        labels=torch.full_like(starts,-100);label_cases=torch.zeros_like(starts);transform_bits=torch.zeros_like(starts)
        terminal=torch.zeros(batch,dtype=torch.bool);kind=torch.zeros(batch,dtype=torch.long)
        active=self.ordinary[None,:].expand(batch,-1).clone()
        for i,row in enumerate(rows):
            n,m=len(row['ids']),len(row['unit_ids'])
            ids[i,0]=row['mode'];ids[i,1:n+1]=torch.tensor(row['ids'],dtype=torch.long)
            cases[i,1:n+1]=torch.tensor(row['cases'],dtype=torch.long)
            ref=torch.tensor(row['refs'],dtype=torch.long);refs[i,1:n+1]=ref+1
            active[i,ref[ref>=0].unique()]=True
            padding[i,:n+1]=False
            starts[i,1:m+1]=torch.tensor(row['starts'],dtype=torch.long)+1
            ends[i,1:m+1]=torch.tensor(row['ends'],dtype=torch.long)+1
            unit_ids[i,1:m+1]=torch.tensor(row['unit_ids'],dtype=torch.long)
            actions[i,:m+1]=True
            labels[i,:m+1]=torch.tensor(row['labels'],dtype=torch.long)
            label_cases[i,:m+1]=torch.tensor(row['label_cases'],dtype=torch.long)
            feature_bits[i,:n+1]=torch.tensor(row['feature_bits'],dtype=torch.long)
            transform_bits[i,1:m+1]=torch.tensor(row['transform_bits'],dtype=torch.long)
            terminal[i]=bool(row['terminal']);kind[i]=row['kind']
        position=torch.arange(pieces)[None,None,:]
        pooling=((position>=starts[:,:,None])&(position<ends[:,:,None])).float()
        pooling/= (ends-starts).clamp_min(1)[:,:,None]
        pooling*=actions[:,:,None]
        legal=torch.zeros((batch,spans,self.tag_size),dtype=torch.bool)
        can_keep=(unit_ids>=0)&(unit_ids<self.vocab)&active.gather(1,unit_ids.clamp(0,self.vocab-1))
        legal[:,:,0]=can_keep;legal[:,0,0]=True
        legal[:,1:,1]=True
        legal[:,1:,2:2+self.vocab]=active[:,None,:]
        legal[:,:,2+self.vocab:self.transform_offset]=active[:,None,:]&(can_keep|torch.arange(spans)[None,:].eq(0))[:,:,None]
        legal[:,:,self.transform_offset:]=(transform_bits[:,:,None]&(2**torch.arange(len(TRANSFORMS)))[None,None,:]).ne(0)
        legal&=actions[:,:,None];legal[:,:,0]|=~actions
        features=(feature_bits[:,:,None]&(2**torch.arange(len(RULES)))[None,None,:]).ne(0).float()
        return {'ids':ids,'cases':cases,'refs':refs,'padding':padding,'spans':pooling,'actions':actions,
                'features':features,'legal':legal,'labels':labels,'label_cases':label_cases,
                'terminal':terminal,'kind':kind}
