"""Source-conditioned absorbing-mask diffusion over discrete edit-action canvases.

Forward process independently masks reference actions at sampled noise levels.
The reverse sampler progressively reveals actions, keeping revealed actions fixed.
This is action diffusion, not token-canvas diffusion or another prose rewrite loop.
Only validated decoded text may be returned; latent action states can be invalid.
"""
import math
import torch
from torch import nn
from tagger import Tagger, Config, Tags, collate, source_units, EncodingError, check, errors
from safe_editor import validate
from copy_fidelity import preserved


class ActionDiffusion(Tagger):
    def __init__(self, tokenizer, config=Config()):
        super().__init__(tokenizer, config)
        self.mask_action=Tags(tokenizer).size
        self.draft_action=nn.Embedding(self.mask_action+1,config.width)
        self.noise_level=nn.Linear(2,config.width)
        # The initial all-mask model equals its warm-started tagger exactly.
        nn.init.zeros_(self.draft_action.weight)
        nn.init.zeros_(self.noise_level.weight);nn.init.zeros_(self.noise_level.bias)

    def forward(self,batch,noisy_actions,noise):
        ids=batch['ids']
        if ids.shape[1]>self.config.max_positions:raise ValueError('position capacity exceeded')
        if noisy_actions.shape!=batch['actions'].shape or noise.shape!=(len(ids),):raise ValueError('noise conditioning shape')
        if ((noise<0)|(noise>1)).any():raise ValueError('noise must be within [0,1]')
        positions=torch.arange(ids.shape[1],device=ids.device)[None,:]
        assignment=batch['spans'].gt(0).transpose(1,2).to(self.token.weight.dtype)
        draft=torch.bmm(assignment,self.draft_action(noisy_actions))
        time=self.noise_level(torch.stack((noise,1-noise),-1))[:,None,:]
        hidden=self.norm(self.token(ids)+self.case(batch['cases'])+self.slot_ref(batch['refs'])+
                         self.position(positions)+self.violation(batch['features'])+draft+time)
        hidden=self.encoder(hidden,src_key_padding_mask=batch['padding'])
        pooled=torch.bmm(batch['spans'],hidden)
        return self.tag_head(pooled).masked_fill(~batch['legal'],float('-inf')),self.case_head(pooled)


def corrupt(labels,valid,mask_action,generator=None):
    """Bernoulli absorbing masks; ensure a nonempty supervised mask per example."""
    noise=.05+.95*torch.rand(len(labels),generator=generator,device=labels.device)
    masked=(torch.rand(labels.shape,generator=generator,device=labels.device)<noise[:,None])&valid
    for row in range(len(labels)):
        if not masked[row].any():masked[row,valid[row].nonzero()[0,0]]=True
    noisy=labels.clamp_min(0).clone();noisy[masked]=mask_action
    return noisy,noise,masked


def reveal_count(remaining,steps_left):
    return math.ceil(remaining/steps_left)


@torch.no_grad()
def rewrite(model,tokenizer,lexicon,text,mode='PROC',max_passes=5,edit_margin=0.):
    if not 1<=max_passes<=5 or not math.isfinite(edit_margin) or edit_margin<0:raise ValueError('invalid sampler settings')
    model.eval();device=next(model.parameters()).device;tags=Tags(tokenizer)
    try:encoded=tokenizer.encode_source(text,mode)
    except EncodingError as exc:return dict(text=text,passes=0,resolved=False,reason=str(exc),trace=[])
    units=source_units(encoded);batch=collate([dict(encoded,src=text,mode=mode)],tokenizer,lexicon,device)
    actions=torch.full(batch['actions'].shape,model.mask_action,device=device,dtype=torch.long)
    casing=torch.zeros_like(actions)
    trace=[];last_valid=None
    # Slot anchors are fixed KEEP in the deployed tagger and in this sampler.
    for i,(value,_) in enumerate(units,1):
        if value in tokenizer.slot_ids:actions[0,i]=tags.KEEP
    for step in range(max_passes):
        unknown=actions.eq(model.mask_action)&batch['actions']
        noise=unknown.sum(-1)/batch['actions'].sum(-1)
        scores,case_scores=model(batch,actions,noise)
        scores=scores[0].clone();scores[:,tags.KEEP]+=edit_margin
        chosen=scores.argmax(-1);confidence=scores.softmax(-1).amax(-1)
        positions=unknown[0].nonzero().flatten()
        if len(positions):
            n=reveal_count(len(positions),max_passes-step)
            selected=positions[confidence[positions].argsort(descending=True,stable=True)[:n]]
            actions[0,selected]=chosen[selected];casing[0,selected]=case_scores[0,selected].argmax(-1)
        # For diagnostics, fill still-masked positions with current predictions.
        # These provisional actions are never fed back as accepted prose.
        complete=torch.where(actions[0].eq(model.mask_action),chosen,actions[0])
        case=torch.where(actions[0].eq(model.mask_action),case_scores[0].argmax(-1),casing[0])
        candidate=None;issues=[]
        try:
            output=tags.apply(units,complete.tolist(),case.tolist(),encoded['slots'])
            values,cases=map(list,zip(*output))
            candidate=text if complete.eq(tags.KEEP).all() else tokenizer.detokenize(values,encoded['slots'],cases)
            issues=validate(text,candidate)
            if not preserved(candidate,encoded['slots']):issues.append('original_slot_fidelity')
            if not issues:last_valid=candidate
        except EncodingError as exc:issues=[str(exc)]
        trace.append(dict(step=step+1,masked_remaining=int(actions.eq(model.mask_action).sum()),proposal=candidate,rejected=issues))
    result=last_valid if last_valid is not None else text
    clean=last_valid is not None and not errors(check(result,lexicon,mode))
    return dict(text=result,passes=max_passes,resolved=clean,reason='checker_clean' if clean else 'unresolved',trace=trace)
