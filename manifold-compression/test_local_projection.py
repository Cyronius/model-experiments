import copy

import torch

from audit_rotation import cache_check, norm_gain
from local_projection_models import build, fit_rms, norms, rms_error, stored_scalars
from projection_models import project_residual_width, install_output_matrix, projected_output_matrix
from test_projection_models import tiny_model
from local_projection_probe import grade_program,grade_case


def bases(d=32):
    torch.manual_seed(184)
    return [torch.linalg.qr(torch.randn(d,d,dtype=torch.float64)).Q.float() for _ in range(2)]


def test_local_fullwidth_preserves_logits_tying_and_cache():
    model=tiny_model('qwen',True)
    ids=torch.randint(0,101,(1,13))
    expected=model(ids,use_cache=False).logits
    qs=bases()
    build(model,qs,[1.]*5,local=True)
    torch.testing.assert_close(model(ids,use_cache=False).logits,expected,atol=3e-6,rtol=3e-5)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert cache_check(model,ids)['numerical_pass']
    assert len(norms(model))==5


def test_shared_rectangular_matches_existing_builder():
    source=tiny_model('qwen',True);old=copy.deepcopy(source)
    p=bases()[0][:,:24];scales=[1.03,1.04,1.02,1.05,1.06]
    gain=norm_gain(old.model.norm)
    project_residual_width(old,p,scales)
    install_output_matrix(old,projected_output_matrix(gain,p,scales[-1]))
    build(source,[p,p],scales)
    ids=torch.randint(0,101,(1,13))
    torch.testing.assert_close(source(ids,use_cache=False).logits,old(ids,use_cache=False).logits,atol=3e-6,rtol=3e-5)


def test_local_rectangular_state_dict_reload_and_accounting():
    source=tiny_model('qwen',True);other=copy.deepcopy(source)
    ps=[q[:,:24] for q in bases()];scales=[1.04]*5
    build(source,ps,scales,local=True);build(other,ps,scales,local=True)
    other.load_state_dict(source.state_dict(),strict=True)
    ids=torch.randint(0,101,(1,13))
    torch.testing.assert_close(other(ids,use_cache=False).logits,source(ids,use_cache=False).logits,atol=0,rtol=0)
    shared=build(tiny_model('qwen',True),[ps[0],ps[0]],scales)
    assert stored_scalars(source)-stored_scalars(shared)==24*24
    assert cache_check(source,ids)['numerical_pass']


def test_state_dependent_rms_fit_generalizes_to_new_states():
    torch.manual_seed(82)
    z=torch.randn(4000,24)
    missing=(.2+z[:,0:1].square())*torch.randn(4000,8)
    x=torch.cat([z,missing],-1);p=torch.eye(32)[:,:24]
    scale,predictor=fit_rms(x[:3000],p)
    errors=rms_error(x[3000:],p,scale,predictor)
    assert errors['predicted_mse']<errors['constant_mse']*.8


def test_execution_grader_detects_mutation_and_wrong_answer():
    tests=[({'intervals':[[3,4],[1,2]]},[[1,2],[3,4]])]
    good='def main_solution(intervals):\n    return sorted(intervals)'
    mutates='def main_solution(intervals):\n    intervals.sort()\n    return intervals'
    assert grade_program(good,tests,True)['correct']
    assert not grade_program(mutates,tests,True)['correct']
    assert not grade_program('def main_solution(intervals):\n    return []',tests,True)['correct']


def test_dynamic_norm_complete_safetensors_reload(tmp_path):
    from safetensors.torch import save_model,load_model
    p=bases()[0][:,:24]
    scale,predictor=fit_rms(torch.randn(200,32),p)
    def make():
        return build(tiny_model('qwen',True),[p,p],[scale]*5,predictors=[predictor]*5)
    source=make();ids=torch.randint(0,101,(1,11))
    expected=source(ids,use_cache=False).logits
    save_model(source,str(tmp_path/'model.safetensors'))
    restored=make()
    with torch.no_grad():
        for param in restored.parameters():param.zero_()
        for norm in norms(restored):
            norm.coef.zero_();norm.mean.zero_();norm.std.zero_();norm.scale.zero_()
    missing,unexpected=load_model(restored,str(tmp_path/'model.safetensors'),strict=True)
    assert not missing and not unexpected
    torch.testing.assert_close(restored(ids,use_cache=False).logits,expected,atol=0,rtol=0)


def test_local_mlp_update_projects_directly_to_destination():
    model=tiny_model('qwen',True)
    original_mlp=copy.deepcopy(model.model.layers[0].mlp)
    gain=norm_gain(model.model.layers[0].post_attention_layernorm)
    ps=[q[:,:24] for q in bases()]
    scale=1.04
    build(model,ps,[scale]*5,local=True)
    layer=model.model.layers[0]
    captured=[]
    hook=layer.inner.post_attention_layernorm.register_forward_pre_hook(
        lambda mod,args:captured.append(args[0].clone()))
    x=torch.randn(1,9,24)
    actual=layer(x,position_embeddings=None,use_cache=False)
    hook.remove()
    z=captured[0]
    h=z*torch.rsqrt(z.square().mean(-1,keepdim=True)+layer.inner.post_attention_layernorm.eps)*scale
    # Independent full-dimensional MLP evaluation, with only its output reduced.
    update=original_mlp((h@ps[0].T)*gain)@ps[1]
    expected=z@(ps[0].T@ps[1])+update
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)


def test_sorting_separates_correct_order_from_numeric_string_format():
    case={'id':'sort','family':'sorting','check':{'kind':'json','expected':[-3,0,8]}}
    result=grade_case('["-3", "0", "8"]',case)
    assert result['correct'] and not result['typed_value_correct'] and not result['format_pass']
    assert not grade_case('["-3", "8", "0"]',case)['correct']
    assert not grade_case('["-3", false, "8"]',case)['correct']


def test_fitted_decoder_folds_into_readers_without_adding_parameters():
    model=tiny_model('qwen',True)
    model.model.layers[0].post_attention_layernorm.weight.copy_(torch.linspace(-.2,.4,32))
    original_mlp=copy.deepcopy(model.model.layers[0].mlp)
    gain=norm_gain(model.model.layers[0].post_attention_layernorm)
    final_gain=norm_gain(model.model.norm)
    embed=model.model.embed_tokens.weight.clone()
    q=bases()[0];p=q[:,:24];decoder=p+q[:,24:]@torch.randn(8,24)*.05
    ordinary=build(copy.deepcopy(model),[p,p],[1.04]*5)
    build(model,[p,p],[1.04]*5,decoders=[decoder]*5)
    assert stored_scalars(model)==stored_scalars(ordinary)
    z=torch.randn(1,9,24)
    normalized=z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)*1.04
    expected=original_mlp((normalized@decoder.T)*gain)@p
    actual=model.model.layers[0].mlp(model.model.layers[0].post_attention_layernorm(z))
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)
    logits=model.lm_head(model.model.norm(z))
    expected_head=(((normalized@decoder.T)*final_gain)@p)@(embed@p).T
    torch.testing.assert_close(logits,expected_head,atol=3e-6,rtol=3e-5)
    assert cache_check(model,torch.randint(0,101,(1,13)))['numerical_pass']
