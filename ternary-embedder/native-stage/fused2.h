#pragma once
// Native 384 post-norm BERT, four token rows per core. Adapted from the
// archived tiny-planner's tt.h / vecmath.h; no cross attention or pre-norm.
// fused2: attention masks come from per-token segment IDs (0 = padding), so any
// set of texts can share one 128-token dispatch. The mask is expanded once per
// dispatch into an additive score bias instead of being tested lane by lane.
// Each weight packet carries its 32 output channels' bias and scale at byte
// 3072, so projections need no separate parameter packets.
#define AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 1
#include "vecmath.h"
using MM = aie::mmul<4,8,8,bfloat16,bfloat16,accauto>;
using TM = aie::mmul<4,16,16,int8,int4>;
static inline v32b bf(v32f v) { accf32 a; a.from_vector(v); return a.to_vector<bfloat16>(); }
static inline v32f fp(v32b v) { accf32 a; a.from_vector(v); return a.to_vector<float>(); }
static inline v32f rep(const float *p) { auto v=aie::load_v<8>(p); return aie::concat(v,v,v,v); }
static inline v32f rows(const float *p) {
  return aie::concat(aie::broadcast<float,8>(p[0]),aie::broadcast<float,8>(p[1]),
                     aie::broadcast<float,8>(p[2]),aie::broadcast<float,8>(p[3]));
}
static inline float sum8(const float *p) { return aie::reduce_add(aie::concat(aie::load_v<8>(p),aie::zeros<float,8>())); }
static inline v32f src(const void *a,int kb,int isbf) {
  return isbf ? fp(aie::load_v<32>((const bfloat16*)a+kb*32)) : aie::load_v<32>((const float*)a+kb*32);
}

inline __attribute__((noinline)) void quant(const void *a,int8 *q,int d,int isbf) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  v32f mx=aie::zeros<float,32>();
  for(int kb=0;kb<d/8;++kb) mx=aie::max(mx,aie::abs(src(a,kb,isbf)));
  alignas(64) float tmp[32], scales[32];
  aie::store_v(tmp,mx);
  for(int r=0;r<32;++r) scales[r]=r<4 ? aie::reduce_max(aie::concat(aie::load_v<8>(tmp+r*8),aie::zeros<float,8>())) : 1.f;
  auto bits=aie::max(aie::load_v<32>(scales),aie::broadcast<float,32>(1e-5f)).cast_to<int32>();
  auto e=aie::add(aie::sub(aie::downshift(bits,23),aie::broadcast<int32,32>(133)),
    aie::select(aie::zeros<int32,32>(),aie::broadcast<int32,32>(1),
      aie::gt(aie::bit_and(bits,aie::broadcast<int32,32>(0x7fffff)),aie::broadcast<int32,32>(0x7e0000))));
  aie::store_v((float*)(q+4*d),aie::upshift(aie::add(e,aie::broadcast<int32,32>(127)),23).cast_to<float>());
  alignas(64) int32 ee[32];
  aie::store_v(ee,aie::upshift(aie::sub(aie::zeros<int32,32>(),e),23));
  // Add exponent bits to multiply fp32 by 2^-e exactly, retaining its mantissa.
  const auto iv=aie::concat(aie::broadcast<int32,8>(ee[0]),aie::broadcast<int32,8>(ee[1]),
                            aie::broadcast<int32,8>(ee[2]),aie::broadcast<int32,8>(ee[3]));
  for(int kb=0;kb<d/16;++kb) {
    auto x0=aie::add(src(a,2*kb,isbf).cast_to<int32>(),iv).cast_to<float>();
    auto x1=aie::add(src(a,2*kb+1,isbf).cast_to<int32>(),iv).cast_to<float>();
    auto [z0,z1]=aie::interleave_zip(x0,x1,8);
    auto c0=aie::to_fixed<int16>(z0,0),c1=aie::to_fixed<int16>(z1,0);
    aie::store_v(q+kb*64,aie::pack(aie::min(aie::max(c0,(int16)-128),(int16)127)));
    aie::store_v(q+kb*64+32,aie::pack(aie::min(aie::max(c1,(int16)-128),(int16)127)));
  }
}

// exp and 1/d at the precision their bf16 consumers keep (GELU output, attention
// probabilities and normalization), cheaper than vecmath.h's ~1e-7 versions.
// exp(x): 2^n * p(f), f in [-0.5, 0.5], degree-4 minimax (2.6e-6 relative), with
// 2^n applied by adding n to the exponent bits. Inputs are clamped to [-87, 88].
inline __attribute__((noinline)) v32f fexp(v32f x) {
  x=aie::max(aie::min(x,88.f),-87.f);
  const v32f t=fscaleN<32>(x,1.44269504f);
  const auto n=aie::to_fixed<int32>(t,0);
  const v32f f=fsubN<32>(t,aie::to_float<float>(n,0));
  v32f p=aie::broadcast<float,32>(9.571806528e-03f);
  p=faddN<32>(fmulN<32>(p,f),aie::broadcast<float,32>(5.591800436e-02f));
  p=faddN<32>(fmulN<32>(p,f),aie::broadcast<float,32>(2.402470261e-01f));
  p=faddN<32>(fmulN<32>(p,f),aie::broadcast<float,32>(6.931217909e-01f));
  p=faddN<32>(fmulN<32>(p,f),aie::broadcast<float,32>(9.999992847e-01f));
  return aie::add(p.cast_to<int32>(),aie::upshift(n,23)).cast_to<float>();
}
// 1/d: hardware estimate refined by one Newton step.
inline __attribute__((noinline)) v32f frecip(v32f d) {
  v32f r=aie::inv(d);
  return fmulN<32>(r,fsubN<32>(aie::broadcast<float,32>(2.f),fmulN<32>(d,r)));
}

inline __attribute__((noinline)) v32f gelu(v32f x) {
  // x * sigmoid(x * (1.59576912 + .0713548164*x^2)), PyTorch tanh GELU.
  auto z=fmulN<32>(x,faddN<32>(fscaleN<32>(fmulN<32>(x,x),.0713548164f),aie::broadcast<float,32>(1.5957691216f)));
  z=aie::max(aie::min(z,16.f),-16.f);
  auto den=faddN<32>(aie::broadcast<float,32>(1.f),fexp(fsubN<32>(aie::zeros<float,32>(),z)));
  return fmulN<32>(x,frecip(den));
}

// kind: Q=0, K transpose=1, V=2, residual=3, GELU=4.
inline __attribute__((noinline)) void project(const uint8 *w,const int8 *aq,
    float *x,bfloat16 *q,uint8 *kv,int j,int kind,int d) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  TM lo,hi;
  // Profiling switches (F2_DEFS): F2_PROF_NOMAC keeps one MAC step, F2_PROF_NOEPI
  // skips the float epilogue, F2_PROF_NOGELU treats GELU outputs as linear.
#ifdef F2_PROF_NOMAC
  for(int kb=0;kb<1;++kb) {
#else
  for(int kb=0;kb<d/16;++kb) {
#endif
    auto a=aie::load_v<64>(aq+kb*64);
    auto b=aie::load_v<128>(w+kb*128);
    lo.mac(a,aie::bit_and((uint8)0xcc,aie::upshift(b.cast_to<uint32>(),2).cast_to<uint8>()).cast_to<int4>());
    hi.mac(a,aie::bit_and((uint8)0xcc,b).cast_to<int4>());
  }
  alignas(64) int32 ci[128];
  aie::store_v(ci,lo.to_vector<int32>());aie::store_v(ci+64,hi.to_vector<int32>());
#ifdef F2_PROF_NOEPI
  aie::store_v((int32*)q+(j%6)*128,aie::load_v<64>(ci));aie::store_v((int32*)q+(j%6)*128+64,aie::load_v<64>(ci+64));
  return;
#endif
  const float *par=(const float*)(w+3072);   // bias[32], then scale[32]
  const v32b sa=bf(rows((const float*)(aq+4*d)));
  for(int t=0;t<2;++t) {
    auto [u0,u1]=aie::interleave_unzip(aie::to_float<float>(aie::load_v<32>(ci+t*64),0),
                                     aie::to_float<float>(aie::load_v<32>(ci+t*64+32),0),8);
    for(int h=0;h<2;++h) {
      auto u=h ? u1:u0;
      auto uh=bf(u),ul=bf(fsubN<32>(u,fp(uh)));
      int ch=t*16+h*8;
      auto scale=aie::mul(bf(rep(par+32+ch)),sa).to_vector<bfloat16>();
      accf32 v;v.from_vector(rep(par+ch));v=aie::mac(v,uh,scale);v=aie::mac(v,ul,scale);
      auto y=v.to_vector<float>();
      int off=j*128+t*64+h*32;
      if(kind==3) aie::store_v(x+off,faddN<32>(aie::load_v<32>(x+off),y));
      else if(kind==1) aie::store_v((bfloat16*)kv+j*256+t*64+h*32,aie::transpose(bf(y),4,8));
      else if(kind==2) aie::store_v((bfloat16*)kv+j*256+128+t*64+h*32,bf(y));
#ifdef F2_PROF_NOGELU
      else aie::store_v(q+off,bf(y));
#else
      else aie::store_v(q+off,bf(kind==4 ? gelu(y):y));
#endif
    }
  }
}

inline __attribute__((noinline)) v16f rsqrt(v16f x) {
  auto r=aie::sub(aie::broadcast<int32,16>(0x5f3759df),aie::downshift(x.cast_to<int32>(),1)).cast_to<float>();
  auto half=fscaleN<16>(x,.5f);
  for(int i=0;i<4;++i) r=fmulN<16>(r,fsubN<16>(aie::broadcast<float,16>(1.5f),fmulN<16>(half,fmulN<16>(r,r))));
  return r;
}

inline __attribute__((noinline)) void norm_sum(const float *x,float *tmp) {
  accf32 s=aie::zeros<accfloat,32>();
  for(int k=0;k<48;++k) s=aie::add(s,aie::load_v<32>(x+k*32));
  aie::store_v(tmp,s.to_vector<float>());
}
inline __attribute__((noinline)) void norm_rows(const float *tmp,float *dest,float eps) {
  for(int r=0;r<16;++r) dest[r]=r<4?sum8(tmp+r*8):384.f;
  aie::store_v(dest,faddN<16>(fscaleN<16>(aie::load_v<16>(dest),1.f/384.f),aie::broadcast<float,16>(eps)));
}
inline __attribute__((noinline)) void norm_sq(const float *x,const float *mean,float *tmp) {
  auto m=rows(mean);accf32 s=aie::zeros<accfloat,32>();
  for(int k=0;k<48;++k) { auto d=fsubN<32>(aie::load_v<32>(x+k*32),m);s=aie::add(s,fmulN<32>(d,d)); }
  aie::store_v(tmp,s.to_vector<float>());
}
inline __attribute__((noinline)) void norm_out(float *x,const float *p,const float *mean,const float *var) {
  auto m=rows(mean),rr=rows(var);
  for(int k=0;k<48;++k) {
    auto y=faddN<32>(fmulN<32>(fmulN<32>(fsubN<32>(aie::load_v<32>(x+k*32),m),rr),rep(p+k*8)),rep(p+512+k*8));
    aie::store_v(x+k*32,y);
  }
}
inline __attribute__((noinline)) void norm(float *x,const float *p) {
  alignas(64) float tmp[32],mean[16],var[16];
  norm_sum(x,tmp);norm_rows(tmp,mean,0.f);norm_sq(x,mean,tmp);norm_rows(tmp,var,1e-12f);
  aie::store_v(var,rsqrt(aie::load_v<16>(var)));
  norm_out(x,p,mean,var);
}

alignas(64) static const bfloat16 ones[64]={
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

inline __attribute__((noinline)) void finish_attention(float *work,bfloat16 *q,int head) {
  auto inv=frecip(aie::max(aie::load_v<32>(work+128),1e-20f));
  for(int d=0;d<4;++d) aie::store_v(q+head*128+d*32,bf(fmulN<32>(aie::load_v<32>(work+d*32),inv)));
}

// bias[block*128+group*32+lane]: 0 where query row core*4+lane/8 may attend to
// key block*32+group*8+lane%8 (same nonzero segment), else -1e30.
inline __attribute__((noinline)) void attention_bias(const int32 *seg,float *bias,int core) {
  for(int i=0;i<512;++i) {
    int key=(i>>7)*32+((i>>5)&3)*8+(i&7),row=core*4+((i>>3)&3);
    bias[i]=seg[key] && seg[key]==seg[row] ? 0.f : -1e30f;
  }
}

// After the last key block (block 3) the head's output replaces its queries in q.
inline __attribute__((noinline)) void attend(bfloat16 *q,const uint8 *packet,float *work,
    float *ml,const float *bias,int head,int block) {
  const auto *e=(const bfloat16*)packet;
  float *s=work+160;
  bfloat16 *p=(bfloat16*)(work+288);
  if(block==0) {
    for(int i=0;i<160;i+=32) aie::store_v(work+i,aie::zeros<float,32>());
    ml[0]=-1e30f;
  }
  v32f maxv=aie::broadcast<float,32>(ml[0]);
  for(int group=0;group<4;++group) {
    MM mm(aie::zeros<accfloat,32>());
    for(int d=0;d<4;++d) {
      auto [b0,b1]=aie::interleave_zip(aie::load_v<32>(e+group*512+d*32),aie::load_v<32>(e+group*512+256+d*32),4);
      mm.mac(aie::load_v<32>(q+head*128+d*32),aie::concat(b0,b1));
    }
    auto sv=faddN<32>(fscaleN<32>(mm.to_vector<float>(),.1767766952966369f),aie::load_v<32>(bias+block*128+group*32));
    aie::store_v(s+group*32,sv);
    maxv=aie::max(maxv,sv);
  }
  float next=aie::reduce_max(maxv);
  auto mn=aie::broadcast<float,32>(next);
  auto rb=fexp(fsubN<32>(aie::broadcast<float,32>(ml[0]),mn));
  for(int i=0;i<160;i+=32) aie::store_v(work+i,fmulN<32>(aie::load_v<32>(work+i),rb));
  ml[0]=next;
  for(int group=0;group<4;++group) {
    auto sv=aie::load_v<32>(s+group*32);
    auto prob=fexp(fsubN<32>(sv,mn));
    prob=aie::select(aie::zeros<float,32>(),prob,aie::gt(sv,-1e20f));
    aie::store_v(p+group*32,bf(prob));
  }
  for(int d=0;d<5;++d) {
    MM mm(accf32(aie::load_v<32>(work+d*32)));
    for(int group=0;group<4;++group) {
      auto vv=d==4 ? aie::load_v<64>(ones) :
        aie::concat(aie::load_v<32>(e+group*512+128+d*32),aie::load_v<32>(e+group*512+384+d*32));
      mm.mac(aie::load_v<32>(p+group*32),vv);
    }
    aie::store_v(work+d*32,mm.to_vector<float>());
  }
  if(block==3) finish_attention(work,q,head);
}
