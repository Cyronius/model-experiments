"""Configuration dataclasses. Every run is fully described by an ExperimentConfig."""

from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Tuple
import json

# ---------------------------------------------------------------- vocabulary

PAD, EOS = 0, 1

# Recall symbols get their own private code points so the whole pipeline can stay
# string-based; table ids disambiguate which lookup table a key belongs to, since
# the same (a, b) key means different things in different tables.
RECALL_N = 64
RECALL_CHARS = "".join(chr(0x100 + i) for i in range(RECALL_N))
# Table sizes the recall task can draw from. Sizes between 16 and 32 exist to
# calibrate the width axis: 16 is free at every tier and 32 is out of reach at
# every tier, so the size that actually separates tiers lies between them.
TABLE_M = (16, 20, 24, 28, 32, 64)
TABLE_IDS = "".join(chr(0x200 + i) for i in range(len(TABLE_M)))

CHARS = "0123456789+=ABR()[]{}<>" + RECALL_CHARS + TABLE_IDS
ITOS = ["<pad>", "<eos>"] + list(CHARS)
STOI = {c: i + 2 for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(ITOS)


@dataclass
class DataConfig:
    block_size: int = 32          # x/y length; longest example is 28 tokens
    easy_digits: Tuple[int, int] = (1, 2)
    hard_digits: Tuple[int, int] = (5, 8)
    easy_depth: Tuple[int, int] = (1, 2)
    hard_depth: Tuple[int, int] = (5, 8)
    max_extra_pairs: int = 4      # closed pairs interleaved into bracket prompts

    # Zero-pads both addition operands to a fixed digit count, pinning every operand
    # digit to a constant absolute position. Without it, learned absolute position
    # embeddings cannot locate the i-th digit of a variable-length span, and 5-8
    # digit addition is unsolvable at any width, depth, or step budget tested --
    # exact-match 0%, digit accuracy at chance in the middle positions. With it the
    # same 4-layer model reaches 100%. Set to 0 to reproduce the old unpadded task.
    arith_pad_digits: int = 8

    # Recall: memorize a fixed random table T[a][b] -> (c, d). Difficulty is table
    # size, i.e. how many arbitrary associations must be stored. This is the only
    # axis in the suite where MLP width is the binding constraint -- see README.
    # m=32 is the calibrated hard size: at the small preset it separates tiers
    # cleanly (h8 20% exact, h64 83%) without either end saturating. m=64 used to be
    # in here and made the cell impossible at every tier -- 4096 associations against
    # 16k MLP parameters -- which flattened the whole width axis. See future-work.md
    # section 4 for the sweep.
    recall_easy_m: Tuple[int, ...] = (16,)      #  256 associations
    recall_hard_m: Tuple[int, ...] = (32,)      # 1024 associations
    recall_table_seed: int = 20260815

    n_eval_per_cell: int = 512    # held-out examples per cell, for loss
    n_exact_per_cell: int = 128   # subset decoded greedily for exact-match


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 32
    n_layer: int = 4
    d_model: int = 128
    n_head: int = 4
    d_ff: int = 512               # reference dense width; the 100%-FLOPs point
    dropout: float = 0.0
    # rope. Learned absolute positions give no sharing across output positions, so
    # each answer digit has to be learned separately and runs freeze with one
    # digit at chance: 7 of 12 dense runs came out clean, versus 12 of 12 with
    # rope. Set to "learned" to reproduce the original trunk.
    pos_encoding: str = "rope"     # learned | rope

    mlp: str = "dense"            # dense | mow | moe
    dense_hidden: Optional[int] = None   # None -> d_ff

    # --- MoW
    tiers: Tuple[int, ...] = (128, 256, 384, 512)
    tier_scales: bool = True      # per-tier learned output scale (load-bearing)

    # --- MoE
    n_experts: int = 4
    expert_hidden: int = 128      # n_experts * expert_hidden should equal d_ff
    moe_mode: str = "topk"        # topk | adaptive
    top_k: int = 1
    k_choices: Tuple[int, ...] = (1, 2, 3, 4)
    balance_coef: float = 1e-2

    # --- router (shared by MoW and MoE)
    router: bool = False
    router_temp0: float = 1.0
    router_temp_min: float = 0.1
    router_hard: bool = False     # straight-through: hard forward, soft backward

    def mlp_hidden_total(self) -> int:
        if self.mlp == "moe":
            return self.n_experts * self.expert_hidden
        return self.d_ff


@dataclass
class TrainConfig:
    seed: int = 0
    steps: int = 3000
    batch_size: int = 32
    lr: float = 2e-3
    min_lr_frac: float = 0.1
    warmup: int = 200
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    eval_every: int = 250
    eval_batches: int = 8
    exact_every: int = 1000       # greedy-decode exact match is slower; run it rarely

    # dense        : one fixed width, no router
    # mow_pretrain : nesting pretrain, no router
    # routed       : router attached (MoW or MoE), FLOP budget loss active
    mode: str = "dense"

    # nesting pretrain: 4 uniform-tier passes + 1 random-per-layer pass per step.
    # The mixed pass matters: without it the trunk never sees the ragged per-layer
    # tier combinations the router actually produces at routing time.
    pretrain_uniform: bool = True
    pretrain_mixed: bool = True

    # Soft-to-hard: anneal temperature over the first hard_after_frac of the run,
    # then switch to straight-through hard selection for the remainder. Eval is
    # always hard, so the tail measures what deployment would actually do.
    hard_after_frac: float = 0.7

    lambda_flops: float = 0.0
    init_from: Optional[str] = None   # path to a checkpoint to warm-start from
    trunk_lr_mult: float = 1.0        # <1 lowers trunk LR during the router phase
    freeze_trunk: bool = False

    entropy_collapse_thresh: float = 0.1   # nats; below this we flag collapse


@dataclass
class ExperimentConfig:
    name: str = "run"
    tag: str = ""                 # condition family, for grouping in plots
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cpu"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=list)

    @staticmethod
    def load(path) -> "ExperimentConfig":
        with open(path) as f:
            d = json.load(f)
        return ExperimentConfig(
            name=d["name"], tag=d.get("tag", ""), device=d.get("device", "cpu"),
            notes=d.get("notes", ""),
            data=DataConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                               for k, v in d["data"].items()}),
            model=ModelConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                                 for k, v in d["model"].items()}),
            train=TrainConfig(**d["train"]),
        )


# ---------------------------------------------------------------- presets

SMALL = ModelConfig(n_layer=4, d_model=128, n_head=4, d_ff=64,
                    tiers=(8, 16, 32, 64), expert_hidden=16)

# Kept so the identical grid can be rerun at the plan's original size on a CUDA
# box without touching anything but the preset name.
FULL = ModelConfig(n_layer=6, d_model=256, n_head=4, d_ff=1024,
                   tiers=(256, 512, 768, 1024), expert_hidden=256)

PRESETS = {"small": SMALL, "full": FULL}


def preset_model(name: str, **overrides) -> ModelConfig:
    return replace(PRESETS[name], **overrides)


# ---------------------------------------------------------------- FLOP accounting
# Per-token forward FLOPs, counting a multiply-accumulate as 2. Only the MLP
# varies across conditions; attention is identical everywhere and reported apart.

def mlp_flops_per_token(d_model: int, hidden: int) -> float:
    return 4.0 * d_model * hidden          # fc (2*d*h) + proj (2*h*d)


def attn_flops_per_token(d_model: int, block_size: int) -> float:
    proj = 8.0 * d_model * d_model         # q,k,v,out
    scores = 2.0 * block_size * d_model    # causal average over the sequence
    return proj + scores


def flops_fraction(hidden: float, d_ff: int) -> float:
    """MLP FLOPs as a fraction of the reference dense-d_ff MLP."""
    return hidden / d_ff
