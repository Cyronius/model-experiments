"""Schedules. All intervals are [start, end); progress is training FLOPs.

The pilot compared the first seven. Follow-up 3 (local/global, drivers/followup-local-global.sh)
adds two that combine whole-network (global) and fixed-block (local) training:

- global_fixed: the whole network first for one equal share, then the fixed blocks bottom to
  top, each one share (7 equal FLOP shares). The pilots showed joint gaining fastest early and
  drifting later, and fixed slow but steady.
- alternating: the fixed bottom-to-top sweep, but the last `global_share` of each of
  `cycles` equal FLOP slices trains the whole network.
- coarse_to_fine: hierarchical reversed (user, 2026-11-11). Equal-FLOP stages of halving window
  width: the whole network, then 16, 8, 4, 2 and 1 layers, each stage swept bottom to top.
- random: a fresh random window every step (user, 2026-11-11): width uniform on 1..depth, then
  start uniform over the positions that fit, so every window size from one layer to the whole
  network is equally likely. (Uniform over (start, end) pairs would train the whole network on
  1 step in 300.) Draws come from `rng`, which run.py seeds separately from the data order.
- random_dwell: random windows drawn the same way, but each is kept until its own training loss
  says to move on (user, 2026-11-11). Losses of different windows are not comparable (a 4-layer
  hidden-state loss is ~0.15, a window ending at the top adds output KL and is ~1.0), so the test
  is relative to the window itself: the mean of its first `dwell_warm` losses is the baseline,
  an EMA (`dwell_beta`) tracks it after that, and the window flips when the EMA is
  `dwell_gain` below the baseline (gain), when the EMA has not set a new best by 1% for
  `dwell_patience` steps (stall), or after `dwell_max` steps (cap). Training rows are used once,
  so each step's loss is measured on unseen data before its update and the rule costs no FLOPs.
  At one sequence per step the loss varies 10-30% step to step while a fixed block improves only
  5-17% over its whole dwell, so these thresholds are guesses; flips and reasons are logged.
"""
from dataclasses import dataclass, field
import math

STRATEGIES = ('layerwise', 'fixed', 'hierarchical', 'overlap', 'joint',
              'adaptive_error', 'adaptive_plateau', 'global_fixed', 'alternating', 'coarse_to_fine',
              'random', 'random_dwell')


def windows(depth, width, overlap=False):
    width = min(width, depth)
    if width < 1:
        raise ValueError('Window width must be positive')
    if width == depth:
        return [(0, depth)]
    stride = max(1, width // 2) if overlap else width
    return [(a, min(a + width, depth)) for a in range(0, depth, stride)]


def hierarchy(depth):
    widths = []
    n = 1
    while n < depth:
        widths.append(n)
        n *= 2
    return widths + [depth]


@dataclass
class Schedule:
    """Window choice. Adaptive arms (v2) move over the same aligned width-`fixed_width` blocks as
    `fixed`, so they differ from it only in which block trains when:

    - adaptive_error: at each decision, the block with the most error *added* inside it
      (sum over its layers of max(0, e_i - e_{i-1}), e_{-1} = 0); earliest block on ties.
      v3 cap: a block that has already received `max_block_share` times its equal share of the
      budget in total is not eligible (unless every block is capped). Without the cap the v2
      pilot spent ~90% of its budget on layers 0-3, whose added error training barely reduced.
    - adaptive_plateau: blocks in order, wrapping; move on when the block's endpoint error
      improved by less than `min_improvement` (relative) since the last decision, or when the
      block has held `max_block_share` times its equal share of the budget.

    v1 used one absolute error threshold (0.15). On a badly damaged student every layer exceeds
    it, so both controllers stayed in layers 0-3 for the whole pilot.
    """
    strategy: str
    depth: int
    fixed_width: int = 4
    min_improvement: float = 0.01
    max_block_share: float = 2.0
    cycles: int = 24
    global_share: float = 0.2
    progress: float = 0.0
    rng: object = None  # numpy Generator; required by `random` and `random_dwell`
    dwell_warm: int = 5
    dwell_beta: float = 0.9
    dwell_gain: float = 0.05
    dwell_patience: int = 15
    dwell_max: int = 60
    within: float = 0.0  # position inside the current window's own FLOP segment, [0, 1)
    observations: list = field(default_factory=list)

    def __post_init__(self):
        if self.strategy not in STRATEGIES or self.depth < 1:
            raise ValueError((self.strategy, self.depth))
        self.blocks = windows(self.depth, self.fixed_width)
        self.block = 0
        self.entered = 0.0
        self.last_error = None
        self.block_spent = [0.0] * len(self.blocks)
        self.last_fraction = 0.0
        self.range = None  # random_dwell: current window and its loss tracking
        self.flip = True

    @property
    def active(self):
        return self.blocks[self.block]

    @property
    def adaptive(self):
        return self.strategy.startswith('adaptive')

    def choose(self, fraction):
        """Fixed/hierarchical blocks each receive equal FLOP shares, not equal steps."""
        p = min(max(fraction, 0.0), math.nextafter(1.0, 0.0))
        self.progress = p
        if self.strategy == 'joint':
            self.within = p
            return (0, self.depth)
        if self.adaptive:
            return self.active
        if self.strategy in ('random', 'random_dwell'):
            if self.rng is None:
                raise ValueError('Random strategies need an rng')
            self.within = p
            if self.strategy == 'random' or self.flip:
                width = int(self.rng.integers(1, self.depth + 1))
                a = int(self.rng.integers(0, self.depth - width + 1))
                self.range, self.flip = (a, a + width), False
                self.dwell, self.warm_losses, self.ema, self.best, self.since_best = 0, [], None, None, 0
            return self.range
        if self.strategy == 'global_fixed':
            segments = [(0, self.depth)] + self.blocks
            k = min(int(p * len(segments)), len(segments) - 1)
            self.within = min(p * len(segments) - k, math.nextafter(1.0, 0.0))
            return segments[k]
        if self.strategy == 'alternating':
            k = min(int(p * len(self.blocks)), len(self.blocks) - 1)
            self.within = min(p * len(self.blocks) - k, math.nextafter(1.0, 0.0))
            slice_position = p * self.cycles - math.floor(p * self.cycles)
            return (0, self.depth) if slice_position >= 1 - self.global_share else self.blocks[k]
        widths = hierarchy(self.depth) if self.strategy in ('hierarchical', 'overlap') else [
            1 if self.strategy == 'layerwise' else self.fixed_width]
        if self.strategy == 'coarse_to_fine':
            widths = hierarchy(self.depth)[::-1]
        stage = min(int(p * len(widths)), len(widths) - 1)
        within = p * len(widths) - stage
        choices = windows(self.depth, widths[stage], self.strategy == 'overlap')
        k = min(int(within * len(choices)), len(choices) - 1)
        self.within = min(within * len(choices) - k, math.nextafter(1.0, 0.0))
        return choices[k]

    def feedback(self, loss, fraction):
        """Training loss of the step just taken on `self.range` (random_dwell only)."""
        if self.strategy != 'random_dwell':
            return
        self.dwell += 1
        if self.dwell <= self.dwell_warm:
            self.warm_losses.append(loss)
            if self.dwell == self.dwell_warm:
                self.baseline = self.ema = self.best = sum(self.warm_losses) / len(self.warm_losses)
            return
        self.ema = self.dwell_beta * self.ema + (1 - self.dwell_beta) * loss
        if self.ema < self.best * 0.99:
            self.best, self.since_best = self.ema, 0
        else:
            self.since_best += 1
        reason = ('gain' if self.ema <= self.baseline * (1 - self.dwell_gain) else
                  'stall' if self.since_best >= self.dwell_patience else
                  'cap' if self.dwell >= self.dwell_max else None)
        if reason:
            self.flip = True
            self.observations.append(dict(fraction=fraction, window=list(self.range), steps=self.dwell,
                                          reason=reason, baseline=self.baseline, ema=self.ema))

    def observe(self, errors, fraction):
        """Errors: per-depth discrepancy of the student's own stream against the teacher on
        held-out controller rows. `fraction`: budget spent so far (training + decisions)."""
        if len(errors) != self.depth or not all(math.isfinite(e) and e >= 0 for e in errors):
            raise ValueError('Expected a finite nonnegative discrepancy for every layer')
        old = self.active
        reason = None
        self.block_spent[self.block] += max(0.0, fraction - self.last_fraction)
        self.last_fraction = fraction
        if self.strategy == 'adaptive_error':
            added = [max(0.0, e - (errors[i - 1] if i else 0.0)) for i, e in enumerate(errors)]
            scores = [sum(added[a:b]) for a, b in self.blocks]
            cap = self.max_block_share / len(self.blocks)
            eligible = [i for i, spent in enumerate(self.block_spent) if spent < cap] or list(range(len(scores)))
            self.block = max(eligible, key=lambda i: (scores[i], -i))
            reason = dict(block_added_error=scores, block_spent=list(self.block_spent),
                          capped=[i for i in range(len(scores)) if i not in eligible])
        elif self.strategy == 'adaptive_plateau':
            current = errors[self.active[1] - 1]
            improvement = (math.inf if self.last_error is None else
                           (self.last_error - current) / max(self.last_error, 1e-12))
            overdue = fraction - self.entered >= self.max_block_share / len(self.blocks)
            if improvement < self.min_improvement or overdue:
                self.block = (self.block + 1) % len(self.blocks)
                self.entered, self.last_error = fraction, None
                reason = 'overdue' if overdue else 'plateau'
            else:
                self.last_error = current
            reason = dict(endpoint_error=current, relative_improvement=improvement if math.isfinite(improvement) else None,
                          moved=reason)
        self.observations.append(dict(fraction=fraction, old=list(old), new=list(self.active),
                                      errors=list(errors), reason=reason))
