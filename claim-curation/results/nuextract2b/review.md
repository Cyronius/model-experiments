# Proposed claims for review

Exact citations are checked mechanically. Semantic support has not been certified.

## C-9cb408964534d7da28c7

Masked softmax scores are bf16 `lowest`, not -inf, because the exp argument is S*c - m*c in fp32 and -inf would stay -inf on a path that expects finite values (dit_fa trap 5).

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), unresolved.

```text
Masked softmax scores are bf16 `lowest`, not -inf, because the exp argument is S*c - m*c in fp32 and -inf would stay -inf on a path that expects finite values (dit_fa trap 5).
```

Citation valid: False. Review status: proposed.

Validation: quote is not verbatim in supplied text; quote has no allowed original-source span

## C-33f426a43492b769a426

The core starts in floor rounding (vegah trap list says the same: 'floor rounding default').

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), unresolved.

```text
The core starts in floor rounding (vegah trap list says the same: 'floor rounding default').
```

Citation valid: False. Review status: proposed.

Validation: quote is not verbatim in supplied text; quote has no allowed original-source span

## C-438656728d7f01dea427

In vae_ew, converting to bf16 before any data kernel ran meant the first dispatch after a context load was 2x less accurate than every later one: 1.8e-4 vs 8.5e-5 rel_fro from identical inputs.

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 28-28.

```text
In vae_ew, converting to bf16 before any data kernel ran meant the first dispatch after a context load was 2x less accurate than every later one: 1.8e-4 vs 8.5e-5 rel_fro from identical inputs.
```

Citation valid: True. Review status: proposed.

## C-58f5d3ed81de33533268

Fix: set the rounding mode in the first kernel of every dispatch (`vew_begin`), not only in the data kernels.

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 28-28.

```text
Fix: set the rounding mode in the first kernel of every dispatch (`vew_begin`), not only in the data kernels.
```

Citation valid: True. Review status: proposed.

## C-8fe7ea58ed89ba3fc662

The ash kernel took 7 ms at batch 2.

Conditions: ["device Cedar", "compiler 1.2", "batch 2"]

Source: [synthetic://synthetic-negation](../../data/sources/synthetic-negation.md), lines 2-2.

```text
On device Cedar with compiler 1.2, the ash kernel took 7 ms at batch 2.
```

Citation valid: True. Review status: proposed.

## C-9a669d03a96569afc647

This measurement does not establish the latency at other batch sizes.

Conditions: []

Source: [synthetic://synthetic-negation](../../data/sources/synthetic-negation.md), lines 2-2.

```text
This measurement does not establish the latency at other batch sizes.
```

Citation valid: True. Review status: proposed.

## C-f243820ea532ba271fac

How much stack would the birch kernel require?

Conditions: []

Source: [synthetic://synthetic-questions](../../data/sources/synthetic-questions.md), lines 2-2.

```text
Questions for future work:
```

Citation valid: True. Review status: proposed.

## C-6f366776dff04e60a349

Could compiler 9.9 change that requirement?

Conditions: []

Source: [synthetic://synthetic-questions](../../data/sources/synthetic-questions.md), lines 4-4.

```text
Could compiler 9.9 change that requirement?
```

Citation valid: True. Review status: proposed.

## C-fc4aa4eed4c5ee66f922

Synthetic test fixture, not a hardware finding.

Conditions: []

Source: [synthetic://synthetic-version-scope](../../data/sources/synthetic-version-scope.md), lines 1-1.

```text
Synthetic test fixture, not a hardware finding.
```

Citation valid: True. Review status: proposed.

## C-f06806de2beaaa3f9d25

For the birch kernel, stack 0x1800 was sufficient only with compiler 1.2. Compiler 1.3 was not tested.

Conditions: ["birch kernel", "stack 0x1800", "compiler 1.3"]

Source: [synthetic://synthetic-version-scope](../../data/sources/synthetic-version-scope.md), lines 2-2.

```text
For the birch kernel, stack 0x1800 was sufficient only with compiler 1.2. Compiler 1.3 was not tested.
```

Citation valid: True. Review status: proposed.

## C-b827c12ef0df09d2b491

The result must not be generalized to compiler 1.3.

Conditions: []

Source: [synthetic://synthetic-version-scope](../../data/sources/synthetic-version-scope.md), lines 2-2.

```text
The result must not be generalized to compiler 1.3.
```

Citation valid: True. Review status: proposed.

