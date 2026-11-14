# Qualitative spot review

These checks were made after inference against the retained input notes. They
do not alter the frozen prompt, gold patterns, or raw automatic scores, and
they are not independent human adjudication of every output.

## NuExtract 2.0 2B

- `synthetic-questions`: emits the two questions as claims. One quotation is
  only `Questions for future work:`. An exact quotation does not make a question
  into an asserted fact or support the interpretation in the statement.
- `unit-equivalence`: calls 64 KiB and 65536 bytes different scope, although
  they are equivalent units for the same capacity.
- `untested-not-failed`: calls “not tested” and “fails” contradictory. The first
  assertion does not establish a result for that configuration.
- Several real technical passages return empty arrays despite containing clear
  assertions. Valid JSON therefore does not imply useful extraction.

## Stock Qwen3-4B

- `shim-wrap` and `softmax-mask`: generated JSON has an extra trailing quote.
  The strict parser rejects it; no silent JSON repair is performed.
- `granite-stack`: the second statement discusses spilling with 0x3000 instead
  of explicitly retaining the source's successful 0x3000 outcome. The frozen
  `work|succeed|correct|resolve` regex incorrectly matches `workers` in that
  statement, producing a false-positive anchor in the original scores. The
  original scores are retained for audit. A separate, regression-tested matcher
  refinement bounds the outcome verb and removes that false positive from the
  refined table and scoped/cited total. Neither table is semantic adjudication.
- Some condition fragments differ from the source only in capitalization, or
  quotations drop backticks. Those still fail the strict exact-match gate.
  A future alignment stage could resolve formatting differences while retaining
  the raw model quote and exact original span; it must not silently rewrite facts.

## NuExtract3

- `rot13`: “The ROT13 path completed in 0.602 ms” loses the **classic** path
  qualifier. The source also reports an ELF path with a different latency.
  The quotation is exact, but the statement/conditions are not sufficiently scoped.
- `softmax-stack`: “A stack of 0x2000 works” drops its connection to `attn_fa`
  softmax with four interleaved exponentials. It must not become a universal
  stack-size recommendation. Another statement ties “version not stated for
  these runs” to firmware, although that quoted fragment originally describes
  the IRON/aiecc/Peano build; the document separately says firmware is unstated.
- `transfer-size`: preserves the successful GEMV exception but omits the
  crucial qualifier that the shim moves the whole drain. “Padding both ... fixed
  it” also retains source pronouns rather than making the claim stand alone.
- `shim-wrap`: changes a recommendation to keep large counts in a middle
  dimension into “prevents the hang,” strengthening the source into a causal
  guarantee. An assertion-type field should distinguish observations,
  recommendations, hypotheses, and limitations in the next schema.
- `unknown-scope` and `untested-not-failed`: the comparison pass incorrectly
  labels both pairs contradictory. Missing scope is not matching scope, and
  untested does not establish failure or success. Qwen4B got both labels right
  in this small supplied-pair benchmark.

## Implication for the next iteration

Use extraction as a proposal stage. Keep the raw quotation and a separately
reviewed, self-contained scoped statement. Avoid automatically promoting claims
because their quotations match. Separate document metadata from observation
extraction to reduce noise and generation cost; test qualifier restoration and
recommendation-vs-observation handling on independently annotated documents.
The lexical outcome-word matcher has been refined with a regression test;
original measurements remain the pilot baseline. Independently reviewed
semantic labels are still needed before treating these scores as a quality gate.
