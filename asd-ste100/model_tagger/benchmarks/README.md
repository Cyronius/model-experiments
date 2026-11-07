# Software review benchmark v1

32 manually authored and assistant-reviewed cases: 16 clean controls and 16 narrow
repairs. Each case lists accepted text, mandatory literal surfaces and a review
rationale. The edits were checked to preserve negation, conditional branches,
temporal order, numeric values and code/identifier spelling. These are proposed
reference judgments, **pending owner review**, not independently verified gold.
Clean cases deliberately require exact preservation; repair variants can be added
through review, never to improve a reported model score after inspection.

The benchmark stays outside training and confidence calibration. Before training,
`evaluate_review.py isolation` checks canonical text of every benchmark input and
reference against all train/dev/test source and target texts in the rebuilt ID
corpus. A collision fails the pipeline. Scoring reports exact accepted text,
mandatory literal copies, content/literal guard failures and unresolved cases;
none of these metrics is a general semantic-equivalence judgment.

Some software prose is already understandable without an article. The repair
references deliberately request an explicit article; missing the reference is a
repair-recall failure, not automatically a meaning failure. The software whitelist
and protected names let the model retain terms without asserting all noun senses
are approved by the ASD-STE100 dictionary.
