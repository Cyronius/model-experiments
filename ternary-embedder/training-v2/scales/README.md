# Prepared ternary scale comparison

Template only; no reservation, approval, package or paid pod yet. The existing
$10 campaign authorization bounds any later allocation together with all other
jobs. Finalize the parent and direct-supervision loss using the earlier frozen
comparisons before launching this profile.

Compare fixed row scales, independently learned row scales with all master
parameters trainable, and only learned row scales with master parameters frozen.
All use the same two-layer ternary parent, data exposure and inference format:
{-1,0,+1} codes plus BF16 row scales. The last option trains6912 scalar values.
Canonical export removes training-only log scales; resumable states retain the
original master weights, log scales, optimizer, sampler and RNG.

The provisional recipe uses the benchmark-clean779098/15912 pair corpus,
batch1024/micro128, two epochs, encoder LR4e-5 and scale LR multiplier25.
One Secure RTX4090 at$0.89/hour, expected0.7–1.2h/$0.65–$1.10; maximum1.5h,
cap$1.50 including temporary60GB disk. Work cutoff70min, independent deletion
watchdog80min, regular SHA-verified off-pod checkpoints. No new native kernel or
additional inference parameter is required. This is an untested quality idea.
