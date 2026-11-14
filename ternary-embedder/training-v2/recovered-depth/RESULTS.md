# Recovered pretrained three-layer baseline

Historical r0_3x384 checkpoints were recovered selectively from the authorized LAN share, checked against FTP listing byte counts and recorded SHA256 receipts. Both fixed checkpoints were selected before this fresh evaluation. No paid training was used.

| Precision | Retrieval mean | SciFact | NFCorpus | STSBenchmark |12-task mean |
|---|---:|---:|---:|---:|---:|
| float | 45.868 | 60.387 | 31.349 | 83.851 | 71.529 |
| ternary | 43.218 | 56.663 | 29.773 | 82.477 | 70.141 |

The pretrained three-layer float model is close to MiniLM on the broader mean, but its retrieval45.87 remains below MiniLM46.92. Ternary retrieval improves over the original two-layer checkpoint, while its broader mean is essentially unchanged. These are observed development tasks, not unseen confirmation or full MTEB.

Each has17241984 parameters. The float checkpoint is the75M-token halfway checkpoint; the original two-layer independent float control saw150M tokens. Their different histories prevent a controlled architecture-depth claim. This provides a better-initialized three-layer candidate for new direct-data training than clone-and-append. Existing native NPU graphs still support only two layers. No deployment change.
