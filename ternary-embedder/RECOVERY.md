# Windows project recovery — 2026-10-24

Recovered from the LAN FTP share, `code/model-experiments/ternary-embedder`.
The Windows copy includes work through October 23, newer than this checkout.

Recovered source, documentation, evaluation results, and the preferred Grid 1
2a_384 float/final checkpoints plus the matched float-control final checkpoint.
486 selected files total 207,234,861 bytes; every selected file has its expected
FTP listing size. The 2a_384 final checkpoint SHA256 matches the retained NPU
oracle report. Existing files were compared and updated from that copy; prior
versions are backed up in /tmp/ternary-recovery/local-before.

Excluded caches, bundled toolchains, generated build directories, native-stage
fixtures, logs directories, smoke work directories, duplicate large archives,
and other training checkpoints. `recovery-inventory.json` lists remote files
and sizes after directory exclusions, including checkpoint files not downloaded.
The nested training repository's `.git` metadata was not copied.

The NPU implementation still refers to Windows toolchain and XRT DLL paths;
recovery does not establish Linux execution. See native-stage/README.md for
the measured state and reproduction dependencies. No training or NPU tests
were run during recovery.
