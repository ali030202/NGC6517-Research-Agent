# Changelog

## 0.2.0 — Local research-agent core

### Added

- Unified dependency-light service, CLI, Gradio and MCP entrypoints.
- SQLite jobs/events, one active task, detached workers, cancellation, timeouts, interruption recovery and explicit fresh retries.
- Bounded ablation loop with validation feedback and per-trial evidence.
- Frozen sources, unique run directories, source/manifest hashes and structured results.
- Manifest checks, local lexical retrieval, code proposals/diffs, backups and real-record reports.
- Configuration template, dependency groups, placeholder CSV example and architecture/operations documentation.

### Fixed

- Removed eager API/vector imports and personal deployment defaults from the local core.
- Test data is no longer evaluated on each best checkpoint; final test is explicit.
- Checkpoint/threshold selection uses validation F1, with cost as tie-breaker.
- Validation accuracy is read from accuracy instead of copying F1.
- Missing files and invalid labels no longer silently disappear or become RFI.
- Slash-compatible relative data paths; NPZ handles close after reading.
- Candidate syntax/interface checks, isolated training and versioned rollback.
- Optional research modules use the managed trainer and validation ledger; BFTS candidates no longer overwrite the active model during exploration.

### Scope

- API/Token setup remains deferred.
- Existing local data/results are preserved; old metrics are not silently imported into the new protocol.
- No new training, runtime tests, GPU benchmarks or performance measurements were run. Static Python parsing and publication-scope checks were performed.
