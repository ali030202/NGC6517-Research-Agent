# Changelog

## 0.3.0 — LangGraph research workflow — 2026-10-25

### Added

- StateGraph nodes for data inspection, persistent data waiting, lexical retrieval, rule planning, trial selection, training submission, task monitoring, validation analysis and evidence reports.
- SQLite graph checkpointers with synchronous durability and restricted checkpoint deserialization.
- Detached workflow workers, one active workflow per project, persistent cancellation, state/history queries and explicit graph resumption.
- Stable per-trial request keys and a unique task index so checkpoint replay reuses submitted training instead of creating a duplicate trial.
- Workflow-wide frozen training sources, CSV fingerprints, remaining-budget job timeouts and validation-only selection.
- CLI, Gradio and MCP workflow controls; dedicated dependency groups and detailed LangGraph architecture/operations documentation.

### Changed

- `run_ai_scientist()` now starts LangGraph orchestration and returns `workflow_id` instead of `job_id`. Existing direct training/search APIs remain available.
- Task launch PID updates no longer replace the PID of a worker that has already claimed a job; Linux zombie workers no longer block recovery.
- Existing SQLite ledgers migrate additively; datasets, credentials, existing results and model checkpoints remain local.

### Validation scope

- Static source parsing, diff review and publication-scope inspection only.
- No tests were added or run. No real training, runtime/end-to-end validation, LLM calls or GPU measurements were performed.
- API/Token configuration remains deferred. The main graph uses rule planning; optional LLM research roles are not automatically invoked.

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
