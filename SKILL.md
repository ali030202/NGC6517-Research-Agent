---
name: ngc6517-research-agent
description: Operate bounded, validation-driven NGC6517 few-shot experiments using the local CLI or MCP tools.
---

# NGC6517 Research Agent protocol

This file describes the project's software workflow for a host assistant.

1. Inspect data and capabilities before submitting an experiment.
2. Prepare a plan with an explicit trial count, timeout and validation target.
3. Start a managed background job; retain its ID and inspect state/logs as needed.
4. Select and stop by **validation F1**, never by test results.
5. Keep failed experiments and source/configuration hashes as evidence.
6. Stage changes as proposals and show diffs before explicit application.
7. Freeze the selected model before explicitly requested final test evaluation.
8. Report only recorded results; missing evidence must remain unavailable.

## Boundaries

- API/Token setup is optional and deferred; local planning, retrieval and reports need no credentials.
- Static checks do not execute candidates or establish safety or numerical correctness. Review candidate code before training.
- Online literature search and PDF downloading are explicit operations, never import-time side effects.
- Drafts, reviewer responses and hypotheses do not establish novelty, validity or guaranteed gains.
- Do not publish datasets, credentials, weights, private logs or copyrighted full-text corpora.

See `python agent.py --help` and `README.md`.
