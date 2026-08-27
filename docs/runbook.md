<!--
Path: docs/runbook.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Runbook

## CI fails before tests

Run `make compile`, `make lint`, and `make format-check` locally. Syntax and import-layout failures are intentionally detected before expensive test jobs.

## Model-serving project is unhealthy

1. Check `/health`.
2. Check container logs for model download/load failures.
3. Confirm adequate memory and writable model cache volume.
4. Confirm the model revision/custom code is compatible with the pinned transformer stack.

## RAG results regress

1. Reproduce with the same input document and query set.
2. Compare chunking parameters and retrieval top-k.
3. Confirm embedding model/provider did not change.
4. Run deterministic retrieval evals before changing the generation model.

## Secret accidentally committed

Revoke/rotate the credential first. Removing the file in a later commit is not sufficient; purge repository history if required and verify downstream caches/artifacts.
