<!--
Path: docs/security.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Security Engineering Notes

## Public-repository controls

- `.env`, credentials, caches, local vector indexes, and model weights are ignored.
- Example environment files contain empty placeholders only.
- API error responses avoid exposing arbitrary exception strings where possible.
- Containers run as non-root users.
- CI uses read-only default token permissions unless a job needs more.
- CodeQL and dependency auditing are separate from functional tests.

## AI-specific trust boundaries

Model repositories that require `trust_remote_code=True` execute Python code obtained with model artifacts. Treat the model revision as a software dependency: review it, pin a revision in production, and do not load untrusted model repositories in privileged environments.
