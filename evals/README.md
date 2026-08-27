<!--
Path: evals/README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# AI Evaluation Convention

Use JSONL datasets with stable case IDs and separate retrieval and generation metrics. A production evaluation record should capture model/provider version, prompt/template version, dataset revision, latency, and any cost/token metrics relevant to the provider.

Project-specific evaluation suites should live beside the project when they depend on its domain or fixtures.
