<!--
Path: projects/langchain/README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# LangChain Foundations

Learning/experimentation area for LangChain document loading and framework exploration. It is intentionally separated from production-style services so tutorial code and framework experiments cannot silently become runtime dependencies of unrelated applications.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
jupyter lab
```

Notebook quality checks validate that committed notebooks remain parseable. Runtime code that becomes reusable should graduate into `src/langchain_foundations/` with deterministic tests.
