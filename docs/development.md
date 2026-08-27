<!--
Path: docs/development.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Development

## Baseline

Use Python 3.11. Create isolated environments per subproject when running heavyweight AI dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
make lint
make format-check
make compile
make test
```

For a specific project, follow its README and install only that project's dependencies. This avoids dependency conflicts between experimental LangChain code, scientific document processing, and model-serving stacks.

## Definition of done

A change is complete when behavior is tested, commands are documented, credentials are not hard-coded, CI can reproduce the check, and operational changes include troubleshooting notes.
