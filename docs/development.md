# Development

## Baseline

Use Python 3.11. Create isolated environments per subproject when running heavyweight AI dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
make check   # lint + format check + compile + Markdown link check
make test
```

`make check` runs the same static gates as `CI / repository-quality`. Ruff is pinned; do not rely on a
globally installed version.

For a specific project, follow its README and install only that project's dependencies. This avoids dependency conflicts between experimental LangChain code, scientific document processing, and model-serving stacks.

## Definition of done

A change is complete when behavior is tested, commands are documented, credentials are not hard-coded, CI can reproduce the check, and operational changes include troubleshooting notes.
