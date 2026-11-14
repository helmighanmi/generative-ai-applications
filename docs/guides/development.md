# Development

## Baseline

Use Python 3.11. Create isolated environments per subproject when running heavyweight AI dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install   # tooling plus every project in editable mode
make check     # lint + format check + compile + Markdown link check
make test
```

`make check` runs the same static gates as `CI / repository-quality`. Ruff is pinned; do not rely on a
globally installed version.

The root `pyproject.toml` declares tooling only. Runtime dependencies live in each
project, so there is one source of truth per project and no duplicated pin lists.
To work on a single project, follow its README and install only that project:
`pip install -e 'projects/<name>[dev]'`.

## Definition of done

A change is complete when behavior is tested, commands are documented, credentials are not hard-coded, CI can reproduce the check, and operational changes include troubleshooting notes.
