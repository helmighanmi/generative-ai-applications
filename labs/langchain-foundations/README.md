# LangChain Foundations

Exploratory notebook for LangChain document loading and framework behaviour.

This is a **lab**, not a project: it is not packaged, not installed, and not
imported by anything under `projects/`. The point is to keep tutorial code and
framework experiments from silently becoming runtime dependencies of unrelated
applications.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install jupyterlab langchain langchain-community
jupyter lab "LangChain document loader-v1.ipynb"
```

## Test

```bash
pytest -q
```

The test only checks that the committed notebook stays parseable and still has
code cells. It does not execute the notebook: that would require network access
and model downloads.

## Graduation rule

If something here becomes reusable, it does not get imported from `labs/`. It
moves to `projects/` with a package, deterministic tests, and a documented run
command.
