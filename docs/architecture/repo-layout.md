# Repository layout

The repository has four kinds of content. Each has one home.

```text
docs/        how the repository works
projects/    installed, tested, runnable code
labs/        exploratory notebooks, never imported
scripts/     repository-level automation
```

## `projects/`

A project is code another engineer can install, test, and run from a clean
checkout. Every project follows the same skeleton. Directories that would be
empty do not exist.

```text
projects/<name>/
├── README.md          # problem, approach, results, run, decisions, limits
├── pyproject.toml     # the only dependency source
├── src/<package>/     # importable code; no scripts, no notebooks
├── tests/             # fast, offline, deterministic
├── configs/           # only if configuration exists
├── notebooks/         # only if a demo notebook exists
├── docker/            # only if the project is containerised
└── evals/             # only if there is a model to measure
```

Rules:

- `pyproject.toml` is the only dependency declaration. Do not add
  `requirements.txt` files; use optional-dependencies instead.
- `src/` contains importable code only. Operational entry points belong in
  `scripts/` or in a `[project.scripts]` console entry point.
- Tests must pass without network access or multi-gigabyte model downloads.
- The README states how to run the project in one command.

## `labs/`

A lab is exploratory: a notebook, a dataset sketch, a framework spike. It is
read, not imported.

- No project may depend on a lab.
- Labs are not packaged and are not part of the dependency audit.
- Their tests only check that the artefacts stay valid (for example, that a
  notebook is parseable).

If a lab becomes load-bearing, it graduates into `projects/` with a package,
tests, and a documented run command. The move is the decision, not a gradual
drift.

## `docs/`

- `architecture/`: how the system is shaped.
- `decisions/`: one ADR per decision, numbered and immutable once accepted.
- `guides/`: how to work on the repository.
- `operations/`: how to run and recover it.
- `history/`: past plans and reviews. Historical records keep their original
  context and are not updated to match current paths.

## `scripts/`

Repository-level automation only. Scripts here may assume the repository root
and must not import project code.

## Automation expectations

- `make check` runs lint, format, compile, link and structure checks.
- `make test` discovers projects and labs instead of listing them.
- CI runs the same commands, so local success means CI success.
