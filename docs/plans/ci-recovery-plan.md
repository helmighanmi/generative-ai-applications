<!--
Path: docs/plans/ci-recovery-plan.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# End-to-End Fix Plan: CI Recovery and Runtime Hardening

**Status:** Implemented (2026-11-08)
**Date:** 2026-11-06
**Baseline commit:** `5e5d95f` (`code cleaning and project refactoring`)
**Goal:** make the repository reproducibly green from a clean checkout: root install, static checks, unit tests, integration smoke tests, Docker builds, and security audit.

---

## 1. Success criteria (definition of done)

A clean clone must pass all of these without undocumented manual steps:

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

And GitHub Actions must be green on `main`:

- `CI / repository-quality`
- `CI / project-tests` for all five projects
- `CI / docker-build` for multimodal RAG, pplx API, and pplx UI
- `CI / integration-smoke` (new)
- `Security / codeql`
- `Security / dependency-audit`

Additional requirements:

- No secrets, caches, model weights, or local data are tracked.
- No test is skipped, no job is marked `continue-on-error`, and no audit failure is hidden with a blanket ignore.
- Any vulnerability exception is documented with an owner, reason, and expiry date.
- README and `docs/` links resolve.
- The runtime path is actually tested end to end: build an index, persist it, reload it in a new process, retrieve, and answer.

---

## 2. Current failures and evidence

### 2.1 GitHub Actions status

| Workflow / run | Result | Detail |
|---|---|---|
| `CI` on `main` — run `33060031414` | **failure** | `repository-quality` fails at `Ruff lint`; `Ruff format check` is skipped. |
| `CI` project tests | pass | All five projects passed at the baseline commit. |
| `CI` docker builds | pass | All three images built successfully. |
| `Security` schedule — run `41453556959` | **failure** | `codeql` passes; `dependency-audit` fails. Weekly security has been red since at least August 2026. |

### 2.2 Repository-quality failure

Reproduced locally with the same command CI runs:

```bash
ruff check projects scripts
# Found 95 errors. (EXIT 1)
```

Breakdown of the 95 errors:

| Rule | Count | Meaning |
|---|---:|---|
| `C408` | 40 | unnecessary `dict()` call in Plotly/Streamlit code |
| `UP006` | 13 | deprecated `typing.List` / `typing.Dict` |
| `BLE001` | 11 | blind `except Exception` |
| `I001` | 8 | unsorted imports |
| `UP035` | 7 | deprecated `typing` imports |
| `UP045` | 6 | use `X | None` instead of `Optional[X]` |
| `F401` | 5 | unused imports |
| `F541` | 3 | f-string without placeholders |
| `RUF059` | 1 | unused unpacked variable |
| `EXE001` | 1 | shebang present but file not executable |

Files with the largest number of errors:

- `projects/pplx-embed-project/scripts/export_projector.py` — 41
- `projects/pplx-embed-project/streamlit_app/app.py` — 19
- `projects/multimodal-rag-pipeline/src/multimodal_rag/pdf_parsing.py` — 12
- `projects/multimodal-rag-pipeline/src/multimodal_rag/generator.py` — 9
- `projects/multimodal-rag-pipeline/src/multimodal_rag/embedding.py` — 7

**Important root cause:** `ruff` uses the nearest `pyproject.toml`. Each project has its own `[tool.ruff]` section, but those sections do not repeat the root `lint.select`, `lint.ignore`, or `exclude` settings. As a result, the root exclusions for notebooks and the Streamlit/projector files are ignored, and the effective rule set changes per project.

Proof: forcing the root config produces only **31** errors instead of 95:

```bash
ruff check projects scripts --config pyproject.toml
# Found 31 errors. (25 fixable)
```

That also removes notebook findings and applies the intended exclusions.

Current format-check result:

```bash
ruff format --check projects scripts
# 11 files would be reformatted (with root config: 6 files)
```

`Ruff format check` did not run in CI because lint failed first. It will fail immediately after lint is fixed.

### 2.3 Dependency-audit failure

Reproduced locally:

```bash
pip-audit -r projects/pplx-embed-project/api/requirements.txt
# Found 33 known vulnerabilities in 2 packages
```

| Package | Resolved version | Vulnerabilities | Notes |
|---|---|---:|---|
| `transformers` | 4.57.6 | 9 | Fixes require `transformers` 5.x |
| `torch` | 2.2.2 | 24 | Fixes require newer `torch`; 2.2.2 is the last macOS x86_64 wheel |

Why `torch` resolved to 2.2.2 locally: this workstation is Intel macOS (`x86_64`), and PyTorch stopped publishing newer wheels for that platform. On the Ubuntu CI runner, the requirement can resolve to a newer fixed release. This means the audit must be authoritative on Linux CI, and local Intel macOS needs an explicit platform marker.

The current requirements are also non-reproducible: broad lower bounds (`transformers>=4.51,<5`, `torch>=2.2,<3`) mean the audited version can change over time.

### 2.4 Root install is broken

The documented quick start fails before installing any dependency:

```text
error: Multiple top-level packages discovered in a flat-layout: ['evals', 'projects'].
```

The root `pyproject.toml` declares a distribution but does not tell setuptools which packages to include. `pip install -e '.[dev]'` therefore cannot build the root development environment.

### 2.5 Repository hygiene regressions from the last refactor

The refactor commit appears to have flattened files downloaded from the browser:

| Tracked file | Should be |
|---|---|
| `download` | `.gitignore` (it is the `.gitignore` contents plus AI artifacts) |
| `download (1)` | `.python-version` (contains `3.11`) |
| `docs/descisions/` | `docs/decisions/` |
| `pre-commit-config.yaml` | `.pre-commit-config.yaml` (pre-commit only reads the dotfile) |
| `CODEOWNERS` | `.github/CODEOWNERS` (duplicate) |
| `dependabot.yml` | `.github/dependabot.yml` (duplicate) |
| `pull_request_template.md` | `.github/pull_request_template.md` (duplicate) |
| `security.yml` | `.github/workflows/security.yml` (duplicate, not executed) |
| `.github/ci.yml` | `.github/workflows/ci.yml` (duplicate, not executed) |

Consequences:

- `.env`, `.venv`, `__pycache__`, model weights, FAISS indexes, and local data are currently **not ignored**, despite what `README.md`, `docs/security.md`, and `SECURITY.md` claim.
- All README links to `docs/decisions/*` are dead.
- The root pre-commit configuration is never loaded.
- Dependabot only covers three of five projects.

### 2.6 End-to-end runtime defects not covered by tests

These are not currently failing CI, but they break the "runnable from a clean checkout" goal:

1. **Stub embeddings are not deterministic across processes.**
   `_deterministic_stub_vector` seeds NumPy from Python's salted `hash(key)`. Two calls in one test process match, but `python main.py build` and a later `python main.py query` run with different hash seeds. The persisted FAISS index is therefore semantically meaningless across process boundaries.

2. **Image stubs collapse to the same vector.**
   The stub embeds only `image_b64[:64]`, which is mostly the shared PNG base64 header. Different page images can produce identical stub vectors.

3. **`/health` returns 200 while the model is still loading.**
   `docker compose` uses the health check to gate Streamlit with `depends_on: service_healthy`, so the UI can start before the API is ready. Either return 503 while loading or add a separate `/ready` endpoint used by the health check.

4. **OpenAI generation response access is likely wrong for the modern SDK.**
   `resp.choices[0].message["content"]` should use attribute access (`.content`) for current `openai` versions.

5. **No real integration test.**
   Tests cover stubs and in-process fakes, but nothing verifies parse -> embed -> FAISS build -> save -> reload -> retrieve -> generate across processes or containers.

---

## 3. Root-cause summary

| # | Root cause | Impact |
|---|---|---|
| R1 | Refactor copied downloaded files under the wrong names | Missing `.gitignore`, dead links, duplicate configs, inactive pre-commit |
| R2 | Ruff configuration drift across nested projects | 95 lint errors instead of 31; intended exclusions ignored; format gate would fail next |
| R3 | Tool versions are not pinned (`ruff>=0.12,<1`, `pip-audit` unpinned) | CI behavior changes over time; pre-commit and CI disagree |
| R4 | Vulnerable dependency ceilings (`transformers<5`, old torch) | Security job red every week |
| R5 | No lockfile / audit is not reproducible | Audit results depend on platform and date |
| R6 | Root project is not a valid installable package | Documented quick start fails |
| R7 | Tests are unit-only and do not test cross-process determinism or readiness | Real pipeline defects stay hidden |

---

## 4. Fix plan

### Phase 0 — Restore repository hygiene (first PR, ~30 minutes)

**Branch:** `chore/repo-hygiene`
**Purpose:** restore the files that the refactor displaced before changing code.

Tasks:

1. Restore `.gitignore` and `.python-version`:
   ```bash
   git mv download .gitignore
   git mv 'download (1)' .python-version
   ```
   Verify with:
   ```bash
   git check-ignore -v .env .venv model/ '**/data/index/*.faiss'
   git status --ignored
   ```
   Remove the "AI/ML local artifacts" block from `.gitignore` only if it is duplicated elsewhere; keep the useful entries.

2. Fix the docs directory name:
   ```bash
   git mv docs/descisions docs/decisions
   ```
   Then verify every `docs/decisions/...` link in `README.md`.

3. Activate pre-commit:
   ```bash
   git mv pre-commit-config.yaml .pre-commit-config.yaml
   ```
   - Update the ruff hook revision to the same exact version used by CI (see Phase 1).
   - Remove or align `projects/pplx-embed-project/.pre-commit-config.yaml`.
   - Either commit a real `.secrets.baseline` and stop ignoring it, or remove the `detect-secrets` hook from the nested config. Do not keep a hook that always fails because its baseline is missing.

4. Remove duplicate root copies:
   - `CODEOWNERS`
   - `dependabot.yml`
   - `pull_request_template.md`
   - `security.yml`
   - `.github/ci.yml`

   Keep the canonical files under `.github/`. Confirm GitHub still recognizes `.github/CODEOWNERS` after the change.

5. Fix environment examples:
   - Rename root `env.example` to `.env.example` (or add `.env.example` and keep `env.example` as a pointer).
   - Add `projects/pplx-embed-project/.env.example` because its README tells users to copy it. Include:
     ```dotenv
     MODEL_CACHE_DIR=./model
     MODEL_AUTO_DOWNLOAD=false
     MODEL_REVISION=
     CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
     API_URL=http://localhost:8000
     ```

6. Update `CHANGELOG.md` with an `Unreleased` entry describing the recovery.

**Acceptance:** no tracked file named `download`; docs links resolve; `pre-commit run --all-files` can run; no duplicate root-level config artifacts.

---

### Phase 1 — Make the Ruff gate green and reproducible (second PR, ~1–2 hours)

**Branch:** `fix/ci-ruff-gate`
**Purpose:** one lint configuration, one pinned version, zero lint/format errors.

Tasks:

1. **Centralize Ruff configuration.** Remove the `[tool.ruff]` sections from these files:
   - `projects/document-parsing/pyproject.toml`
   - `projects/langchain/pyproject.toml`
   - `projects/multimodal-rag-pipeline/pyproject.toml`
   - `projects/pplx-embed-project/pyproject.toml`
   - `projects/rag-qa-bot-langchain/pyproject.toml`

   All currently repeat the same `target-version = "py311"` and `line-length = 100` as the root, so nothing is lost. If a project ever needs an override, add `extend = "../../pyproject.toml"` instead of a standalone config.

2. **Pin Ruff exactly.** Use the version CI currently resolves (`0.17.4`) in:
   - root `pyproject.toml` dev extras
   - `.github/workflows/ci.yml`
   - `.pre-commit-config.yaml`

   Example:
   ```toml
   dev = ["ruff==0.17.4", ...]
   ```

3. **Fix lint errors** in this order:
   ```bash
   ruff check projects scripts --fix
   ruff format projects scripts
   ```
   Then fix the remaining manual items:
   - `UP042` in `classifier.py`: replace `class PageType(str, Enum)` with `class PageType(StrEnum)` from `enum`.
   - `UP006`, `UP035`, `UP045`: use built-in generics and `X | None`.
   - `F401`: remove unused imports.
   - `RUF059`: rename `alt_cs` to `_alt_cs`.
   - `I001`: accept auto-fix.
   - If the Streamlit/projector files are brought into scope: replace `dict(...)` with literals (`C408`), remove f-string prefixes (`F541`), and either narrow broad `except Exception` (`BLE001`) or add a justified `# noqa: BLE001` at the UI/API boundary.
   - `EXE001`: either `chmod +x scripts/export_projector.py` or remove the shebang.

4. **Decide the notebook policy.** Recommended: keep `**/*.ipynb` excluded from Ruff. If notebooks are meant to be checked, move them to a separate, explicitly scoped command so a notebook format change cannot block application PRs.

5. **Remove root exclusions only if the code is fixed.** Today the root excludes `streamlit_app/app.py` and `scripts/export_projector.py`. Prefer removing those exclusions after fixing the files so the gate covers all Python source. Keep the notebook exclusion.

6. Add a local convenience target so developers run exactly what CI runs:
```make
check:
	ruff check projects scripts
	ruff format --check projects scripts
	python -m compileall -q projects scripts
```

**Acceptance:**

```bash
ruff check projects scripts
ruff format --check projects scripts
python -m compileall -q projects scripts
```

All exit 0, and `CI / repository-quality` is green.

---

### Phase 2 — Fix the failed security pipeline (~2–4 hours)

**Branch:** `fix/dependency-audit`
**Purpose:** remove known vulnerabilities and make dependency auditing deterministic and complete.

Tasks:

1. **Upgrade the embedding stack in one place.**
   Update both `projects/pplx-embed-project/pyproject.toml` and `projects/pplx-embed-project/api/requirements.txt` together. Minimum targets verified against OSV on 2026-11-06:

   | Package | New range | Reason |
   |---|---|---|
   | `transformers` | `>=5.22.2,<6` | Current 4.x advisories are fixed only in 5.x |
   | `sentence-transformers` | `>=6.2,<7` | Current release supports `transformers` 5.x |
   | `huggingface-hub` | `>=1.31,<3` | Required by `transformers` 5.22 |
   | `torch` | `>=2.15,<3` on Linux; `>=2.2,<2.3` on Intel macOS | Newer torch has no Intel macOS wheels |

   Example platform marker:
   ```toml
   "torch>=2.15,<3; not (platform_system == 'Darwin' and platform_machine == 'x86_64')",
   "torch>=2.2,<2.3; platform_system == 'Darwin' and platform_machine == 'x86_64'",
   ```

   Validate that `perplexity-ai/pplx-embed-v1-0.6b` still loads with `sentence-transformers` 6 / `transformers` 5. This is the main upgrade risk. Run a real model smoke test on a machine with the model cached:
   ```bash
   cd projects/pplx-embed-project
   MODEL_AUTO_DOWNLOAD=true python scripts/download_model.py
   MODEL_AUTO_DOWNLOAD=false python -c "
   from api.embedder import EmbeddingModel
   m = EmbeddingModel()
   print(m.encode(['hello'])[0][:3])
   "
   ```
   If the model is incompatible, stop and choose a documented fallback; do not silently downgrade and leave the audit red.

2. **Pin the model revision.** Set a default reviewed revision in `api/embedder.py` and compose, e.g. the current model SHA `2c4d510dd4a732063c31a0f70193e35067b51fd8`, or require `MODEL_REVISION` in production. This reduces supply-chain and reproducibility risk.

3. **Make the audit reproducible.**
   Recommended: add lockfiles per service project with `pip-compile` (or `uv lock`), then audit the lock:
   ```bash
   pip-audit -r projects/pplx-embed-project/requirements.lock --strict
   ```
   Minimum acceptable alternative: pin the top-level versions and run `pip-audit` on the exact installed environment in CI:
   ```bash
   pip install -e '.[dev]'
   pip-audit --strict
   ```

4. **Audit every project, not only the pplx API.**
   Update `.github/workflows/security.yml` to run a matrix over:
   - `projects/document-parsing`
   - `projects/langchain`
   - `projects/multimodal-rag-pipeline`
   - `projects/pplx-embed-project`
   - `projects/rag-qa-bot-langchain`

   At minimum audit the service/runtime dependency sets; do not claim repository-wide auditing while only one requirements file is checked.

5. **Pin the audit tool.**
   Replace `pip install --upgrade pip pip-audit` with an exact or bounded version:
   ```bash
   pip install 'pip-audit==2.10.1'
   ```
   Keep `--strict` so resolution failures fail the job instead of silently passing.

6. **Create a vulnerability exception policy.**
   Add `docs/security-audit-policy.md` with:
   - who can approve an exception,
   - required reason and mitigation,
   - expiry date,
   - command-line form (`--ignore-vuln PYSEC-...`),
   - a CI check that fails when an exception expires.

   No permanent blanket ignores. Intel macOS torch is a local-development limitation, not a CI exception; document it and audit on Linux.

7. **Expand Dependabot coverage.**
   Add `/projects/document-parsing`, `/projects/langchain`, and `/projects/rag-qa-bot-langchain` to `.github/dependabot.yml`. Group minor/patch updates to reduce PR noise.

**Acceptance:** `Security / dependency-audit` green on `main`, green on the weekly schedule, and no unignored vulnerabilities in the resolved Linux dependency set.

---

### Phase 3 — Repair root packaging and the documented quick start (~30 minutes)

**Branch:** `fix/root-packaging`
**Purpose:** make `pip install -e '.[dev]'` work at the repository root.

Tasks:

1. Explicitly disable package auto-discovery in root `pyproject.toml` (the root is a tooling workspace, not an importable distribution):
   ```toml
   [tool.setuptools]
   packages = []
   ```
   or:
   ```toml
   [tool.setuptools.packages.find]
   include = []
   ```

2. Verify in a completely clean environment:
   ```bash
   python3.11 -m venv .venv-clean
   source .venv-clean/bin/activate
   python -m pip install --upgrade pip
   pip install -e '.[dev]'
   make test
   ```

3. Add `pythonpath` to each project's pytest config so project tests work both via the `scripts/run_tests.py` wrapper and from inside the project:
   ```toml
   [tool.pytest.ini_options]
   testpaths = ["tests"]
   pythonpath = ["src"]   # or ["."] for pplx
   ```

4. Confirm `make test` is not relying on the caller's installed packages. The wrapper currently sets `PYTHONPATH` to `src`, which is good; keep it and add a test that runs a project test from the repository root.

**Acceptance:** clean-clone quick start succeeds; `make test` passes; root install no longer raises setuptools discovery errors.

---

### Phase 4 — Fix end-to-end runtime behavior and add smoke tests (~3–4 hours)

**Branch:** `fix/runtime-e2e`
**Purpose:** make the offline demo genuinely work across processes and containers.

Tasks:

1. **Make stub embeddings cross-process deterministic.**
   Replace Python's salted `hash()` with a stable digest:
   ```python
   import hashlib

   def _stable_seed(key: str) -> int:
       return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
   ```
   Use it in `_deterministic_stub_vector`.

2. **Fix image stub inputs.** Hash the full image payload, not `image_b64[:64]`, so different images get different vectors.

3. **Add regression tests.**
   - A subprocess test that builds an index in one process and queries it in another, asserting the expected text is retrieved.
   - A test that two different image payloads produce different stub vectors.
   - A test that the same payload in two processes produces the same vector (or assert a fixed expected vector).

4. **Fix readiness.**
   - Option A: make `/health` return HTTP 503 with `status="loading"` while the model is not loaded.
   - Option B: add `/ready` returning 503 until loaded, and point the Docker health check / compose condition at `/ready`.
   - Keep `/health` useful for liveness if you choose option B.
   - Update `tests/test_api.py` accordingly.

5. **Fix the OpenAI response handling** in `multimodal_rag/generator.py`:
   ```python
   return resp.choices[0].message.content
   ```
   Add a fake-response unit test that exercises the OpenAI branch without a network call.

6. **Add a real multimodal integration test.**
   Use a tiny generated PDF and a temporary directory:
   - configure `use_tables=false`, `use_images=false`, `use_page_images=false` for speed,
   - build text chunks, embed with stub, build FAISS, save,
   - reload the store, query, ask, and assert the answer contains the retrieved context.
   This should run in CI without Java, network, or model downloads.

7. **Add a document-parsing CLI smoke test.** Generate a two-page PDF with PyMuPDF, run the CLI in a subprocess, and validate one JSON object per page.

8. **Add a container smoke test for the API.** Introduce an explicit fake-embedder mode for CI (for example `EMBEDDER_BACKEND=fake`) so the image can start without downloading the model. Then:
   ```bash
   docker run -d --rm -e EMBEDDER_BACKEND=fake -p 8000:8000 pplx-api
   curl --fail http://localhost:8000/health
   curl --fail -X POST http://localhost:8000/embed -H 'Content-Type: application/json' \
     -d '{"texts":["hello"]}'
   ```
   This turns the current "build only" Docker job into a real readiness check.

9. **Add a CI `integration-smoke` job.** Run after project tests, with a timeout, and fail on non-zero exit.

10. **Reduce unnecessary CI install weight.**
    - `projects/multimodal-rag-pipeline`: move `multimodal-rag-pipeline[ai]` out of `dev`; tests use stubs. Keep `ai` as an explicit extra.
    - Keep pplx runtime dependencies in its own extra; tests already inject a fake embedder.
    - This shortens CI and makes failures easier to attribute.

**Acceptance:** new tests fail before the fixes and pass after; build -> persist -> new-process query works; container starts and answers `/health` and `/embed` with the fake backend.

---

### Phase 5 — CI hardening, docs, and final verification (~2 hours)

**Branch:** `ci/docs-hardening`

Tasks:

1. **CI**
   - Add `timeout-minutes` to each job.
   - Add `pip check` after installs.
   - Add `docker compose config` validation for the pplx stack.
   - Add the API container smoke test from Phase 4.
   - Scope permissions per job (`contents: read`; `security-events: write` only for CodeQL/SARIF).
   - Optionally pin third-party actions by commit SHA.
   - Add a `dependency-review` job for pull requests.

2. **Docs**
   - Update `README.md` quick start and quality-gate list.
   - Fix all `docs/decisions/...` links after the directory rename.
   - Add `docs/security-audit-policy.md`.
   - Update `docs/runbook.md` with concrete steps for "CI ruff failure", "dependency audit failure", and "model load failure".
   - Update pplx README: `.env.example`, model revision, fake-backend smoke mode, Intel macOS torch limitation.
   - Update project READMEs to match actual commands and extras.
   - Add a "last verified" date and the exact Python version.

3. **Add a docs link check.**
   A small script or a CI step that verifies relative Markdown links is enough. Do not merge documentation with known dead links.

4. **Update Dependabot and CODEOWNERS** after removing duplicates. Confirm the canonical `.github/` files are the only ones.

5. **Final full verification** from a clean clone:
   ```bash
   make lint
   make format-check
   make compile
   make test
   cd projects/multimodal-rag-pipeline && python -m pytest -q
   cd ../pplx-embed-project && python -m pytest -q
   cd ../document-parsing && python -m pytest -q
   cd ../langchain && python -m pytest -q
   cd ../rag-qa-bot-langchain && python -m pytest -q
   ```
   Then verify the GitHub workflows on `main`.

**Acceptance:** every workflow green; docs links valid; no duplicate configs; changelog updated.

---

## 5. Recommended PR sequence

| Order | PR | Scope | Estimate |
|---|---|---|---|
| 1 | `chore/repo-hygiene` | `.gitignore`, `.python-version`, docs directory, duplicate configs, pre-commit activation | 30 min |
| 2 | `fix/ci-ruff-gate` | Ruff config consolidation, pin, lint/format fixes | 1–2 h |
| 3 | `fix/root-packaging` | Root setuptools config, clean-install verification | 30 min |
| 4 | `fix/dependency-audit` | Dependency upgrade, lock/audit policy, Dependabot coverage | 2–4 h |
| 5 | `fix/runtime-e2e` | Deterministic stubs, readiness, OpenAI fix, integration tests | 3–4 h |
| 6 | `ci/docs-hardening` | Smoke jobs, docs, runbook, final cleanup | 2 h |

Phase 1 and Phase 3 can be combined if a fast green CI is the priority. Do not combine the dependency upgrade with the lint cleanup: the dependency PR will be large and the lint PR should be easy to review and revert.

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `transformers` 5 / `sentence-transformers` 6 breaks the pplx custom model | medium | high | Run the model load smoke test before merging; keep a fallback branch/pin and document it |
| New torch wheel unavailable on Intel macOS | high | medium | Platform markers; treat Linux CI audit as authoritative; document local limitation |
| Lockfile update surfaces new transitive vulnerabilities | medium | medium | Run `pip-audit` in the same PR; update or add a time-boxed triage entry |
| Removing nested Ruff config changes project-local behavior | low | low | All five configs currently duplicate root settings; verify `ruff check` from each project |
| Formatting notebooks creates a huge diff | medium | low | Keep notebooks excluded from Ruff; handle them in a separate PR if desired |
| Deleting duplicate root configs breaks an automation that reads them | low | medium | `.github/` canonical files already exist; verify GitHub settings after the PR |
| Container smoke test needs a model download | high | high | Add the fake-embedder backend for CI; keep the real-model test manual/nightly |
| Root packaging change affects editable installs | low | low | Verify in a clean venv; root is not imported by any project |

---

## 7. What not to do

- Do **not** add `continue-on-error`, `|| true`, or `pytest -k 'not ...'` to make the pipeline green.
- Do **not** lower test coverage or delete tests to avoid fixing runtime behavior.
- Do **not** permanently ignore audit findings without a documented, expiring triage entry.
- Do **not** downgrade to a known-vulnerable version just because it is the only one available on Intel macOS; pin that limitation locally.
- Do **not** exclude an entire project from linting without a written reason.
- Do **not** mix the large dependency upgrade with the formatting/lint cleanup in one PR.

---

## 8. Verification commands

```bash
# Repository quality
ruff check projects scripts
ruff format --check projects scripts
python -m compileall -q projects scripts

# Root developer environment
python3.11 -m venv .venv-clean
source .venv-clean/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
make test

# Per-project tests
for project in document-parsing langchain multimodal-rag-pipeline pplx-embed-project rag-qa-bot-langchain; do
  (cd "projects/$project" && python -m pytest -q)
done

# Security
pip-audit -r projects/pplx-embed-project/requirements.lock --strict

# Containers
docker compose -f projects/pplx-embed-project/docker-compose.yml config
docker build -f projects/pplx-embed-project/docker/Dockerfile.api projects/pplx-embed-project
docker build -f projects/pplx-embed-project/docker/Dockerfile.streamlit projects/pplx-embed-project
docker build -f projects/multimodal-rag-pipeline/Dockerfile projects/multimodal-rag-pipeline
```

GitHub checks:

- `https://github.com/helmighanmi/generative-ai-applications/actions/workflows/ci.yml`
- `https://github.com/helmighanmi/generative-ai-applications/actions/workflows/security.yml`

---

## 9. Final acceptance checklist

- [ ] `download` and `download (1)` are gone; `.gitignore` and `.python-version` are tracked.
- [ ] `docs/decisions/` exists and every README link resolves.
- [ ] Only canonical `.github/` configuration files exist.
- [ ] `.pre-commit-config.yaml` is active and uses the same Ruff version as CI.
- [ ] Root `pip install -e '.[dev]'` succeeds in a clean venv.
- [ ] `ruff check` and `ruff format --check` pass on all Python source.
- [ ] `CI / repository-quality` is green on `main`.
- [ ] All five project test jobs are green.
- [ ] Three Docker image builds are green.
- [ ] New integration smoke job is green.
- [ ] `Security / codeql` is green.
- [ ] `Security / dependency-audit` is green on `main` and on schedule.
- [ ] Stub embeddings are deterministic across processes.
- [ ] Build -> save -> reload -> query works across two processes.
- [ ] `/health` (or `/ready`) does not report ready before the model is loaded.
- [ ] No unignored vulnerabilities; exceptions documented with expiry.
- [ ] Runbook and security audit policy are updated.
- [ ] `CHANGELOG.md` records the recovery.

---

## 10. Evidence appendix

Commands used during diagnosis:

```bash
git log --oneline -20
git status
ruff check projects scripts
ruff format --check projects scripts
pip-audit -r projects/pplx-embed-project/api/requirements.txt
python scripts/run_tests.py
python -m compileall -q projects scripts
```

Observed results:

- `ruff check projects scripts`: 95 errors, exit 1.
- `ruff format --check projects scripts`: 11 files would be reformatted, exit 1.
- `pip-audit -r projects/pplx-embed-project/api/requirements.txt`: 33 vulnerabilities in 2 packages, exit 1.
- Root `pip install -e '.[dev]'`: fails with "Multiple top-level packages discovered in a flat-layout: ['evals', 'projects']."
- `python scripts/run_tests.py` with manually installed root dev dependencies: all project test suites pass (document-parsing, langchain, multimodal-rag-pipeline, pplx-embed-project, rag-qa-bot-langchain).
- `python -m compileall -q projects scripts`: exit 0.

GitHub Actions evidence:

- CI run `33060031414` on `main` (`5e5d95f`): `repository-quality` failure at Ruff lint; all other jobs pass.
- Security run `41453556959` on `main` (`5e5d95f`): `codeql` pass; `dependency-audit` failure.
