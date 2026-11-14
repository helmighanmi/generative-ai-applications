# Review record — PR #17, CI and end-to-end recovery

- **PR:** [#17](https://github.com/helmighanmi/generative-ai-applications/pull/17)
  (`fix/ci-e2e-recovery` → `main`)
- **Scope:** 6 commits, 58 files, +1,771 / −556
- **Merged:** `d9b6a0f` (merge commit)
- **Verdict:** approved after the review fixes below

## Verified independently of CI

| Check | Result |
| --- | --- |
| `ruff check projects scripts` (pinned `0.17.4`) | pass |
| `ruff format --check projects scripts` | pass — 38 files |
| `python scripts/check_docs_links.py` | pass — 18 Markdown files |
| CI on the review head | 17/17 checks success |
| Leftover `chromadb` references | none |

The hygiene work was validated as well: `fitz` → `pymupdf`, `StrEnum`, SHA-256 stub seeding,
full-payload image hashing, `resp.choices[0].message.content`, Chroma → FAISS, duplicate root
GitHub configs removed, `.gitignore` / `.python-version` restored, `docs/descisions` →
`docs/decisions`. `/health` (liveness) and `/ready` (503 until the model is loaded) are split
correctly, and both the Dockerfile and Compose use `/ready`.

## Findings fixed before merge (`1451bf9`)

1. **The ML stack was not audited for `multimodal-rag-pipeline`.** The audit job ran
   `pip install .`, while `openai`, `sentence-transformers`, `transformers` and
   `huggingface-hub` live in the `ai` extra. The matrix now carries a per-entry `install`
   field and the project audits `.[ai]`.
2. **`.[dev]` no longer installs the AI stack, but the README said it did.** The project
   README and [`docs/runbook.md`](../operations/runbook.md) now document `pip install -e '.[dev,ai]'`
   for real-provider use.
3. **`integration-smoke` duplicated `project-tests (multimodal-rag-pipeline)`.** The job now
   installs `.[ai,dev]` and asserts that `openai`, `sentence_transformers`, `torch` and
   `transformers` import, so it proves something the matrix does not.
4. **The fake backend impersonated the real model.** `FakeEmbeddingModel.model_name` is now
   `fake/deterministic-embedder`, so `/ready` and `/embed` no longer report the 0.6B model
   when the deterministic backend is active.
5. **`pymupdf>=1.24` was below the first release that ships the `pymupdf` module name.** The
   floor is now `1.24.3` in the root and both project manifests.

## Open follow-ups

These were raised in review and deliberately left out of the PR:

- **Pin drift.** `huggingface-hub` is `>=1.31,<2` in `rag-qa-bot-langchain` but `>=1.31,<3`
  elsewhere; `langchain-text-splitters` is `>=0.3,<2` there versus `>=1.1.2,<2` in the
  multimodal pipeline. Aligning these needs a resolver run, not just an edit.
- **Two sources of truth for pplx runtime dependencies.** `projects/pplx-embed-project/api/requirements.txt`
  and its `pyproject.toml` are consistent today but can drift. Either assert equality in CI or
  keep a single manifest.
- **CodeQL on fork pull requests.** `security-events: write` is job-scoped, but GitHub
  downgrades it for `pull_request` from forks, so the upload step would fail. Only relevant if
  outside contributions are accepted.
- **`.pre-commit-config.yaml` revs.** Dependabot has no pre-commit ecosystem, so hook updates
  (currently `pre-commit-hooks` `v5.0.0`) stay a manual `pre-commit autoupdate`.

## Dependency audit policy

Any exception must follow [`docs/security-audit-policy.md`](../operations/security-audit-policy.md): an
owner, a reason, a mitigation and an expiry date of at most 90 days. No blanket ignores and no
`continue-on-error`.
