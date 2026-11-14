# Documentation

Start here. Each section answers a different question.

## Architecture

- [Overview](architecture/overview.md): what the monorepo is and where the
  boundaries are.
- [Repository layout](architecture/repo-layout.md): the rules every project and
  lab follows.

## Decisions

Architecture decision records, one decision per file.

- [ADR-001: Python 3.11 baseline](decisions/001-python-runtime.md)
- [ADR-002: Independent projects in a monorepo](decisions/002-monorepo-structure.md)
- [ADR-003: Separate tests from evaluation](decisions/003-testing-and-evals.md)
- [ADR-004: Projects versus labs](decisions/004-projects-and-labs.md)

## Guides

- [Development](guides/development.md): environment, commands, workflow.
- [Testing](guides/testing.md): unit and integration tests.
- [Evaluation](guides/evaluation.md): measuring model quality.

## Operations

- [Runbook](operations/runbook.md): what to do when CI or a service fails.
- [Security notes](operations/security.md): repository controls and secrets.
- [Dependency audit policy](operations/security-audit-policy.md): exceptions and
  expiry.

## History

These are records of past work, preserved for context. They describe the
repository as it was, not as it is.

- [CI recovery plan](history/ci-recovery.md)
- [PR #17 review](history/pr-17-review.md)
