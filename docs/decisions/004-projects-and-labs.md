# ADR-004: Separate projects from labs

**Status:** Accepted

## Context

The repository mixes production-shaped services with a single exploratory
notebook. Both lived under `projects/`, which made the directory name a claim it
could not support: a reader could not tell whether a folder was installed and
tested or simply a place where code had accumulated.

Two failure modes follow from that ambiguity:

1. Tutorial code becomes a runtime dependency because nothing marks it as
   non-production.
2. A folder with no package, no entry point and no deployment story dilutes the
   signal of the folders that do have them.

## Decision

Keep two top-level buckets:

- `projects/` contains code that is installed, tested and run. Every project has
  a `pyproject.toml`, a `src/` package and offline tests.
- `labs/` contains exploratory artefacts. Labs are not packaged, are not part of
  the dependency audit, and are never imported by a project.

Promotion is explicit: a lab becomes a project only when it gains a package,
tests and a documented run command. It moves; it does not get imported in place.

## Consequences

- Adding a project now has a visible cost, which is the point.
- `scripts/run_tests.py` and CI discover both buckets instead of listing them.
- `labs/` is excluded from dependency auditing, because it has no declared
  dependency surface.
- The root README can state what `projects/` means without qualification.
