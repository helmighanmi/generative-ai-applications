# Dependency Audit Exception Policy

The `Security / dependency-audit` workflow installs each project on Linux and runs
`pip-audit --strict`. Linux CI is authoritative because Intel macOS cannot resolve the current
`torch` line.

## Default rule

A known vulnerability fails CI. Do not add blanket ignores, `|| true`, or `continue-on-error`.

## Requesting an exception

An exception is allowed only when all of the following are true:

1. No fixed release is installable for the affected dependency.
2. The vulnerable code path is not reachable in the affected service, or a documented mitigation
   removes the exposure.
3. The repository owner (`.github/CODEOWNERS`) approves the exception.
4. The exception has an expiry date no more than 90 days in the future.

Record the exception in the workflow command with a comment:

```yaml
- name: Audit resolved dependencies
  run: |
    pip-audit --strict \
      --ignore-vuln PYSEC-XXXX-YYYY  # owner: @handle; reason: ...; mitigation: ...; expires: 2027-01-31
```

An exception without an owner, reason, mitigation, and expiry is invalid and must be rejected in
review.

## Expiry enforcement

The expiry date is part of the review contract. Before the expiry date the owner must either:

- upgrade to a fixed release and remove the ignore, or
- open a new, reviewed exception with a new justification and expiry.

A weekly scheduled audit keeps the exception visible. Reviewers must block a stale exception even
if the workflow itself is green.

## Platform limitations

Intel macOS is a local-development platform, not a CI exception. The pplx dependency metadata uses
platform markers so Intel macOS can keep a locally installable `torch` while Linux resolves and
audits the fixed line. Do not add a CI ignore for Intel macOS `torch`.
