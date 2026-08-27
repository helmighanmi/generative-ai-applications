<!--
Path: SECURITY.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Security Policy

Do not open a public issue for a vulnerability that exposes credentials or creates a practical remote-exploitation path. Use GitHub's private vulnerability reporting feature when it is enabled for this repository.

## Repository rules

- Secrets are supplied only through environment variables or external secret managers.
- `.env` files and model caches are ignored.
- External model code requiring `trust_remote_code=True` is documented as a trust boundary and should be pinned/reviewed before production use.
- Public API deployments must replace permissive development CORS settings with explicit origins.
- Dependency and CodeQL scans run in GitHub Actions.
