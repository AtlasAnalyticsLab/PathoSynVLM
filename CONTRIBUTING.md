# Contributing to the project website

Thank you for helping keep the PathoSynVLM project page accurate and usable.

## Before editing

- Confirm that you are on `gh-pages`, not `main`.
- Read [DEVELOPMENT.md](DEVELOPMENT.md), especially the content-source and link rules.
- Keep research code changes on `main`; keep website source changes on `gh-pages`.

## Make a website change

1. Create a short-lived branch from `gh-pages`.
2. Edit the root `index.html`, `static/` assets, and maintainer documentation as needed.
3. Run `python3 scripts/validate_site.py .`.
4. Preview with `python3 -m http.server 8000 --directory .` at desktop and mobile widths.
5. Open a pull request with base branch `gh-pages`.

The pull request should explain the visible change, identify the source for new scientific claims or numbers, and include screenshots for layout changes. The validation workflow runs on the pull request. Merging to `gh-pages` deploys automatically.

## Content requirements

- Use claims supported by the current paper or `main` branch.
- Keep metric scales and experimental settings intact; do not combine incomparable runs.
- Do not describe model weights as public until the URL works without authentication.
- Do not claim clinical readiness or state-of-the-art performance without an explicit, current paper source.
- Give every informative image useful alternative text.
- Keep internal asset links relative so the site works below `/PathoSynVLM/`.
- Avoid new runtime dependencies unless maintainers agree that the benefit outweighs the maintenance cost.

Report scientific or code issues in the [main repository issue tracker](https://github.com/AtlasAnalyticsLab/PathoSynVLM/issues).
