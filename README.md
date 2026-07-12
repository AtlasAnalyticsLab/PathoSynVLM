# PathoSynVLM project website

[![Website validation](https://github.com/AtlasAnalyticsLab/PathoSynVLM/actions/workflows/validate-site.yml/badge.svg?branch=gh-pages)](https://github.com/AtlasAnalyticsLab/PathoSynVLM/actions/workflows/validate-site.yml)

This orphan `gh-pages` branch contains the project website for [PathoSynVLM](https://github.com/AtlasAnalyticsLab/PathoSynVLM). It intentionally has no shared history with the `main` research-code branch and uses a branch-root static publishing layout.

- Production URL: <https://atlasanalyticslab.github.io/PathoSynVLM/>
- Paper: <https://arxiv.org/abs/2605.30716>
- Research code: <https://github.com/AtlasAnalyticsLab/PathoSynVLM/tree/main>
- Maintainer guide: [DEVELOPMENT.md](DEVELOPMENT.md)

The default `main` working tree contains no website files. A standard multi-branch clone may still fetch the `gh-pages` Git objects; use the single-branch command below when transfer-level isolation matters.

## Repository layout

```text
.
├── index.html                         # page served at the project URL
├── 404.html
├── .nojekyll                          # serve the static files without Jekyll
├── static/
│   ├── css/index.css
│   ├── js/index.js
│   └── images/
├── .github/workflows/validate-site.yml
├── scripts/validate_site.py
├── CONTRIBUTING.md
└── DEVELOPMENT.md
```

The site is plain HTML, CSS, and JavaScript. It has no package manager, generated files, remote font dependency, or frontend framework.

## Quick local preview

From this branch:

```bash
python3 scripts/validate_site.py .
python3 -m http.server 8000 --directory .
```

Open <http://localhost:8000/>. Stop the preview server with <kbd>Ctrl</kbd>+<kbd>C</kbd>.

## Get only the website branch

```bash
git clone --branch gh-pages --single-branch \
  git@github.com:AtlasAnalyticsLab/PathoSynVLM.git \
  PathoSynVLM-website
```

To keep one clone and two working directories instead, see the worktree instructions in [DEVELOPMENT.md](DEVELOPMENT.md#worktree-setup-from-an-existing-code-clone).

## Deployment summary

The repository uses GitHub Pages branch publishing from `gh-pages` and `/(root)`:

1. A pull request or push runs the committed dependency-free `PathoSynVLM website checks` workflow.
2. GitHub's managed `pages build and deployment` workflow publishes the branch root after each push.
3. Root `index.html` is served directly; `.nojekyll` prevents README/Jekyll rendering from replacing it.

These are the only two workflows expected for a normal `gh-pages` push. The Actions page keeps completed runs as history, so multiple rows there do not mean that multiple workflow definitions are active.

Website changes therefore remain independent of `main` and deploy automatically when they reach `gh-pages`. Initial setup, rollback, and troubleshooting instructions are in [DEVELOPMENT.md](DEVELOPMENT.md).

## License

Website content and first-party paper assets are provided under [CC BY-NC-SA 4.0](LICENSE). Linked datasets, third-party models, and externally hosted resources retain their own terms.
