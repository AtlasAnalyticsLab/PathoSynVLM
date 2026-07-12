# PathoSynVLM project website

[![Website deployment](https://github.com/AtlasAnalyticsLab/PathoSynVLM/actions/workflows/deploy-pages.yml/badge.svg?branch=gh-pages)](https://github.com/AtlasAnalyticsLab/PathoSynVLM/actions/workflows/deploy-pages.yml)

This orphan `gh-pages` branch contains the project website for [PathoSynVLM](https://github.com/AtlasAnalyticsLab/PathoSynVLM). It intentionally has no shared history with the `main` research-code branch.

- Production URL: <https://atlasanalyticslab.github.io/PathoSynVLM/>
- Paper: <https://arxiv.org/abs/2605.30716>
- Research code: <https://github.com/AtlasAnalyticsLab/PathoSynVLM/tree/main>
- Maintainer guide: [DEVELOPMENT.md](DEVELOPMENT.md)

Keeping the histories separate means the default `main` working tree contains only the Python project. A standard multi-branch clone may still fetch the `gh-pages` Git objects; use the single-branch command below when transfer-level isolation matters. Website maintainers can clone this branch alone or attach it as a second Git worktree.

## Repository layout

```text
.
├── .github/workflows/deploy-pages.yml  # validate and deploy on gh-pages pushes
├── site/                               # exact artifact published to Pages
│   ├── index.html
│   ├── 404.html
│   ├── robots.txt
│   ├── sitemap.xml
│   └── static/
│       ├── css/site.css
│       ├── js/site.js
│       └── images/
├── scripts/validate_site.py            # dependency-free local/CI validation
├── CONTRIBUTING.md
└── DEVELOPMENT.md
```

The site is plain HTML, CSS, and JavaScript. It has no package manager, generated files, remote font dependency, or frontend framework.

## Quick local preview

From this branch:

```bash
python3 scripts/validate_site.py site
python3 -m http.server 8000 --directory site
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

Every push to `gh-pages` runs the committed workflow:

1. Validate HTML metadata, local paths, anchors, required files, the sitemap, and asset sizes.
2. Upload only `site/` as the GitHub Pages artifact.
3. Deploy that artifact to the protected `github-pages` environment.

Pull requests targeting `gh-pages` run validation but do not deploy. The initial repository setup requires an administrator to select **GitHub Actions** under **Settings → Pages → Source**. Full setup, rollback, and troubleshooting instructions are in [DEVELOPMENT.md](DEVELOPMENT.md).

## License

Website content and first-party assets are provided under [CC BY-NC-SA 4.0](LICENSE). Linked datasets, third-party models, and externally hosted resources retain their own terms.
