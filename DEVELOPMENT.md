# Website development and operations

This document is the operational source of truth for the PathoSynVLM project website. It covers branch isolation, local editing, GitHub Pages deployment, rollback, and launch checks.

## 1. Architecture and branch policy

The repository has two intentionally independent histories:

| Branch | Purpose | Checked out in the default working tree? | Deployment effect |
|---|---|---:|---|
| `main` | Python package, paper configs, research docs, and code assets | Yes | None |
| `gh-pages` | Static website, website checks, and this guide | No | A push validates and deploys the site |

`gh-pages` is an orphan branch: it has no common ancestor with `main`. Do not merge one branch into the other. Transfer a specific asset or small documentation change by copying it deliberately and recording its source in the commit message.

Only `site/` is uploaded to GitHub Pages. Repository notes, scripts, and workflow files are not public site artifacts.

## 2. First-time administrator setup

These steps are required once, after the local `gh-pages` branch has been reviewed.

### 2.1 Publish the branch

From the website worktree:

```bash
git push --set-upstream origin gh-pages
```

The first workflow run can validate the site but its deployment step will fail until Pages is enabled. That is expected.

### 2.2 Enable GitHub Actions as the Pages source

An organization or repository administrator must:

1. Open **Settings → Pages** in `AtlasAnalyticsLab/PathoSynVLM`.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Open **Settings → Environments → github-pages** if the environment exists. Confirm that `gh-pages` is permitted by any deployment branch rule and that no unintended reviewer gate blocks automatic updates.
4. Confirm the organization Actions policy permits the four official `actions/*` dependencies pinned in `.github/workflows/deploy-pages.yml`.
5. Re-run the failed deployment job from its Actions run, or push a small follow-up commit to `gh-pages`.

Do not choose **Deploy from a branch** while the custom workflow is in use. That is the native MOOZY-style alternative, but combining both mechanisms makes ownership and troubleshooting ambiguous.

### 2.3 Finish repository metadata

After the workflow succeeds and the URL returns HTTP 200:

1. Set the repository **About → Website** field to `https://atlasanalyticslab.github.io/PathoSynVLM/`.
2. Confirm the prepared project-page badge/link on `main` resolves correctly; add a launch note if desired.
3. Consider protecting `gh-pages`: require a pull request and the **Validate static site** check, while allowing the Pages workflow to deploy.

The custom workflow lives only on `gh-pages`, so GitHub's normal manual **Run workflow** control is not promised. Automatic push deployment and the **Re-run jobs** control on an existing run are supported.

## 3. Local development

### Single-branch website clone

Use this when a maintainer only needs the website:

```bash
git clone --branch gh-pages --single-branch \
  git@github.com:AtlasAnalyticsLab/PathoSynVLM.git \
  PathoSynVLM-website
cd PathoSynVLM-website
```

### Worktree setup from an existing code clone

Use this to keep code and website working directories side by side without mixing their files:

```bash
cd PathoSynVLM
git fetch origin refs/heads/gh-pages:refs/remotes/origin/gh-pages
git worktree add -b gh-pages ../PathoSynVLM-website origin/gh-pages
cd ../PathoSynVLM-website
```

If a local `gh-pages` branch already exists, omit `-b gh-pages`:

```bash
git worktree add ../PathoSynVLM-website gh-pages
```

Check the separation at any time:

```bash
git worktree list
git branch --show-current
```

### Validate and preview

No installation is required beyond Python 3:

```bash
python3 scripts/validate_site.py site
python3 -m http.server 8000 --directory site
```

Open <http://localhost:8000/>. Test at narrow and wide viewport sizes, keyboard navigation, the mobile menu, figure links, and the BibTeX copy button. Clipboard access uses the browser's secure-context API on Pages and a fallback for local HTTP preview.

## 4. Normal change workflow

```bash
git switch gh-pages
git pull --ff-only
git switch -c website/short-description

# edit and preview
python3 scripts/validate_site.py site

git add site README.md DEVELOPMENT.md CONTRIBUTING.md scripts .github
git commit -m "website: describe the visible change"
git push -u origin website/short-description
```

Open a pull request with **base: `gh-pages`**. A pull request targeting `main` is the wrong destination for website changes. The workflow validates pull requests but deploys only after a push reaches `gh-pages`.

## 5. Sources of truth

Scientific content should not drift independently of the paper and code.

| Website content | Primary source |
|---|---|
| Title, authors, abstract, venue status | arXiv record `2605.30716` |
| Model components and training stages | `main/README.md`, `main/MODEL_CARD.md`, paper |
| Headline metrics | `main/configs/reported_results.json` |
| Architecture figure | `main/assets/paper_architecture.png` |
| Results figure | `main/assets/reported_results.svg` |
| Commands and reproduction links | `main/docs/paper_pipeline.md` |
| Citation | arXiv record and `main/README.md` |

When a paper version changes, compare the title, authors, abstract, venue, citation, claims, and metrics before editing the page. Use the paper's wording for scientific claims; do not infer stronger claims from a single metric.

The public page currently omits a model button because the documented Hugging Face endpoint is not accessible without authentication. Add the button only after an unauthenticated request reaches the intended public model page. Remove the release note at the same time.

Before the first public launch, the authors must also confirm the exact Stage 2 release configuration. Paper Table 3's headline values (`0.2495/0.1988/0.0525/0.3018`) reappear as the B1 prompt-repetition setting in Table 7, while Tables 11–12 report WSI-marker variants separately. The current `main/configs/stage2_main_paper.json` enables both prompt repetition and WSI markers. Do not relabel a marker result or change that config based on inference; reconcile the paper, config, and website with the experiment owner.

## 6. URL and asset rules

- Use relative paths such as `static/images/figure.png` in `index.html`. GitHub hosts this project below `/PathoSynVLM/`, not at the domain root.
- The root-relative links in `404.html` are an intentional exception: a 404 can be served from an arbitrary nested path and must return to `/PathoSynVLM/`.
- Use HTTPS for every external link.
- Prefer first-party, local assets. Remote styles, fonts, and scripts add privacy, availability, and supply-chain risk.
- Keep a single asset below 5 MiB; the validator enforces this limit. Optimize large raster figures before committing.
- Provide accurate `width`, `height`, and `alt` attributes for informative images.
- The Open Graph image URL must be absolute. If the production host changes, update the canonical URL, Open Graph URL/image, `robots.txt`, `sitemap.xml`, `404.html`, and constants in `scripts/validate_site.py` together.
- Do not edit the copied paper figures on this branch and leave `main` stale. Update the source asset on `main`, then deliberately copy the approved result to `gh-pages`.

## 7. Validation and deployment internals

`scripts/validate_site.py` uses only the Python standard library. It checks:

- required Pages, CSS, JavaScript, image, sitemap, and marker files;
- HTML language, title, viewport, description, and canonical metadata;
- duplicate IDs and same-page anchors;
- local files referenced by `href` and `src`;
- non-empty image alternative text;
- HTTPS for external links;
- sitemap and robots URL consistency; and
- the 5 MiB per-file asset budget.

`.github/workflows/deploy-pages.yml` pins immutable commits for:

- `actions/checkout` (version noted in the inline comment),
- `actions/configure-pages`,
- `actions/upload-pages-artifact`, and
- `actions/deploy-pages`.

When updating an action, verify its release in the official action repository, replace the full commit SHA, update the version comment, and review its release notes. Never replace a pinned official action with an unreviewed third-party action merely for convenience.

The deploy job has the minimum required `contents: read`, `pages: write`, and `id-token: write` permissions. Pull requests receive only read permission and never run the deploy job.

## 8. Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| `configure-pages` says the site is not found | Pages is disabled or its source is not GitHub Actions | Complete section 2.2, then re-run the deployment job |
| Deploy job receives `403` or OIDC/environment errors | Organization Actions policy or `github-pages` protection blocks the branch | Review Actions policy and environment deployment rules with an admin |
| Validation passes but CSS/images are missing online | A project-site path was changed to a domain-root path | Restore relative paths and run the validator |
| Production still shows an older revision | A run failed, environment approval is pending, or edge caching has not expired | Check the latest Actions run and deployment SHA; after success, allow several minutes |
| The workflow has no manual Run button | Its definition is isolated from the default branch by design | Push a commit or use **Re-run jobs** on an existing run |
| The model link prompts for authentication | The model repository is private or gated | Keep the public model button absent until release policy changes |

## 9. Safe rollback

Use a revert so the audit trail remains intact and the rollback itself triggers deployment:

```bash
git switch gh-pages
git pull --ff-only
git log --oneline -10
git revert <bad-commit-sha>
git push
```

Do not force-push `gh-pages` to roll back. If GitHub Pages itself is experiencing an incident, avoid unrelated commits and consult <https://www.githubstatus.com/>.

## 10. Launch and release checklist

Before announcing a new website version:

- [ ] `python3 scripts/validate_site.py site` passes.
- [ ] The page is readable at desktop and mobile widths.
- [ ] Keyboard navigation, the menu, copy button, and figure links work.
- [ ] Paper, code, documentation, and citation links reach the intended public resources.
- [ ] Scientific claims and metrics match their sources of truth.
- [ ] The experiment owner has confirmed the Stage 2 headline configuration and reconciled the paper/config labels.
- [ ] The latest `gh-pages` commit matches the successful deployment SHA.
- [ ] <https://atlasanalyticslab.github.io/PathoSynVLM/> returns HTTP 200.
- [ ] The browser console contains no site-owned errors.
- [ ] The `main` README and repository About URL point to the live project page.

## 11. License and third-party material

First-party website content and copied paper assets follow [CC BY-NC-SA 4.0](LICENSE). External datasets, pretrained encoders, language models, and linked resources have their own licenses and terms. Do not copy third-party media onto the site without confirming redistribution rights and attribution requirements.
