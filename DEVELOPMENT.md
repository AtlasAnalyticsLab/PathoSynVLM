# Website development and operations

This document is the operational source of truth for the PathoSynVLM project website. It covers branch isolation, local editing, GitHub Pages deployment, rollback, and launch checks.

## 1. Architecture and branch policy

The repository has two intentionally independent histories:

| Branch | Purpose | Checked out in the default working tree? | Deployment effect |
|---|---|---:|---|
| `main` | Python package, paper configs, research docs, and code assets | Yes | None |
| `gh-pages` | Static project page, validation, and website maintenance docs | No | A push runs the single website validation-and-deployment workflow |

`gh-pages` is an orphan branch: it has no common ancestor with `main`. Do not merge one branch into the other. Transfer an approved paper asset or small documentation change deliberately and record its source in the commit message.

The actual page lives at `index.html` in the `gh-pages` root. The deployment job stages that page and its public assets into a temporary `_site` artifact; repository notes, scripts, and workflow files are not published.

## 2. GitHub Pages configuration

The required repository setting is:

1. Open **Settings → Pages** in `AtlasAnalyticsLab/PathoSynVLM`.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Open **Settings → Environments → github-pages** if deployment restrictions are configured. Confirm `gh-pages` is allowed and no unintended reviewer gate blocks automatic publication.

Do not select **Deploy from a branch** while the committed workflow is in use. That setting creates a second GitHub-managed Pages run for every push and gives the repository two competing publishers.

After the site returns HTTP 200:

1. Set the repository **About → Website** field to `https://atlasanalyticslab.github.io/PathoSynVLM/`.
2. Confirm the project-page badge/link on `main` resolves correctly.
3. Consider protecting `gh-pages`: require a pull request and the **Check PathoSynVLM static site** check.

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
python3 scripts/validate_site.py .
python3 -m http.server 8000 --directory .
```

Open <http://localhost:8000/>. Test wide and narrow viewport sizes, keyboard navigation, result tabs, figure lightboxes, and the BibTeX copy button.

## 4. Normal change workflow

```bash
git switch gh-pages
git pull --ff-only
git switch -c website/short-description

# edit and preview
python3 scripts/validate_site.py .

git add index.html 404.html static robots.txt sitemap.xml .nojekyll \
  README.md DEVELOPMENT.md CONTRIBUTING.md scripts .github
git commit -m "website: describe the visible change"
git push -u origin website/short-description
```

Open a pull request with **base: `gh-pages`**. A pull request targeting `main` is the wrong destination for website changes. Pull requests validate without publishing; merging to `gh-pages` runs validation and then deployment in the same workflow.

## 5. Sources of truth

Scientific content should not drift independently of the paper and code.

| Website content | Primary source |
|---|---|
| Title, authors, abstract, venue status | arXiv record `2605.30716` |
| Model components and training stages | Paper, `main/README.md`, `main/MODEL_CARD.md` |
| Headline metrics | Paper Table 3, `main/configs/reported_results.json` |
| Efficiency values | Paper Table 4 |
| Evidence-audit preference counts | Current paper, AI-preference comparison table |
| Pathologist-audit scores and limitations | Current paper, pathologist-audit table; DeLTA 2026 presentation slide 19/notes for the no-significance-test qualifier |
| Architecture figure | `main/assets/paper_architecture.png` |
| Training-data figure | Paper Figure 4 (`a_dataset_distribution.png`) |
| Qualitative examples | Paper Figure 8 (`baseline_stage2_val_examples_histai_wsi_gt_pred.png`) |
| Commands and reproduction links | `main/docs/paper_pipeline.md` |
| Citation | arXiv record and `main/README.md` |

When a paper version changes, compare the title, authors, abstract, venue, citation, claims, figures, and metrics before editing the page. Use the paper's wording for scientific claims; do not infer stronger claims from a single metric or qualitative example.

The hero intentionally presents only the Paper and Code links. Keep duplicate paper-format links, venue-acceptance messaging, and a separate reproduction button off the public hero; the Quick Start section already links to the maintained paper pipeline.

The public page omits a model button because the documented Hugging Face endpoint is not accessible without authentication. Add the button only after an unauthenticated request reaches the intended public model page.

The experiment owner must confirm the exact Stage 2 release configuration. Paper Table 3's headline values (`0.2495/0.1988/0.0525/0.3018`) reappear as the B1 prompt-repetition setting in Table 7, while Tables 11–12 report WSI-marker variants separately. The current `main/configs/stage2_main_paper.json` enables both prompt repetition and WSI markers. Do not relabel a marker result or change that config based on inference; reconcile the paper, config, and website with the experiment owner.

Paper Figure 4 labels the Stage 2 total as `43,619` cases and the mixed group as `20,925`, while Table 2 reports `43,618` and `20,924`. The site preserves the published figure but deliberately avoids restating the conflicting Stage 2 total in prose. The paper owner should reconcile those source values before a future figure revision.

The Results evidence audit is native HTML and CSS, not a pasted slide image. When updating it, keep the limitations ahead of the charts and update the visible counts, bar percentages, sample sizes, and accessible labels together. Preserve the comparator-specific protocols, HistoGPT skin-only qualifier, one-reader/no-significance-test limitation, and the statement that neither audited system is clinically adequate. The ≈93% composition note describes the full Stage 2 corpus, not the pathologist-audit subset. Ignore unrelated leftover slide objects that are not part of the stress-test content.

## 6. URL and asset rules

- Use relative paths such as `static/images/figure.png` in `index.html`. GitHub hosts this project below `/PathoSynVLM/`, not at the domain root.
- Keep the header's lab link pointed at the canonical Atlas Analytics Lab site: `https://atlasanalyticslab.github.io/`.
- Header section links must target existing IDs in `index.html`. When changing the header, test the mobile menu, outside-click close, Escape close, and sticky-anchor offset at 320 px and desktop widths.
- The root-relative links in `404.html` are an intentional exception because a 404 can be served from an arbitrary nested URL.
- Use HTTPS for every external link.
- Prefer first-party, local assets. Remote styles, fonts, and scripts add availability and supply-chain risk.
- Keep a single asset below 5 MiB; the validator enforces this limit. Load large paper figures lazily.
- Provide accurate `width`, `height`, and `alt` attributes for informative images.
- The Open Graph image URL must be absolute. If the production host changes, update the canonical URL, Open Graph URL/image, `robots.txt`, `sitemap.xml`, `404.html`, and validator constants together.
- Paper-derived figures follow the paper's CC BY-NC-SA 4.0 license. Record the paper figure number and source filename in this guide when adding one.
- If a copied asset also lives on `main`, update the source asset there first, then copy the approved result to `gh-pages`.

## 7. Validation and deployment internals

`scripts/validate_site.py` uses only the Python standard library. It checks:

- required HTML, CSS, JavaScript, image, sitemap, and marker files;
- HTML language, title, viewport, description, and canonical metadata;
- duplicate IDs and same-page anchors;
- local files referenced by `href` and `src`;
- non-empty image alternative text;
- HTTPS for external links;
- sitemap and robots URL consistency; and
- the 5 MiB per-file asset budget.

`.github/workflows/validate-site.yml` defines the single `PathoSynVLM website` workflow. It runs on pushes and pull requests targeting `gh-pages`:

- The `validate` job grants only `contents: read` and runs the dependency-free validator.
- The `deploy` job runs only for pushes, depends on successful validation, and receives the minimum `contents: read`, `pages: write`, and `id-token: write` permissions required by GitHub Pages.
- The deploy job stages `index.html`, `404.html`, `.nojekyll`, `robots.txt`, `sitemap.xml`, and `static/` into `_site`. Only that temporary directory is uploaded as the Pages artifact.

The workflow pins immutable commits for `actions/checkout`, `actions/configure-pages`, `actions/upload-pages-artifact`, and `actions/deploy-pages`. When updating an action, verify its release in the official repository, replace the full commit SHA, update the version comment, and review its release notes.

GitHub adds a new run record whenever the workflow is triggered. This is normal audit history and does not create another workflow definition. If a separate `pages build and deployment` run also appears for the same SHA, the Pages source is still set to branch publishing; change it to **GitHub Actions**.

## 8. Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| `configure-pages` says the site is not configured for workflows | Pages still uses branch publishing or is disabled | Set **Settings → Pages → Source** to **GitHub Actions**, then rerun or push a follow-up commit |
| A separate `pages build and deployment` run appears | Pages still uses **Deploy from a branch** | Switch the Pages source to **GitHub Actions** so `PathoSynVLM website` is the only publisher |
| Multiple rows named `PathoSynVLM website` appear | GitHub is retaining completed runs of the same workflow | This is normal audit history; delete old completed runs only when their records are no longer needed |
| Validation passes but CSS/images are missing | A project-site path was changed to a domain-root path | Restore relative paths and run the validator |
| Production shows an older revision | The deploy job failed, is awaiting environment approval, or edge caching has not expired | Check the latest `PathoSynVLM website` run and allow several minutes after success |
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

- [ ] `python3 scripts/validate_site.py .` passes.
- [ ] The page is readable at desktop and mobile widths without horizontal page overflow.
- [ ] Keyboard navigation, tabs, lightboxes, copy button, and figure links work.
- [ ] Paper, code, documentation, and citation links reach the intended public resources.
- [ ] Scientific claims, figures, and metrics match their sources of truth.
- [ ] The experiment owner has confirmed the Stage 2 headline configuration and reconciled the paper/config labels.
- [ ] The latest `gh-pages` commit matches the successful `PathoSynVLM website` deployment SHA.
- [ ] The production root, figure URLs, custom 404, and sitemap return successfully.
- [ ] The browser console contains no site-owned errors.
- [ ] The `main` README and repository About URL point to the live project page.

## 11. License and third-party material

First-party website content and copied paper assets follow [CC BY-NC-SA 4.0](LICENSE). External datasets, pretrained encoders, language models, and linked resources have their own licenses and terms. Do not copy third-party media onto the site without confirming redistribution rights and attribution requirements.
