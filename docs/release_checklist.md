# Release Checklist

Before releasing the repository:

- Confirm the repository `LICENSE` is acceptable for release. This repo currently follows MOOZY with CC BY-NC-SA 4.0.
- Upload `weights/pathosynvlm-stage2-main/` to the selected artifact host after running `scripts/export_release_weights.py`.
- Replace the placeholder Hugging Face command in the README with the final model-weight URL or repository id.
- Replace `<repo-url>` in the README after the GitHub repository is created.
- Add the final paper citation if a conference or journal version supersedes the arXiv preprint.
- Re-run Stage 2 evaluation from the uploaded weights on the filtered HISTAI metadata and compare with `configs/reported_results.json`.

Do not commit raw WSIs, H5 embeddings, training checkpoints, or dataset metadata with restricted redistribution terms.
