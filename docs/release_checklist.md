# Release Checklist

Before releasing the repository:

- Confirm the repository `LICENSE` is acceptable for release. The current license is CC BY-NC-SA 4.0.
- Upload the validated Hugging Face model repository root to the selected model repository.
- Confirm the Hugging Face upload root contains `llm/model.safetensors` and `vlm_state.pt`, and that `find "$PATHOSYNVLM_HF_ROOT" -type l` prints nothing.
- Replace the Hugging Face command in the README with the final model-weight URL or repository id.
- Add the final paper citation if a conference or journal version supersedes the arXiv preprint.
- Re-run Stage 2 evaluation from the released weights on the filtered HISTAI metadata and compare with `configs/reported_results.json`.

Do not commit raw WSIs, H5 embeddings, training checkpoints, or dataset metadata with restricted redistribution terms.
