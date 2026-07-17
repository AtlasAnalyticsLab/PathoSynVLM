# Release Checklist

Before releasing the repository:

- Confirm the repository `LICENSE` is acceptable for release. The current license is CC BY-NC-SA 4.0.
- Upload the validated model root to [AtlasAnalyticsLab/PathoSynVLM](https://huggingface.co/AtlasAnalyticsLab/PathoSynVLM).
- Confirm the Hugging Face upload root contains `llm/model.safetensors` and `vlm_state.pt`, and that `find "$PATHOSYNVLM_HF_ROOT" -type l` prints nothing.
- Confirm `hf download AtlasAnalyticsLab/PathoSynVLM --dry-run` lists the complete model package and expected artifact sizes.
- Confirm every model-weight badge and download link resolves directly to `https://huggingface.co/AtlasAnalyticsLab/PathoSynVLM`.
- Confirm the final proceedings citation is synchronized across the README, model card, and project website.
- Re-run Stage 2 evaluation from the released weights on the filtered HISTAI metadata and compare with `configs/reported_results.json`.

Do not commit raw WSIs, H5 embeddings, training checkpoints, or dataset metadata with restricted redistribution terms.
