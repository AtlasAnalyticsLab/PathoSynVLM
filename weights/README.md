# Weights

Large model artifacts are not committed to Git.

For normal inference, users should download the author-uploaded release artifact into this directory. They should not need to run `scripts/export_release_weights.py` unless they are packaging their own trained checkpoint.

Expected inference package layout:

```text
weights/pathosynvlm-stage2-main/
  config.json
  vlm_state.pt
  tokenizer/
  llm/              # merged full model, preferred for release
  best_checkpoint_summary.json
```

Use `scripts/export_release_weights.py` to create this layout from a completed local training run before uploading the official release package.
