# Weights

Large model artifacts are not committed to Git.

For normal inference, users should download the released model artifact into this directory. They should not need to run `scripts/export_release_weights.py` unless they are packaging their own trained checkpoint.

Expected inference package layout:

```text
$PATHOSYNVLM_WEIGHTS_ROOT/pathosynvlm-stage2-main/
  config.json
  vlm_state.pt
  tokenizer/
  llm/              # merged full model for the paper release
  best_checkpoint_summary.json
```

Use `scripts/export_release_weights.py` only when creating this layout from a completed local training run. For normal use, download the released Hugging Face package directly.
