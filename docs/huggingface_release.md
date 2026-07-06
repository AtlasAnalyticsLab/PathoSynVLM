# Hugging Face Release

This document describes how to refresh and validate the Hugging Face model repository files for PathoSynVLM.

## Release Root

The Hugging Face model repository root should contain the exported model weights and model-card files:

```text
README.md
LICENSE
.gitattributes
model_index.json
labels.json
config.json
vlm_state.pt
best_checkpoint_summary.json
tokenizer/
llm/
assets/
examples/
```

The paper Stage 2 run used `unfreeze_llm_base=true`, so `llm/` must contain the merged/full language-model weights. A LoRA adapter alone is not sufficient for exact release inference.

Set the paths for your local release workspace:

```bash
export PATHOSYNVLM_HF_REPO=AtlasAnalyticsLab/pathosynvlm-stage2-main
export PATHOSYNVLM_GITHUB_URL=https://github.com/AtlasAnalyticsLab/PathoSynVLM
export PATHOSYNVLM_RUNS_ROOT="${PATHOSYNVLM_RUNS_ROOT:-$PWD/runs}"
export PATHOSYNVLM_STAGE2_RUN="$PATHOSYNVLM_RUNS_ROOT/stage2_main"
export PATHOSYNVLM_HF_ROOT=release/huggingface/pathosynvlm-stage2-main
```

## Refresh Metadata Without Re-exporting Weights

Run this when the model card, labels, assets, GitHub URL, or Hub repo id changes:

```bash
python scripts/prepare_hf_release.py \
  --output-dir release/huggingface \
  --hf-repo-dir "$PATHOSYNVLM_HF_ROOT" \
  --repo-id "$PATHOSYNVLM_HF_REPO" \
  --github-url "$PATHOSYNVLM_GITHUB_URL" \
  --source-run-dir "$PATHOSYNVLM_STAGE2_RUN"
```

This command keeps the actual weights in `PATHOSYNVLM_HF_ROOT` and refreshes the Hugging Face README, license, model index, labels, examples, upload instructions, and release manifest.

## Re-export Weights

Only re-export if the source checkpoint or export logic changes. Run the export on a compute node with enough memory:

```bash
python scripts/export_release_weights.py \
  --run_dir "$PATHOSYNVLM_STAGE2_RUN" \
  --output_dir "$PATHOSYNVLM_HF_ROOT" \
  --checkpoint_step 30400 \
  --overwrite
```

Then refresh metadata with `scripts/prepare_hf_release.py` as shown above.

## Validate

```bash
test -f "$PATHOSYNVLM_HF_ROOT/llm/model.safetensors"
test -f "$PATHOSYNVLM_HF_ROOT/vlm_state.pt"
test "$(find "$PATHOSYNVLM_HF_ROOT" -type l | wc -l)" = "0"
python -m json.tool release/huggingface/release_manifest.json >/dev/null
python -m json.tool "$PATHOSYNVLM_HF_ROOT/model_index.json" >/dev/null
python -m json.tool "$PATHOSYNVLM_HF_ROOT/labels.json" >/dev/null
```

The symlink check should print `0`; upload only real files.

## Upload

After the final Hugging Face repository id is chosen:

```bash
hf repos create "$PATHOSYNVLM_HF_REPO" --type model --exist-ok
hf upload-large-folder "$PATHOSYNVLM_HF_REPO" "$PATHOSYNVLM_HF_ROOT" --type model
```

Use `hf upload-large-folder` because the merged LLM directory is large and resumable upload is safer.
