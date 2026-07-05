# Hugging Face Release

This document describes the Hugging Face model repository preparation flow for PathoSynVLM.

The target Hub repository should look similar in spirit to AtlasAnalyticsLab/MOOZY and AtlasAnalyticsLab/AtlasPatch: concise model-card metadata, quickstart commands, input format, architecture/training summary, metrics, citation, and clear non-commercial license language.

## Current Upload Root

Treat this folder as the Hugging Face model repository root:

```text
/home/chengj60/scratch/PathoSynVLM_HF_release_review/hf_repo_preview
```

Upload this folder only. Do not upload its parent review folder.

The current package contains the actual exported weights:

```text
llm/model.safetensors  # merged/full Qwen2.5-3B weights
vlm_state.pt           # aligner and WSI marker tensors
```

It should not contain symlinks:

```bash
find /home/chengj60/scratch/PathoSynVLM_HF_release_review/hf_repo_preview -type l
```

The command above should print nothing.

## Expected Hub Root

Before upload, the Hub repo folder should contain:

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

## Refresh Metadata Without Re-exporting Weights

Run this when the model card, labels, manifest, assets, GitHub URL, or final Hub repo id changes:

```bash
python scripts/prepare_hf_release.py \
  --output-dir /home/chengj60/scratch/PathoSynVLM_HF_release_review \
  --hf-repo-dir /home/chengj60/scratch/PathoSynVLM_HF_release_review/hf_repo_preview \
  --repo-id <ORG_OR_USER>/pathosynvlm-stage2-main \
  --github-url <GITHUB_REPO_URL>
```

This command keeps the actual weights in `hf_repo_preview/` and refreshes the Hugging Face README, license, model index, labels, examples, upload instructions, and release manifest.

The script writes internal checkpoint provenance to:

```text
/home/chengj60/scratch/PathoSynVLM_HF_release_review/source_artifacts_reference.json
```

That file is outside the Hub repo root and should not be uploaded.

## Re-export Weights

Only re-export if the source checkpoint or export logic changes. The paper Stage 2 run used `unfreeze_llm_base=true`, so the Hugging Face model must include the exported merged/full LLM directory rather than only a LoRA adapter.

Run the export on a compute node with enough memory:

```bash
python scripts/export_release_weights.py \
  --run_dir /home/chengj60/scratch/VLM_MVP/runs/histai_finetune-20260221/histai_ft_5x512_from_5x512_all_baseline_prompt_double \
  --output_dir /home/chengj60/scratch/PathoSynVLM_HF_release_review/hf_repo_preview \
  --checkpoint_step 30400 \
  --overwrite
```

Then refresh metadata:

```bash
python scripts/prepare_hf_release.py \
  --output-dir /home/chengj60/scratch/PathoSynVLM_HF_release_review \
  --hf-repo-dir /home/chengj60/scratch/PathoSynVLM_HF_release_review/hf_repo_preview \
  --repo-id <ORG_OR_USER>/pathosynvlm-stage2-main \
  --github-url <GITHUB_REPO_URL>
```

## Validate

```bash
HF=/home/chengj60/scratch/PathoSynVLM_HF_release_review/hf_repo_preview
test -f "$HF/llm/model.safetensors"
test -f "$HF/vlm_state.pt"
test "$(find "$HF" -type l | wc -l)" = "0"
python -m json.tool /home/chengj60/scratch/PathoSynVLM_HF_release_review/release_manifest.json >/dev/null
python -m json.tool "$HF/model_index.json" >/dev/null
python -m json.tool "$HF/labels.json" >/dev/null
```

## Upload

After the final Hugging Face repository id is chosen:

```bash
hf repos create <ORG_OR_USER>/pathosynvlm-stage2-main --type model --private --exist-ok
hf upload-large-folder <ORG_OR_USER>/pathosynvlm-stage2-main \
  /home/chengj60/scratch/PathoSynVLM_HF_release_review/hf_repo_preview \
  --type model
```

Use `hf upload-large-folder` because the merged LLM directory is large and resumable upload is safer.
