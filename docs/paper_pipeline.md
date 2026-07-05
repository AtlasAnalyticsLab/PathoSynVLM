# Paper Pipeline

This document maps paper claims to commands and configs.

## Environment

```bash
conda create -n pathosynvlm python=3.11 -y
conda activate pathosynvlm
export PYTHONNOUSERSITE=1
pip install -e .
```

On clusters, run the commands inside an allocated compute session. Keeping `PYTHONNOUSERSITE=1` avoids accidentally importing packages from a pre-existing user site.

## Stage 1 Baseline

Config: [configs/stage1_alignment_paper.json](../configs/stage1_alignment_paper.json)

```bash
python scripts/train_stage1_alignment.py \
  --metadata_json data/stage1/merged_metadata_3datasets_filtered_conch_v15.json \
  --dataset_embeddings_root data/embeddings \
  --datasets histgen,reg_dataset \
  --patch_level 5x_512 \
  --output_dir runs/stage1_alignment
```

Reported Stage 1 baseline metrics:

| ROUGE-L | METEOR | BLEU-4 | BERTScore F1 |
|---:|---:|---:|---:|
| 0.4743 | 0.4810 | 0.1247 | 0.4253 |

## Stage 2 Main Result

Config: [configs/stage2_main_paper.json](../configs/stage2_main_paper.json)

```bash
python scripts/train_stage2_histai.py \
  --metadata_standardized_json data/histai/standardized_metadata_fixed_filtered_5x_512.json \
  --dataset_embeddings_root data/embeddings \
  --aligner_init runs/stage1_alignment/best_aligner_weights.pt \
  --output_dir runs/stage2_main \
  --prompt_style double \
  --max_text_length 384 \
  --max_vision_tokens 4096 \
  --use_wsi_markers \
  --unfreeze_llm_base \
  --gradient_checkpoint
```

Evaluate:

```bash
python scripts/evaluate_checkpoint.py \
  --finetune_run_dir runs/stage2_main \
  --dataset_scope histai \
  --histai_metadata_standardized_json data/histai/standardized_metadata_fixed_filtered_5x_512.json \
  --dataset_embeddings_root data/embeddings \
  --output_json runs/stage2_main/eval_histai.json
```

Reported Stage 2 main metrics:

| ROUGE-L | METEOR | BLEU-4 | BERTScore F1 | Diagnosis Exact | Diagnosis Relaxed | Certainty |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2495 | 0.1988 | 0.0525 | 0.3018 | 0.1667 | 0.3333 | 0.9000 |

The training code logs sacreBLEU as a percentage, so `5.2512` in JSON corresponds to `0.0525` in the paper.

## Ablations

Paper ablation values are stored in [configs/reported_results.json](../configs/reported_results.json). The corresponding command settings are summarized in [configs/stage2_wsi_marker_ablation.json](../configs/stage2_wsi_marker_ablation.json).
