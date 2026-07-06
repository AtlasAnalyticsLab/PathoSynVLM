#!/bin/bash
#SBATCH --job-name=pathosynvlm-s2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

: "${PATHOSYNVLM_DATA_ROOT:=$PWD/data}"
: "${PATHOSYNVLM_EMBEDDINGS_ROOT:=$PATHOSYNVLM_DATA_ROOT/embeddings}"
: "${PATHOSYNVLM_HISTAI_METADATA_DIR:=$PATHOSYNVLM_DATA_ROOT/histai}"
: "${PATHOSYNVLM_RUNS_ROOT:=$PWD/runs}"
export PATHOSYNVLM_DATA_ROOT PATHOSYNVLM_EMBEDDINGS_ROOT PATHOSYNVLM_HISTAI_METADATA_DIR PATHOSYNVLM_RUNS_ROOT

python scripts/train_stage2_histai.py \
  --metadata_standardized_json "$PATHOSYNVLM_HISTAI_METADATA_DIR/standardized_metadata_fixed_filtered_5x_512.json" \
  --dataset_embeddings_root "$PATHOSYNVLM_EMBEDDINGS_ROOT" \
  --aligner_init "$PATHOSYNVLM_RUNS_ROOT/stage1_alignment/best_aligner_weights.pt" \
  --output_dir "$PATHOSYNVLM_RUNS_ROOT/stage2_main" \
  --prompt_style double \
  --max_text_length 384 \
  --max_vision_tokens 4096 \
  --use_wsi_markers \
  --unfreeze_llm_base \
  --gradient_checkpoint
