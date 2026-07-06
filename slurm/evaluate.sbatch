#!/bin/bash
#SBATCH --job-name=pathosynvlm-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

: "${PATHOSYNVLM_DATA_ROOT:=$PWD/data}"
: "${PATHOSYNVLM_EMBEDDINGS_ROOT:=$PATHOSYNVLM_DATA_ROOT/embeddings}"
: "${PATHOSYNVLM_HISTAI_METADATA_DIR:=$PATHOSYNVLM_DATA_ROOT/histai}"
: "${PATHOSYNVLM_RUNS_ROOT:=$PWD/runs}"
export PATHOSYNVLM_DATA_ROOT PATHOSYNVLM_EMBEDDINGS_ROOT PATHOSYNVLM_HISTAI_METADATA_DIR PATHOSYNVLM_RUNS_ROOT

python scripts/evaluate_checkpoint.py \
  --finetune_run_dir "$PATHOSYNVLM_RUNS_ROOT/stage2_main" \
  --dataset_scope histai \
  --histai_metadata_standardized_json "$PATHOSYNVLM_HISTAI_METADATA_DIR/standardized_metadata_fixed_filtered_5x_512.json" \
  --dataset_embeddings_root "$PATHOSYNVLM_EMBEDDINGS_ROOT" \
  --output_json "$PATHOSYNVLM_RUNS_ROOT/stage2_main/eval_histai.json"
