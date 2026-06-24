#!/bin/bash
#SBATCH --job-name=pathosynvlm-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

python scripts/evaluate_checkpoint.py \
  --finetune_run_dir runs/stage2_main \
  --dataset_scope histai \
  --histai_metadata_standardized_json data/histai/standardized_metadata_fixed_filtered_5x_512.json \
  --dataset_embeddings_root data/embeddings \
  --output_json runs/stage2_main/eval_histai.json
