#!/bin/bash
#SBATCH --job-name=pathosynvlm-s1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

python scripts/train_stage1_alignment.py \
  --metadata_json data/stage1/merged_metadata_3datasets_filtered_conch_v15.json \
  --dataset_embeddings_root data/embeddings \
  --datasets histgen,reg_dataset \
  --output_dir runs/stage1_alignment
