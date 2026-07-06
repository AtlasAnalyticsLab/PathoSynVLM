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

: "${PATHOSYNVLM_DATA_ROOT:=$PWD/data}"
: "${PATHOSYNVLM_EMBEDDINGS_ROOT:=$PATHOSYNVLM_DATA_ROOT/embeddings}"
: "${PATHOSYNVLM_STAGE1_METADATA_DIR:=$PATHOSYNVLM_DATA_ROOT/stage1}"
: "${PATHOSYNVLM_RUNS_ROOT:=$PWD/runs}"
export PATHOSYNVLM_DATA_ROOT PATHOSYNVLM_EMBEDDINGS_ROOT PATHOSYNVLM_STAGE1_METADATA_DIR PATHOSYNVLM_RUNS_ROOT

python scripts/train_stage1_alignment.py \
  --metadata_json "$PATHOSYNVLM_STAGE1_METADATA_DIR/merged_metadata_3datasets_filtered_conch_v15.json" \
  --dataset_embeddings_root "$PATHOSYNVLM_EMBEDDINGS_ROOT" \
  --datasets histgen,reg_dataset \
  --output_dir "$PATHOSYNVLM_RUNS_ROOT/stage1_alignment"
