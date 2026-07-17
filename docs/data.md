# Data

This project uses datasets with their own access flow or terms.

## Local Paths

By default, scripts read and write under `data/` in the repository. To use datasets or embeddings that already live elsewhere, export path roots before running the commands:

```bash
source configs/paths.example.env

export PATHOSYNVLM_RAW_DATA_ROOT="/data/pathosynvlm/raw"
export PATHOSYNVLM_EMBEDDINGS_ROOT="/features/pathosynvlm/conch_embeddings"
export PATHOSYNVLM_STAGE1_METADATA_DIR="/outputs/pathosynvlm/stage1_metadata"
export PATHOSYNVLM_HISTAI_METADATA_DIR="/outputs/pathosynvlm/histai_metadata"
```

The command-line path flags override these defaults when needed.

## Stage 1

Stage 1 trains only the two-layer vision-language aligner on single-WSI text supervision.

Use the paper default:

- HistGen: https://github.com/dddavid4real/HistGen and https://huggingface.co/datasets/david4real/HistGen
- REG2025: https://reg2025.grand-challenge.org/

Expected raw metadata:

```text
$PATHOSYNVLM_RAW_DATA_ROOT/histgen/annotation_update.json
$PATHOSYNVLM_RAW_DATA_ROOT/reg2025/train.json
```

Build filtered Stage 1 metadata after embeddings exist:

```bash
python scripts/prepare_stage1_metadata.py \
  --histgen-json "$PATHOSYNVLM_RAW_DATA_ROOT/histgen/annotation_update.json" \
  --reg-json "$PATHOSYNVLM_RAW_DATA_ROOT/reg2025/train.json" \
  --dataset-embeddings-root "$PATHOSYNVLM_EMBEDDINGS_ROOT" \
  --patch-level 5x_512
```

Output:

```text
$PATHOSYNVLM_STAGE1_METADATA_DIR/merged_metadata_3datasets_raw.json
$PATHOSYNVLM_STAGE1_METADATA_DIR/merged_metadata_3datasets_filtered_conch_v15.json
$PATHOSYNVLM_STAGE1_METADATA_DIR/merged_metadata_3datasets_filtered_conch_v15_stats.json
```

## Stage 2

Stage 2 trains on case-report pairs from HISTAI:

- Hugging Face metadata and WSI access: https://huggingface.co/datasets/histai/HISTAI-metadata
- Hugging Face WSI collection: https://huggingface.co/collections/histai/histai-whole-slide-images-dataset
- Source documentation: https://github.com/HistAI/HISTAI

Expected raw metadata:

```text
$PATHOSYNVLM_RAW_DATA_ROOT/histai/standardized_metadata_fixed.json
```

Build filtered Stage 2 metadata after embeddings exist:

```bash
python scripts/prepare_histai_metadata.py \
  --metadata-standardized-json "$PATHOSYNVLM_RAW_DATA_ROOT/histai/standardized_metadata_fixed.json" \
  --dataset-embeddings-root "$PATHOSYNVLM_EMBEDDINGS_ROOT" \
  --patch-levels 5x_512
```

Output:

```text
$PATHOSYNVLM_HISTAI_METADATA_DIR/standardized_metadata_fixed_filtered_5x_512.json
$PATHOSYNVLM_HISTAI_METADATA_DIR/standardized_metadata_fixed_filtered_5x_512_dropped_cases.txt
$PATHOSYNVLM_HISTAI_METADATA_DIR/standardized_metadata_filter_stats.json
$PATHOSYNVLM_HISTAI_METADATA_DIR/standardized_metadata_filter_stats.md
```

## Splits

Both Stage 1 and Stage 2 use deterministic hash-based validation splitting with `split_seed=42`. This avoids dependence on input row order after metadata filtering.
