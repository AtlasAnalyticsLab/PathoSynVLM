# Data

This project uses datasets with their own access flow or terms.

## Stage 1

Stage 1 trains only the two-layer vision-language aligner on single-WSI text supervision.

Use the paper default:

- HistGen: https://github.com/dddavid4real/HistGen and https://huggingface.co/datasets/david4real/HistGen
- REG2025: https://reg2025.grand-challenge.org/

The internal research repo also contains PathText experiments. PathText is not part of the paper-default command.

Expected raw metadata:

```text
data/raw/histgen/annotation_update.json
data/raw/reg2025/train.json
```

Build filtered Stage 1 metadata after embeddings exist:

```bash
python scripts/prepare_stage1_metadata.py \
  --histgen-json data/raw/histgen/annotation_update.json \
  --reg-json data/raw/reg2025/train.json \
  --dataset-embeddings-root data/embeddings \
  --patch-level 5x_512
```

Output:

```text
data/stage1/merged_metadata_3datasets_raw.json
data/stage1/merged_metadata_3datasets_filtered_conch_v15.json
data/stage1/merged_metadata_3datasets_filtered_conch_v15_stats.json
```

## Stage 2

Stage 2 trains on case-report pairs from HISTAI:

- GitHub: https://github.com/HistAI/HISTAI
- Hugging Face collection: https://huggingface.co/collections/histai/histai-whole-slide-images-dataset

Expected raw metadata:

```text
data/raw/histai/standardized_metadata_fixed.json
```

Build filtered Stage 2 metadata after embeddings exist:

```bash
python scripts/prepare_histai_metadata.py \
  --metadata-standardized-json data/raw/histai/standardized_metadata_fixed.json \
  --dataset-embeddings-root data/embeddings \
  --patch-levels 5x_512
```

Output:

```text
data/histai/standardized_metadata_fixed_filtered_5x_512.json
data/histai/standardized_metadata_fixed_filtered_5x_512_dropped_cases.txt
data/histai/standardized_metadata_filter_stats.json
data/histai/standardized_metadata_filter_stats.md
```

## Splits

Both Stage 1 and Stage 2 use deterministic hash-based validation splitting with `split_seed=42`. This avoids dependence on input row order after metadata filtering.
