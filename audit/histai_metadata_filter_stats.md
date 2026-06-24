# Filtered Standardized Metadata Stats

## Inputs

- Metadata: `data/raw/histai/standardized_metadata_fixed.json`
- Embeddings root: `data/embeddings`
- Feature key: `conch_v15`
- Patch levels: `1x_512, 5x_512`
- Validity mode: `case_level_embedding_presence` (no per-file h5 probe)

## Standardized Dedup Stats

- Input rows: `45170`
- Rows missing `case_mapping`: `0`
- Rows invalid `case_mapping`: `0`
- Duplicate case rows dropped: `1546`
- Unique cases after dedup: `43624`

## Per Patch

| Patch | Embedding Cases (All) | Embedding Cases (Valid) | Filtered Rows | Filtered Unique Cases | Std Unique Cases Without Embedding |
|---|---:|---:|---:|---:|---:|
| 1x_512 | 47273 | 47273 | 43619 | 43619 | 5 |
| 5x_512 | 47273 | 47273 | 43619 | 43619 | 5 |

## Dataset Used for Training (Current Split)

Split config used in finetune runs:

- `val_size=0.2`
- `split_seed=42`
- metadata input for training: `standardized_metadata_fixed_filtered_5x_512.json` (same row count as `1x_512`)

Pair counts:

- Total usable pairs: `43619`
- Train pairs: `34895`
- Val pairs: `8724`

Per-group pair counts:

| Group | Total | Train | Val |
|---|---:|---:|---:|
| HISTAI-breast | 1486 | 1177 | 309 |
| HISTAI-colorectal-b1 | 812 | 646 | 166 |
| HISTAI-colorectal-b2 | 57 | 41 | 16 |
| HISTAI-gastrointestinal | 107 | 87 | 20 |
| HISTAI-hematologic | 188 | 144 | 44 |
| HISTAI-mixed | 20925 | 16780 | 4145 |
| HISTAI-skin-b1 | 1441 | 1135 | 306 |
| HISTAI-skin-b2 | 18035 | 14434 | 3601 |
| HISTAI-thorax | 568 | 451 | 117 |

Patch-token usage under `max_vision_tokens=4096` (5x_512 cap analysis):

| Split | Expected All Patches (No Cap) | Actual Patches Used | Tokens Clipped |
|---|---:|---:|---:|
| train (readable=34894) | 12957545 | 12276406 | 681139 |
| val (readable=8724) | 3373856 | 3129294 | 244562 |
| total | 16331401 | 15405700 | 925701 |
