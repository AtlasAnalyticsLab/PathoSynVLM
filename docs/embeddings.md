# Embeddings

PathoSynVLM trains and runs on precomputed WSI patch embeddings, not raw WSIs.

The paper uses:

- Patch encoder: CONCHv1.5
- Feature key: `conch_v15`
- Patch level: `5x_512` for the main result
- Optional ablation: `1x_512`

Each `.h5` file must contain a two-dimensional feature matrix:

```text
/features/conch_v15  # shape: (num_patches, 768)
```

If `/features` contains exactly one dataset, the loaders can fall back to that dataset, but the recommended path is to store features under `/features/conch_v15`.

## Expected Layout

```text
data/embeddings/
  HistGen-train/conch_v15/5x_512/patches/*.h5
  HistGen-val/conch_v15/5x_512/patches/*.h5
  HistGen-test/conch_v15/5x_512/patches/*.h5
  REG_dataset/REG_train/conch_v15/5x_512/patches/*.h5
  REG_dataset/REG_test/REG_test1/conch_v15/5x_512/patches/*.h5
  REG_dataset/REG_test/REG_test2_revised/conch_v15/5x_512/patches/*.h5
  HISTAI-skin-b2/conch_v15/5x_512/patches/*.h5
  HISTAI-*/conch_v15/5x_512/patches/*.h5
```

## Feature Extraction

The paper used AtlasPatch for tissue detection, patch extraction, and CONCHv1.5 embedding generation. Any extractor is acceptable if it writes the H5 layout above with the same CONCHv1.5 feature dimension.

Recommended workflow:

1. Download the raw WSIs and metadata following [data.md](data.md).
2. Extract tissue patches at `5x_512`.
3. Encode each patch with CONCHv1.5.
4. Write one H5 feature file per WSI into the dataset-specific folder above.
5. Run the metadata-preparation scripts, which keep only cases with available embeddings.

Sanity check a HistAI batch without loading the LLM:

```bash
python pathosynvlm/histai_dataset.py \
  --metadata-standardized-json data/histai/standardized_metadata_fixed_filtered_5x_512.json \
  --dataset-embeddings-root data/embeddings \
  --feature-key conch_v15 \
  --patch-level 5x_512 \
  --no-tokenizer \
  --limit 2
```
