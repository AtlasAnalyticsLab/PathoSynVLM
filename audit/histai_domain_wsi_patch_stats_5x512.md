# HistAI Domain WSI/Patch Stats

## Config
- metadata_standardized_json: `data/histai/standardized_metadata_fixed_filtered_5x_512.json`
- dataset_embeddings_root: `data/embeddings`
- feature_key: `conch_v15`
- patch_level: `5x_512`
- val_size: `0.2`
- split_seed: `42`
- patch_cap_for_reference: `4096`

## Domain Totals (Readable/Used)
| Group | Cases (Readable) | WSIs (Readable) | Patches (No Cap) | WSIs/Case | Patches/WSI (No Cap) | Patches/Case (No Cap) | Patches (Cap) |
|---|---:|---:|---:|---:|---:|---:|---:|
| HISTAI-breast | 1,486 | 1,679 | 345,382 | 1.130 | 205.71 | 232.42 | 344,674 |
| HISTAI-colorectal-b1 | 812 | 4,436 | 826,795 | 5.463 | 186.38 | 1018.22 | 535,794 |
| HISTAI-colorectal-b2 | 57 | 88 | 15,241 | 1.544 | 173.19 | 267.39 | 13,550 |
| HISTAI-gastrointestinal | 107 | 183 | 28,954 | 1.710 | 158.22 | 270.60 | 27,363 |
| HISTAI-hematologic | 188 | 188 | 40,849 | 1.000 | 217.28 | 217.28 | 40,849 |
| HISTAI-mixed | 20,924 | 52,019 | 8,271,097 | 2.486 | 159.00 | 395.29 | 8,007,058 |
| HISTAI-skin-b1 | 1,441 | 6,206 | 1,282,142 | 4.307 | 206.60 | 889.76 | 1,138,042 |
| HISTAI-skin-b2 | 18,035 | 37,173 | 5,390,889 | 2.061 | 145.02 | 298.91 | 5,170,041 |
| HISTAI-thorax | 568 | 731 | 130,052 | 1.287 | 177.91 | 228.96 | 128,329 |
| **Total** | **43,618** | **102,703** | **16,331,401** | **2.355** | **159.02** | **374.42** | **15,405,700** |

## Output Files
- JSON/CSV were internal audit artifacts; the repo retains this Markdown summary for audit notes.
