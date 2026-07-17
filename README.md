# PathoSynVLM: Case-Level Pathology Report Generation

[![arXiv](https://img.shields.io/badge/arXiv-2605.30716-b31b1b.svg)](https://arxiv.org/abs/2605.30716)
[![Project Page](https://img.shields.io/badge/Project-Page-0e8a9c.svg)](https://atlasanalyticslab.github.io/PathoSynVLM/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](https://www.python.org/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](LICENSE)
[![Model weights](https://img.shields.io/badge/model_weights-Hugging_Face-blue.svg)](https://huggingface.co/AtlasAnalyticsLab/PathoSynVLM)

![PathoSynVLM architecture from the paper](assets/paper_architecture.png)

**PathoSynVLM is a simple token-efficient vision-language model for generating case-level pathology synoptic reports from one or more whole-slide images.** It keeps the pathology patch encoder frozen, learns a compact visual-to-language aligner, and uses a Qwen2.5 instruction decoder to produce structured report fields:

```text
Diagnosis: ...
Certainty: ...
Conclusion: ...
```

Code and model-release utilities for:

> **Simple Token-Efficient Vision-Language Model for Case-level Pathology Synoptic Report Generation**  
> Zhiyuan Yang, Jiahao Cheng, Vincent Quoc-Huy Trinh, Mahdi S. Hosseini  
> *Proceedings of the 7th International Conference on Deep Learning Theory and Applications*, 2026, pp. 514–537

Paper: https://arxiv.org/abs/2605.30716

Project page: https://atlasanalyticslab.github.io/PathoSynVLM/

Model weights: https://huggingface.co/AtlasAnalyticsLab/PathoSynVLM

## News

- **2026/07** PathoSynVLM model weights released on Hugging Face.
- **2026/06** Repository prepared with paper-aligned training, evaluation, inference, and weight-export utilities.
- **2026/05** PathoSynVLM preprint released on arXiv.

## Table Of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Weights](#model-weights)
- [Configure Local Paths](#configure-local-paths)
- [Headline Results](#headline-results)
- [Method Overview](#method-overview)
- [Data And Embeddings](#data-and-embeddings)
- [Run The Paper Pipeline](#run-the-paper-pipeline)
- [Repository Map](#repository-map)
- [Runtime Notes](#runtime-notes)
- [Optional SLURM Jobs](#optional-slurm-jobs)
- [Acknowledgments](#acknowledgments)
- [Notes From The Authors](#notes-from-the-authors)
- [Citation](#citation)
- [License](#license)

## Installation

PathoSynVLM uses standard Python tooling and does not require SLURM. A CUDA-capable GPU is recommended for practical inference and required for full training runs; CPU inference is useful for smoke tests but can be slow.

```bash
git clone https://github.com/AtlasAnalyticsLab/PathoSynVLM
cd PathoSynVLM

conda create -n pathosynvlm python=3.11 -y
conda activate pathosynvlm
export PYTHONNOUSERSITE=1
pip install -e .
```

`PYTHONNOUSERSITE=1` is optional on clean workstations, but it helps on shared servers where user-site packages may conflict with the environment.

## Quick Start

### Option A: Use The Released Model

This is the path for users who want to generate reports without retraining.

Model weights are distributed separately from Git. Download the released weight package into:

```text
$PATHOSYNVLM_WEIGHTS_ROOT/pathosynvlm-stage2-main/
```

Expected layout:

```text
$PATHOSYNVLM_WEIGHTS_ROOT/pathosynvlm-stage2-main/
  config.json
  vlm_state.pt
  tokenizer/
  llm/
  best_checkpoint_summary.json
```

Download from Hugging Face:

```bash
source configs/paths.example.env

export PATHOSYNVLM_HF_REPO=AtlasAnalyticsLab/PathoSynVLM
hf download "$PATHOSYNVLM_HF_REPO" \
  --local-dir "$PATHOSYNVLM_WEIGHTS_ROOT/pathosynvlm-stage2-main"

python scripts/generate_case_report.py \
  --embeddings HISTAI-skin-b2/conch_v15/5x_512/patches/example_1.h5 \
               HISTAI-skin-b2/conch_v15/5x_512/patches/example_2.h5 \
  --output_json report.json
```

Replace the example `.h5` paths with the slide embedding files for one case. Relative embedding paths are resolved under `PATHOSYNVLM_EMBEDDINGS_ROOT`; absolute paths also work.

Users **do not need to create weights themselves** for inference once the model weights are available. The export script exists for converting a completed training run into the release weight layout.

### Option B: Train From Scratch

This is the path for rerunning the paper training and evaluation pipeline end to end.

Then follow:

1. Download the datasets described in [docs/data.md](docs/data.md).
2. Generate CONCHv1.5 patch embeddings using the layout in [docs/embeddings.md](docs/embeddings.md).
3. Prepare metadata with `scripts/prepare_stage1_metadata.py` and `scripts/prepare_histai_metadata.py`.
4. Train Stage 1 alignment.
5. Train Stage 2 HISTAI report generation.
6. Evaluate and compare against [configs/reported_results.json](configs/reported_results.json).
7. Optionally export your trained checkpoint with `scripts/export_release_weights.py`.

## Model Weights

There are two different weight workflows:

| Workflow | Who uses it | What happens |
|---|---|---|
| **Download released weights** | Most users | Download the released `$PATHOSYNVLM_WEIGHTS_ROOT/pathosynvlm-stage2-main/` package and run inference or evaluation directly. |
| **Export weights** | Retrainers | Run `scripts/export_release_weights.py` on a completed Stage 2 training run to create the inference package. |

The paper Stage 2 run used `unfreeze_llm_base=true`, so a LoRA adapter alone is not enough for exact release inference. The Hugging Face package contains a merged/full Hugging Face LLM directory plus `vlm_state.pt` for the aligner and WSI marker tensors.

Official model repository: [AtlasAnalyticsLab/PathoSynVLM](https://huggingface.co/AtlasAnalyticsLab/PathoSynVLM).

Export command for the authors:

```bash
python scripts/export_release_weights.py \
  --run_dir "$PATHOSYNVLM_RUNS_ROOT/stage2_main" \
  --output_dir "$PATHOSYNVLM_WEIGHTS_ROOT/pathosynvlm-stage2-main" \
  --overwrite
```

Read [docs/weights.md](docs/weights.md) for the download/export distinction.
The Hugging Face upload root and validation steps are in [docs/huggingface_release.md](docs/huggingface_release.md).

The release model card is in [MODEL_CARD.md](MODEL_CARD.md).

## Configure Local Paths

PathoSynVLM can use repo-local storage or external dataset/embedding folders. By default, scripts look under `data/`, `runs/`, and `weights/` inside the repository. If your raw metadata, embeddings, runs, or weights already live elsewhere, export these variables once before running commands:

```bash
export PATHOSYNVLM_DATA_ROOT="$PWD/data"
export PATHOSYNVLM_RAW_DATA_ROOT="$PATHOSYNVLM_DATA_ROOT/raw"
export PATHOSYNVLM_EMBEDDINGS_ROOT="$PATHOSYNVLM_DATA_ROOT/embeddings"
export PATHOSYNVLM_STAGE1_METADATA_DIR="$PATHOSYNVLM_DATA_ROOT/stage1"
export PATHOSYNVLM_HISTAI_METADATA_DIR="$PATHOSYNVLM_DATA_ROOT/histai"
export PATHOSYNVLM_RUNS_ROOT="$PWD/runs"
export PATHOSYNVLM_WEIGHTS_ROOT="$PWD/weights"
```

The same defaults are provided in [configs/paths.example.env](configs/paths.example.env):

```bash
source configs/paths.example.env
```

For external storage, set only the paths you need, for example:

```bash
export PATHOSYNVLM_RAW_DATA_ROOT="/data/pathosynvlm/raw"
export PATHOSYNVLM_EMBEDDINGS_ROOT="/features/pathosynvlm/conch_embeddings"
export PATHOSYNVLM_STAGE1_METADATA_DIR="/outputs/pathosynvlm/stage1_metadata"
export PATHOSYNVLM_HISTAI_METADATA_DIR="/outputs/pathosynvlm/histai_metadata"
export PATHOSYNVLM_RUNS_ROOT="/outputs/pathosynvlm/runs"
export PATHOSYNVLM_WEIGHTS_ROOT="/models/pathosynvlm"
```

Every path can still be overridden per command with flags such as `--dataset-embeddings-root`, `--metadata_json`, `--metadata_standardized_json`, `--embedding-root`, `--weights`, or `--output_dir`.

## Headline Results

![PathoSynVLM reported results](assets/reported_results.svg)

### Stage 1 Alignment

| ROUGE-L | METEOR | BLEU-4 | BERTScore F1 |
|---:|---:|---:|---:|
| 0.4743 | 0.4810 | 0.1247 | 0.4253 |

### Stage 2 HISTAI Main Result

| ROUGE-L | METEOR | BLEU-4 | BERTScore F1 | Diagnosis Exact | Diagnosis Relaxed | Certainty |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2495 | 0.1988 | 0.0525 | 0.3018 | 0.1667 | 0.3333 | 0.9000 |

The training logs use sacreBLEU percentage scale, so `5.2512` in JSON corresponds to `0.0525` in the paper.

## Method Overview

PathoSynVLM follows a two-stage recipe:

1. **Stage 1: token alignment.** Train only a two-layer MLP aligner that maps frozen CONCHv1.5 patch embeddings into the Qwen2.5-3B-Instruct hidden space. The LLM and pathology encoder stay frozen.
2. **Stage 2: case-level report finetuning.** Finetune on HISTAI case-report pairs with one or more WSIs per case. WSI marker tokens help the decoder separate evidence from different slides.

This repository focuses on the experiments and workflows reported in the PathoSynVLM paper.

## Data And Embeddings

Expected data sources:

- HistGen: https://github.com/dddavid4real/HistGen and https://huggingface.co/datasets/david4real/HistGen
- REG2025: https://reg2025.grand-challenge.org/
- HISTAI: https://github.com/HistAI/HISTAI

Expected layout under the configured roots:

```text
$PATHOSYNVLM_RAW_DATA_ROOT/
  histgen/annotation_update.json
  reg2025/train.json
  histai/standardized_metadata_fixed.json
$PATHOSYNVLM_STAGE1_METADATA_DIR/
  merged_metadata_3datasets_filtered_conch_v15.json
$PATHOSYNVLM_HISTAI_METADATA_DIR/
  standardized_metadata_fixed_filtered_5x_512.json
$PATHOSYNVLM_EMBEDDINGS_ROOT/
  HistGen-train/conch_v15/5x_512/patches/*.h5
  REG_dataset/REG_train/conch_v15/5x_512/patches/*.h5
  HISTAI-*/conch_v15/5x_512/patches/*.h5
```

Each H5 file should contain:

```text
/features/conch_v15  # shape: (num_patches, 768)
```

See [docs/data.md](docs/data.md) and [docs/embeddings.md](docs/embeddings.md) for details.

## Run The Paper Pipeline

Prepare Stage 1 metadata:

```bash
python scripts/prepare_stage1_metadata.py \
  --histgen-json "$PATHOSYNVLM_RAW_DATA_ROOT/histgen/annotation_update.json" \
  --reg-json "$PATHOSYNVLM_RAW_DATA_ROOT/reg2025/train.json" \
  --dataset-embeddings-root "$PATHOSYNVLM_EMBEDDINGS_ROOT" \
  --patch-level 5x_512
```

Prepare Stage 2 metadata:

```bash
python scripts/prepare_histai_metadata.py \
  --metadata-standardized-json "$PATHOSYNVLM_RAW_DATA_ROOT/histai/standardized_metadata_fixed.json" \
  --dataset-embeddings-root "$PATHOSYNVLM_EMBEDDINGS_ROOT" \
  --patch-levels 5x_512
```

Train Stage 1:

```bash
python scripts/train_stage1_alignment.py \
  --metadata_json "$PATHOSYNVLM_STAGE1_METADATA_DIR/merged_metadata_3datasets_filtered_conch_v15.json" \
  --dataset_embeddings_root "$PATHOSYNVLM_EMBEDDINGS_ROOT" \
  --datasets histgen,reg_dataset \
  --output_dir "$PATHOSYNVLM_RUNS_ROOT/stage1_alignment"
```

Train Stage 2 main paper run:

```bash
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
```

Evaluate:

```bash
python scripts/evaluate_checkpoint.py \
  --finetune_run_dir "$PATHOSYNVLM_RUNS_ROOT/stage2_main" \
  --dataset_scope histai \
  --histai_metadata_standardized_json "$PATHOSYNVLM_HISTAI_METADATA_DIR/standardized_metadata_fixed_filtered_5x_512.json" \
  --dataset_embeddings_root "$PATHOSYNVLM_EMBEDDINGS_ROOT" \
  --output_json "$PATHOSYNVLM_RUNS_ROOT/stage2_main/eval_histai.json"
```

The consolidated Stage 1, Stage 2, evaluation, and inference settings are recorded in [configs/paper_hyperparameters.json](configs/paper_hyperparameters.json). The launcher-oriented configs remain in [configs/](configs), and the detailed run guide is in [docs/paper_pipeline.md](docs/paper_pipeline.md).
Path fields in the JSON configs use `$PATHOSYNVLM_*` notation to show the intended roots; the training scripts do not read these JSON files automatically, so expand or replace those strings if you feed the JSON into your own launcher.

## Repository Map

| Path | Purpose |
|---|---|
| `pathosynvlm/` | Model, data loaders, alignment modules, and metrics. |
| `scripts/` | Metadata prep, training, evaluation, inference, and weight export entry points. |
| `configs/` | Paper-aligned configs and reported result values. |
| `docs/` | Data, embedding, paper-pipeline, weight-release, Hugging Face release, and release-checklist docs. |
| `MODEL_CARD.md` | Intended use, limitations, and release-weight notes for the model. |
| `slurm/` | Cluster job templates. |
| `assets/` | README figures. |
| `weights/` | Local directory for downloaded model artifacts. |

## Runtime Notes

- **Inference:** pass one or more `.h5` WSI embedding files to `scripts/generate_case_report.py`. The model accepts multiple slides for a single case in one command.
- **Hardware:** GPU inference is recommended for normal use. CPU inference works for smoke tests and debugging.
- **Training:** reproducing the full paper pipeline requires the datasets, precomputed CONCHv1.5 embeddings, and a GPU environment with enough memory for Qwen2.5-3B-Instruct finetuning.
- **Clusters:** use the same Python commands inside your cluster allocation or batch system. The `slurm/` files are optional examples, not a requirement of the codebase.

## Optional SLURM Jobs

For SLURM clusters, run inside a compute allocation rather than on the login node. Templates are provided:

```bash
sbatch slurm/stage1_alignment.sbatch
sbatch slurm/stage2_histai.sbatch
sbatch slurm/evaluate.sbatch
```

For interactive work:

```bash
export SLURM_ACCOUNT=your-account
salloc --account="$SLURM_ACCOUNT" --time=04:00:00 --gres=gpu:1 --cpus-per-task=4 --mem=64G
srun --pty bash -l
conda activate pathosynvlm
export PYTHONNOUSERSITE=1
```

## Acknowledgments

[OpenAI Codex](https://openai.com/codex/) was used to assist with repository documentation, project-website implementation, GitHub Pages automation, asset preparation, configuration auditing, and maintenance. All AI-assisted changes were directed and reviewed by the project maintainers.

## Notes From The Authors

- This repo is scoped to the experiments reported in the paper.
- `PathText` support remains as an optional compatibility path, but the Stage 1 default is HistGen + REG2025.
- The WSI-marker ablation settings are summarized in [configs/stage2_wsi_marker_ablation.json](configs/stage2_wsi_marker_ablation.json).
- Raw WSIs, extracted H5 embeddings, checkpoints, and released weights are intentionally kept outside Git.

## Citation

Yang, Z.; Cheng, J.; Trinh, V. Q.-H. and Hosseini, M. S. (2026). **Simple Token-Efficient Vision-Language Model for Case-Level Pathology Synoptic Report Generation.** In *Proceedings of the 7th International Conference on Deep Learning Theory and Applications*, ISSN 2184-9277, pages 514–537.

```bibtex
@inproceedings{yang2026simpletokenvlm,
  title     = {Simple Token-Efficient Vision-Language Model for Case-Level Pathology Synoptic Report Generation},
  author    = {Yang, Zhiyuan and Cheng, Jiahao and Trinh, Vincent Quoc-Huy and Hosseini, Mahdi S.},
  booktitle = {Proceedings of the 7th International Conference on Deep Learning Theory and Applications},
  pages     = {514--537},
  year      = {2026},
  issn      = {2184-9277}
}
```

## License

This repository uses **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. See [LICENSE](LICENSE).

Datasets, pretrained third-party models, and externally hosted model weights may have separate terms.
