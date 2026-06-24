# Internal Repo Audit Notes

Internal source reviewed: `/home/chengj60/scratch/VLM_MVP`.

Release destination prepared: `/home/chengj60/scratch/PathoSynVLM`.

## Pre-existing Internal Dirty State

The internal repo had pre-existing local changes before release repo preparation:

- Modified: `evaluations/PRISM/run_prism_extract_virchow_embeddings_fir.sh`
- Untracked: `finetune-20260309/logs/`

These were not modified or copied.

## Paper-Relevant Code Paths

Carried into the release repo:

- `3datasets/alignment_dataset_loader.py` -> `pathosynvlm/alignment_dataset.py`
- `3datasets/model_alignment.py` -> `pathosynvlm/alignment_model.py`
- `3datasets/main_alignment.py` -> `scripts/train_stage1_alignment.py`
- `finetune-20260309/model.py` -> `pathosynvlm/model.py`
- `finetune-20260309/histai_dataset.py` -> `pathosynvlm/histai_dataset.py`
- `finetune-20260309/histai_vlm_output_evaluate.py` -> `pathosynvlm/metrics.py`
- `finetune-20260309/train_stage2.py` -> `scripts/train_stage2_histai.py`
- `finetune-20260309/evaluate_stage2_checkpoint.py` -> `scripts/evaluate_checkpoint.py`

Additional scripts added:

- `scripts/prepare_stage1_metadata.py`
- `scripts/prepare_histai_metadata.py`
- `scripts/export_release_weights.py`
- `scripts/generate_case_report.py`

## Internal Areas Intentionally Omitted

Not copied into the release:

- `wandb/`, `runs/`, `logs/`, `__pycache__/`, `.venv/`
- HistoGPT/PRISM/MedGemma/WSI-LLaVA/TRIDENT/prov-gigapath evaluation folders except where results are reflected in the paper.
- `finetune-20260309` follow-up experiments: full/norm-only/second-half LLM finetuning, anchor regularization, mixed Stage 1 replay, micro-protocol target, EMA variants.
- PathText as part of the default Stage 1 pipeline. The metadata script keeps an optional `--include-pathtext` compatibility flag, but the paper default is HistGen + REG2025.

## Result Mapping

Stage 2 internal runs corresponding to paper tables:

- Main Stage 2 table: `histai_ft_5x512_from_5x512_all_baseline_prompt_double`
- Baseline ablation row: `histai_ft_5x512_from_5x512_all_baseline`
- Unique WSI marker row: `histai_ft_5x512_from_5x512_all_baseline_wsi_markers`
- Shared WSI marker row: `histai_ft_5x512_from_5x512_all_baseline_use_markers_no_index`
- Vision-token dropout row: `histai_ft_5x512_from_5x512_all_baseline_dropout_02`

The internal run configs show `use_wsi_markers=true` for multiple rows, while the paper table labels one row as no WSI marker. The configs therefore preserve the internal run arguments and report the paper values separately in `configs/reported_results.json`.

## Weight Release Note

The paper Stage 2 runs used `unfreeze_llm_base=true`. A standalone LoRA adapter is not enough for exact inference unless the trained base update is included. The release should upload a merged HF model package generated with `scripts/export_release_weights.py`, plus `vlm_state.pt` for the aligner and WSI marker tensors.
