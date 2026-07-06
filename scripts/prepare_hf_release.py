from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("release/huggingface")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_checkpoint_step(source_run_dir: Path, checkpoint_step: int) -> int:
    if int(checkpoint_step) >= 0:
        return int(checkpoint_step)
    summary = _load_json(source_run_dir / "best_checkpoint_summary.json")
    return int(summary["best_step"])


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _format_metric(value: Any) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else "n/a"


def _model_card(
    *,
    repo_id: str,
    github_url: str,
    train_args: dict[str, Any],
    best_summary: dict[str, Any],
    reported_results: dict[str, Any],
) -> str:
    stage1 = reported_results.get("stage1_baseline", {})
    stage2 = reported_results.get("stage2_main_prompt_double", {})
    base_model = str(train_args.get("llm", "Qwen/Qwen2.5-3B-Instruct"))
    checkpoint_step = int(best_summary.get("best_step", 30400))
    checkpoint_epoch = int(best_summary.get("best_epoch", 7))
    val_loss = float(best_summary.get("best_val_loss", 1.0108920872211455))
    return dedent(
        f"""\
        ---
        license: cc-by-nc-sa-4.0
        language:
        - en
        library_name: transformers
        pipeline_tag: image-text-to-text
        base_model:
        - {base_model}
        tags:
        - pathology
        - computational-pathology
        - digital-pathology
        - histopathology
        - whole-slide-image
        - vision-language-model
        - report-generation
        - synoptic-report
        - case-level
        - conch
        - qwen2.5
        datasets:
        - david4real/HistGen
        metrics:
        - rouge
        - meteor
        - bleu
        - bertscore
        arxiv: 2605.30716
        model-index:
        - name: PathoSynVLM
          results:
          - task:
              type: image-text-to-text
              name: Case-level pathology synoptic report generation
            dataset:
              name: HISTAI
              type: HISTAI case-report pairs
            metrics:
            - type: rouge
              name: ROUGE-L
              value: {_format_metric(stage2.get("rougeL"))}
            - type: meteor
              name: METEOR
              value: {_format_metric(stage2.get("meteor"))}
            - type: bleu
              name: BLEU-4
              value: {_format_metric(stage2.get("bleu4"))}
            - type: bertscore
              name: BERTScore F1
              value: {_format_metric(stage2.get("bertscore_f1"))}
        ---

        # PathoSynVLM: Case-Level Pathology Synoptic Report Generation

        PathoSynVLM is a token-efficient vision-language model for generating
        case-level pathology synoptic reports from one or more whole-slide
        images represented as precomputed CONCHv1.5 patch embeddings.

        Paper: [Simple Token-Efficient Vision-Language Model for Case-level
        Pathology Synoptic Report Generation](https://arxiv.org/abs/2605.30716)

        Code: [{github_url}]({github_url})

        ![PathoSynVLM architecture](assets/paper_architecture.png)

        ## What This Repository Contains

        This Hugging Face repository contains the exported Stage 2 PathoSynVLM
        model package:

        ```text
        README.md
        LICENSE
        config.json
        vlm_state.pt
        labels.json
        tokenizer/
        llm/
        best_checkpoint_summary.json
        assets/
        ```

        The paper run used `unfreeze_llm_base=true`, so the release package
        includes the merged/full language-model weights under `llm/`, not only a
        LoRA adapter.

        ## Quickstart

        Install the code repository:

        ```bash
        git clone {github_url} PathoSynVLM
        cd PathoSynVLM
        conda create -n pathosynvlm python=3.11 -y
        conda activate pathosynvlm
        export PYTHONNOUSERSITE=1
        pip install -e .
        ```

        Download this model repository:

        ```bash
        source configs/paths.example.env
        hf download {repo_id} --local-dir "$PATHOSYNVLM_WEIGHTS_ROOT/pathosynvlm-stage2-main"
        ```

        Generate a case-level report from one or more slide embedding files:

        ```bash
        python scripts/generate_case_report.py \\
          --embeddings slide_1.h5 slide_2.h5 \\
          --output_json report.json
        ```

        Relative `--embeddings` paths are resolved under
        `PATHOSYNVLM_EMBEDDINGS_ROOT`. Absolute `.h5` paths also work.

        The output follows:

        ```text
        Diagnosis: ...
        Certainty: ...
        Conclusion: ...
        ```

        ## Input Format

        PathoSynVLM runs on precomputed WSI patch embeddings, not raw WSIs. Each
        `.h5` file should contain:

        ```text
        /features/conch_v15  # shape: (num_patches, 768)
        ```

        Use the code repository docs for dataset placement, metadata filtering,
        and embedding generation details.

        ### From Precomputed H5 Feature Files

        This is the fastest path. Put one or more WSI embedding files for a case
        into the `--embeddings` argument:

        ```bash
        python scripts/generate_case_report.py \\
          --embeddings case_slide_1.h5 case_slide_2.h5 \\
          --output_json report.json
        ```

        ### From Raw Whole-Slide Images

        First extract tissue patches and CONCHv1.5 patch embeddings using a WSI
        preprocessing pipeline that writes the H5 layout above. Then pass the
        resulting H5 files to `scripts/generate_case_report.py`.

        ## Running The Paper Pipeline

        To rerun training and evaluation, follow the code repository guide:

        ```bash
        git clone {github_url} PathoSynVLM
        cd PathoSynVLM
        export PYTHONNOUSERSITE=1
        pip install -e .

        # 1. Download HistGen, REG2025, and HISTAI metadata/WSIs.
        # 2. Extract CONCHv1.5 H5 patch embeddings.
        # 3. Prepare Stage 1 and Stage 2 metadata.
        # 4. Train Stage 1 alignment.
        # 5. Train Stage 2 HISTAI report generation.
        # 6. Evaluate with scripts/evaluate_checkpoint.py.
        ```

        ## Training Recipe

        - Stage 1: train the two-layer MLP aligner on HistGen + REG2025 while
          keeping the CONCHv1.5 patch encoder and LLM frozen.
        - Stage 2: finetune on HISTAI case-report pairs with WSI marker tokens.

        Checkpoint selected for release:

        - checkpoint step: `{checkpoint_step}`
        - checkpoint epoch: `{checkpoint_epoch}`
        - validation loss: `{val_loss:.6f}`
        - prompt style: `{train_args.get("prompt_style", "double")}`
        - patch level: `{train_args.get("patch_level", "5x_512")}`
        - max vision tokens: `{train_args.get("max_vision_tokens", 4096)}`

        ## Reported Metrics

        Stage 1 aligner-only training:

        | ROUGE-L | METEOR | BLEU-4 | BERTScore F1 |
        |---:|---:|---:|---:|
        | {_format_metric(stage1.get("rougeL"))} | {_format_metric(stage1.get("meteor"))} | {_format_metric(stage1.get("bleu4"))} | {_format_metric(stage1.get("bertscore_f1"))} |

        Stage 2 HISTAI main result:

        | ROUGE-L | METEOR | BLEU-4 | BERTScore F1 | Diagnosis Exact | Diagnosis Relaxed | Certainty |
        |---:|---:|---:|---:|---:|---:|---:|
        | {_format_metric(stage2.get("rougeL"))} | {_format_metric(stage2.get("meteor"))} | {_format_metric(stage2.get("bleu4"))} | {_format_metric(stage2.get("bertscore_f1"))} | {_format_metric(stage2.get("diagnosis_exact"))} | {_format_metric(stage2.get("diagnosis_relaxed"))} | {_format_metric(stage2.get("certainty_exact"))} |

        ## Intended Use

        This model is intended for research on pathology report generation from
        precomputed WSI patch embeddings.

        It is not a clinical diagnostic device and should not be used for patient
        care without appropriate validation, regulatory review, and expert
        oversight.

        ## License and Commercial Use

        This repository uses CC BY-NC-SA 4.0. Research and non-commercial use only.
        Dataset access, pretrained third-party models, and any externally hosted
        model weights remain subject to their own terms.

        ## Citation

        ```bibtex
        @misc{{yang2026simpletokenefficientvisionlanguage,
          title={{Simple Token-Efficient Vision-Language Model for Case-level Pathology Synoptic Report Generation}},
          author={{Zhiyuan Yang and Jiahao Cheng and Vincent Quoc-Huy Trinh and Mahdi S. Hosseini}},
          year={{2026}},
          eprint={{2605.30716}},
          archivePrefix={{arXiv}},
          primaryClass={{cs.CV}},
          url={{https://arxiv.org/abs/2605.30716}}
        }}
        ```
        """
    )


def _model_index(reported_results: dict[str, Any]) -> dict[str, Any]:
    stage2 = reported_results.get("stage2_main_prompt_double", {})
    return {
        "name": "PathoSynVLM",
        "results": [
            {
                "task": {
                    "type": "image-text-to-text",
                    "name": "Case-level pathology synoptic report generation",
                },
                "dataset": {
                    "name": "HISTAI",
                    "type": "HISTAI case-report pairs",
                },
                "metrics": [
                    {"type": "rouge", "name": "ROUGE-L", "value": stage2.get("rougeL")},
                    {"type": "meteor", "name": "METEOR", "value": stage2.get("meteor")},
                    {"type": "bleu", "name": "BLEU-4", "value": stage2.get("bleu4")},
                    {"type": "bertscore", "name": "BERTScore F1", "value": stage2.get("bertscore_f1")},
                    {"type": "accuracy", "name": "Diagnosis Exact", "value": stage2.get("diagnosis_exact")},
                    {"type": "accuracy", "name": "Diagnosis Relaxed", "value": stage2.get("diagnosis_relaxed")},
                    {"type": "accuracy", "name": "Certainty", "value": stage2.get("certainty_exact")},
                ],
            }
        ],
    }


def _labels(train_args: dict[str, Any]) -> dict[str, Any]:
    target_field = str(train_args.get("report_target_field", "conclusion") or "conclusion")
    target_label = str(train_args.get("report_target_label", "") or "")
    conclusion_label = target_label or target_field.replace("_", " ").title()
    if target_field == "conclusion":
        conclusion_label = target_label or "Conclusion"
    return {
        "task": "case_level_pathology_synoptic_report_generation",
        "input": {
            "type": "one_or_more_h5_wsi_embedding_files",
            "feature_dataset": "/features/conch_v15",
            "feature_dim": int(train_args.get("vision_dim", 768)),
            "patch_level": str(train_args.get("patch_level", "5x_512")),
        },
        "output_fields": [
            {
                "name": "diagnosis",
                "label": "Diagnosis",
                "type": "free_text",
                "required": True,
            },
            {
                "name": "certainty",
                "label": "Certainty",
                "type": "free_text_or_percentage",
                "required": True,
            },
            {
                "name": target_field,
                "label": conclusion_label,
                "type": "free_text",
                "required": True,
            },
        ],
        "expected_text_format": f"Diagnosis: ...\\nCertainty: ...\\n{conclusion_label}: ...",
    }


def _effective_train_args_subset(train_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "llm": train_args.get("llm", "Qwen/Qwen2.5-3B-Instruct"),
        "vision_dim": int(train_args.get("vision_dim", 768)),
        "feature_key": train_args.get("feature_key", "conch_v15"),
        "patch_level": train_args.get("patch_level", "5x_512"),
        "prompt_style": train_args.get("prompt_style", "double"),
        "max_text_length": int(train_args.get("max_text_length", 384)),
        "max_vision_tokens": int(train_args.get("max_vision_tokens", 4096)),
        "use_wsi_markers": bool(train_args.get("use_wsi_markers", True)),
        "use_wsi_index_emb": bool(train_args.get("use_wsi_index_emb", True)),
        "use_lora": bool(train_args.get("use_lora", True)),
        "unfreeze_llm_base": bool(train_args.get("unfreeze_llm_base", True)),
        "lora_r": int(train_args.get("lora_r", 16)),
        "lora_alpha": int(train_args.get("lora_alpha", 32)),
        "lora_dropout": float(train_args.get("lora_dropout", 0.05)),
        "lora_target": train_args.get("lora_target", "q_proj,k_proj,v_proj,o_proj"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Hugging Face model repo root and upload notes.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-repo-dir", type=Path, default=None, help="Directory used as the actual HF model repo root.")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face model repo id, e.g. org/pathosynvlm-stage2-main.")
    parser.add_argument("--github-url", type=str, required=True, help="Public GitHub repository URL.")
    parser.add_argument("--source-run-dir", type=Path, required=True, help="Completed Stage 2 training run used for metadata refresh.")
    parser.add_argument("--checkpoint-step", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    hf_repo_dir = (args.hf_repo_dir or output_dir / "hf_repo_preview").resolve()
    source_run_dir = args.source_run_dir.resolve()

    if bool(args.overwrite) and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_repo_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_step = _resolve_checkpoint_step(source_run_dir, int(args.checkpoint_step))
    best_summary = _load_json(source_run_dir / "best_checkpoint_summary.json")
    train_args = _load_json(source_run_dir / "train_args.json")
    reported_results = _load_json(REPO_ROOT / "configs" / "reported_results.json")

    assets_dir = hf_repo_dir / "assets"
    _copy_if_exists(REPO_ROOT / "assets" / "paper_architecture.png", assets_dir / "paper_architecture.png")
    _copy_if_exists(REPO_ROOT / "assets" / "reported_results.svg", assets_dir / "reported_results.svg")
    _copy_if_exists(REPO_ROOT / "LICENSE", hf_repo_dir / "LICENSE")

    (hf_repo_dir / "README.md").write_text(
        _model_card(
            repo_id=str(args.repo_id),
            github_url=str(args.github_url),
            train_args=train_args,
            best_summary=best_summary,
            reported_results=reported_results,
        ),
        encoding="utf-8",
    )
    (hf_repo_dir / "model_index.json").write_text(
        json.dumps(_model_index(reported_results), indent=2) + "\n",
        encoding="utf-8",
    )
    (hf_repo_dir / "labels.json").write_text(
        json.dumps(_labels(train_args), indent=2) + "\n",
        encoding="utf-8",
    )
    examples_dir = hf_repo_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    (examples_dir / "case_input_example.json").write_text(
        json.dumps(
            {
                "case_id": "example_case",
                "slide_embeddings": [
                    "HISTAI-example/conch_v15/5x_512/patches/slide_1.h5",
                    "HISTAI-example/conch_v15/5x_512/patches/slide_2.h5",
                ],
                "feature_key": "conch_v15",
                "command": (
                    "python scripts/generate_case_report.py "
                    "--embeddings HISTAI-example/conch_v15/5x_512/patches/slide_1.h5 "
                    "HISTAI-example/conch_v15/5x_512/patches/slide_2.h5 --output_json report.json"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (hf_repo_dir / ".gitattributes").write_text(
        dedent(
            """\
            *.safetensors filter=lfs diff=lfs merge=lfs -text
            *.bin filter=lfs diff=lfs merge=lfs -text
            *.pt filter=lfs diff=lfs merge=lfs -text
            *.pth filter=lfs diff=lfs merge=lfs -text
            """
        ),
        encoding="utf-8",
    )

    required_weight_paths = ["config.json", "vlm_state.pt", "tokenizer", "llm", "best_checkpoint_summary.json"]
    missing = [p for p in required_weight_paths if not (hf_repo_dir / p).exists()]
    if missing:
        missing_list = "\n".join(f"- {p}" for p in missing)
        raise SystemExit(
            "HF repo root is missing required exported weight entries:\n"
            f"{missing_list}\n\n"
            "Run scripts/export_release_weights.py first, then rerun this script."
        )

    manifest = {
        "staging_version": 1,
        "repo_id": str(args.repo_id),
        "github_url": str(args.github_url),
        "hf_repo_dir": str(hf_repo_dir),
        "checkpoint_step": int(checkpoint_step),
        "best_summary": best_summary,
        "train_args_subset": _effective_train_args_subset(train_args),
        "required_hf_files": required_weight_paths
        + ["README.md", "LICENSE", "model_index.json", "labels.json", ".gitattributes"],
        "missing_weight_entries": missing,
        "reported_results": reported_results,
    }
    (output_dir / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    export_section = dedent(
        f"""\
        ## 1. Weight Status

        Actual uploadable weights are present in `{hf_repo_dir}`.

        Regenerate them only if the source checkpoint or export logic changes:

        ```bash
        cd {REPO_ROOT}
        export PATHOSYNVLM_RUNS_ROOT="${{PATHOSYNVLM_RUNS_ROOT:-$PWD/runs}}"
        export PATHOSYNVLM_STAGE2_RUN="$PATHOSYNVLM_RUNS_ROOT/stage2_main"
        python scripts/export_release_weights.py \\
          --run_dir "$PATHOSYNVLM_STAGE2_RUN" \\
          --output_dir {hf_repo_dir} \\
          --checkpoint_step {checkpoint_step} \\
          --overwrite
        python scripts/prepare_hf_release.py \\
          --output-dir {output_dir} \\
          --hf-repo-dir {hf_repo_dir} \\
          --repo-id {args.repo_id} \\
          --github-url {args.github_url} \\
          --source-run-dir "$PATHOSYNVLM_STAGE2_RUN"
        ```
        """
    ).strip()
    upload_instructions = f"""# Hugging Face Upload Instructions

Review folder:

```text
{output_dir}
```

HF repo root:

```text
{hf_repo_dir}
```

Weight status: `present`

{export_section}

## Inspect required files

```bash
find {hf_repo_dir} -maxdepth 2 -type f | sort
find {hf_repo_dir} -type l
python -m json.tool {output_dir}/release_manifest.json >/dev/null
```

The final HF repo root should contain:

```text
README.md
LICENSE
.gitattributes
model_index.json
labels.json
config.json
vlm_state.pt
best_checkpoint_summary.json
tokenizer/
llm/
assets/
```

## Upload after you provide the final repo id

```bash
hf repos create {args.repo_id} --type model --private --exist-ok
hf upload-large-folder {args.repo_id} {hf_repo_dir} --type model
```

Upload only `{hf_repo_dir}`.
"""
    (output_dir / "UPLOAD_INSTRUCTIONS.md").write_text(upload_instructions, encoding="utf-8")

    print(f"Wrote HF review staging folder: {output_dir}")
    print(f"HF repo preview folder: {hf_repo_dir}")
    print(f"Missing weight entries: {missing}")


if __name__ == "__main__":
    main()
