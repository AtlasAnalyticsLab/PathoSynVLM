from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathosynvlm.model import VLM_MVP


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_checkpoint_step(run_dir: Path, checkpoint_step: int) -> int:
    if int(checkpoint_step) >= 0:
        return int(checkpoint_step)
    summary_path = run_dir / "best_checkpoint_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint summary: {summary_path}")
    summary = _load_json(summary_path)
    step = int(summary.get("best_step", -1))
    if step < 0:
        raise ValueError(f"Invalid best_step in {summary_path}")
    return step


def _resolve_lora_dir(run_dir: Path, step: int) -> Path | None:
    for name in (f"lora_step_{int(step)}", f"llm_step_{int(step)}", "best_llm"):
        p = run_dir / name
        if p.is_dir():
            return p
    return None


def _torch_load(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _release_train_config(train_args: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "llm",
        "vision_dim",
        "feature_key",
        "patch_level",
        "use_wsi_markers",
        "use_wsi_index_emb",
        "prompt_style",
        "report_target_field",
        "report_target_label",
        "max_text_length",
        "max_vision_tokens",
        "vision_token_dropout",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target",
        "use_lora",
        "unfreeze_llm_base",
    ]
    return {k: train_args[k] for k in keys if k in train_args}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a PathoSynVLM run into a compact inference weight package.")
    p.add_argument("--run_dir", type=Path, required=True, help="Training run directory containing train_args.json.")
    p.add_argument("--output_dir", type=Path, required=True, help="Destination release-weight directory.")
    p.add_argument("--checkpoint_step", type=int, default=-1, help="-1 uses best_checkpoint_summary.json.")
    p.add_argument("--base_llm", type=str, default="", help="Override base LLM path/name from train_args.json.")
    p.add_argument("--no_merge_lora", action="store_true", help="Keep PEFT adapter separate instead of saving merged LLM.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not bool(args.overwrite):
            raise FileExistsError(f"Output exists; pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_args = _load_json(run_dir / "train_args.json")
    step = _resolve_checkpoint_step(run_dir, int(args.checkpoint_step))
    state_path = run_dir / f"trainer_state_step_{step}.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing trainer state: {state_path}")

    base_llm = str(args.base_llm or train_args.get("llm", "Qwen/Qwen2.5-3B-Instruct"))
    dtype = None
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16

    model = VLM_MVP(
        llm_name_or_path=base_llm,
        vision_dim=int(train_args.get("vision_dim", 768)),
        feature_key=str(train_args.get("feature_key", "conch_v15")),
        torch_dtype=dtype,
        use_wsi_markers=bool(train_args.get("use_wsi_markers", True)),
        use_index_emb=bool(train_args.get("use_wsi_index_emb", True)),
    )

    use_lora = bool(train_args.get("use_lora", True))
    lora_dir = _resolve_lora_dir(run_dir, step)
    if use_lora and lora_dir is not None:
        from peft import PeftModel

        model.llm = PeftModel.from_pretrained(model.llm, str(lora_dir), is_trainable=False)

    state = _torch_load(state_path, map_location="cpu")
    llm_state = state.get("llm_state_dict") if isinstance(state, dict) else None
    if isinstance(llm_state, dict):
        missing, unexpected = model.llm.load_state_dict(llm_state, strict=False)
        print(f"Loaded llm_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    elif bool(train_args.get("unfreeze_llm_base", False)):
        print(
            "WARNING: train_args has unfreeze_llm_base=true, but this checkpoint has no "
            "llm_state_dict. Exported weights may not exactly match in-memory validation."
        )

    aligner_sd = state.get("aligner")
    if isinstance(aligner_sd, dict):
        model.aligner.load_state_dict(aligner_sd, strict=True)
    if bool(getattr(model, "use_wsi_markers", False)):
        idx_sd = state.get("wsi_index_emb")
        if isinstance(idx_sd, dict) and hasattr(model, "wsi_index_emb"):
            model.wsi_index_emb.load_state_dict(idx_sd, strict=True)
        wsi_sep = state.get("wsi_sep")
        if torch.is_tensor(wsi_sep) and hasattr(model, "wsi_sep"):
            model.wsi_sep.data.copy_(wsi_sep.to(dtype=model.wsi_sep.dtype))

    tokenizer_source = run_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source if tokenizer_source.is_dir() else base_llm),
        trust_remote_code=bool(args.trust_remote_code),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    merge_lora = bool(use_lora and lora_dir is not None and not args.no_merge_lora)
    if merge_lora and hasattr(model.llm, "merge_and_unload"):
        model.llm = model.llm.merge_and_unload()
        llm_output = output_dir / "llm"
        model.llm.save_pretrained(llm_output, safe_serialization=True)
        llm_kind = "merged"
    elif use_lora and lora_dir is not None:
        adapter_output = output_dir / "adapter"
        shutil.copytree(lora_dir, adapter_output)
        llm_output = Path(base_llm)
        llm_kind = "peft_adapter"
    else:
        llm_output = output_dir / "llm"
        model.llm.save_pretrained(llm_output, safe_serialization=True)
        llm_kind = "full"

    tokenizer.save_pretrained(output_dir / "tokenizer")
    torch.save(
        {
            "aligner": model.aligner.state_dict(),
            "use_wsi_markers": bool(getattr(model, "use_wsi_markers", False)),
            "use_wsi_index_emb": bool(getattr(model, "use_index_emb", True)),
            "wsi_index_emb": model.wsi_index_emb.state_dict() if hasattr(model, "wsi_index_emb") else None,
            "wsi_sep": model.wsi_sep.detach().cpu() if hasattr(model, "wsi_sep") else None,
        },
        output_dir / "vlm_state.pt",
    )

    release_config = {
        "format_version": 1,
        "source_run_name": run_dir.name,
        "checkpoint_step": int(step),
        "base_llm": base_llm,
        "llm_kind": llm_kind,
        "llm_path": str(llm_output.name if llm_output.is_relative_to(output_dir) else llm_output),
        "vision_dim": int(train_args.get("vision_dim", 768)),
        "feature_key": str(train_args.get("feature_key", "conch_v15")),
        "patch_level": str(train_args.get("patch_level", "5x_512")),
        "use_wsi_markers": bool(train_args.get("use_wsi_markers", True)),
        "use_wsi_index_emb": bool(train_args.get("use_wsi_index_emb", True)),
        "prompt_style": str(train_args.get("prompt_style", "single")),
        "report_target_field": str(train_args.get("report_target_field", "conclusion")),
        "report_target_label": str(train_args.get("report_target_label", "")),
        "training_config": _release_train_config(train_args),
    }
    (output_dir / "config.json").write_text(json.dumps(release_config, indent=2), encoding="utf-8")
    if (run_dir / "best_checkpoint_summary.json").exists():
        shutil.copy2(run_dir / "best_checkpoint_summary.json", output_dir / "best_checkpoint_summary.json")
    print(f"Wrote release package: {output_dir}")


if __name__ == "__main__":
    main()
