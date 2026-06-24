from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathosynvlm.histai_dataset import resolve_prompt_text, resolve_target_field_label
from pathosynvlm.model import VLM_MVP


def _read_h5_features(path: Path, feature_key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        features_obj = f.get("features")
        if features_obj is None or not isinstance(features_obj, h5py.Group):
            raise KeyError(f"Missing /features group in {path}")
        ds = features_obj.get(feature_key)
        if ds is None:
            keys = list(features_obj.keys())
            if len(keys) != 1:
                raise KeyError(f"Missing /features/{feature_key} in {path}; available={keys}")
            ds = features_obj.get(keys[0])
        if not isinstance(ds, h5py.Dataset):
            raise TypeError(f"Feature object is not an HDF5 dataset in {path}")
        arr = ds[:]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix in {path}, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_release_model(weights_dir: Path, device: torch.device, dtype: torch.dtype | None) -> tuple[VLM_MVP, Any, dict[str, Any]]:
    cfg = _load_json(weights_dir / "config.json")
    llm_path = weights_dir / str(cfg.get("llm_path", "llm"))
    if not llm_path.exists():
        llm_path = Path(str(cfg.get("base_llm", "Qwen/Qwen2.5-3B-Instruct")))

    model = VLM_MVP(
        llm_name_or_path=str(llm_path),
        vision_dim=int(cfg.get("vision_dim", 768)),
        feature_key=str(cfg.get("feature_key", "conch_v15")),
        torch_dtype=dtype if device.type == "cuda" else None,
        use_wsi_markers=bool(cfg.get("use_wsi_markers", True)),
        use_index_emb=bool(cfg.get("use_wsi_index_emb", True)),
    )

    adapter_dir = weights_dir / "adapter"
    if adapter_dir.is_dir():
        from peft import PeftModel

        model.llm = PeftModel.from_pretrained(model.llm, str(adapter_dir), is_trainable=False)

    state_path = weights_dir / "vlm_state.pt"
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")

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

    tok_path = weights_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path if tok_path.is_dir() else llm_path), use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    return model, tokenizer, cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a case-level pathology report from precomputed WSI embeddings.")
    p.add_argument("--weights", type=Path, required=True, help="Release weight package directory.")
    p.add_argument("--embeddings", type=Path, nargs="+", required=True, help="One or more slide .h5 embedding files for one case.")
    p.add_argument("--feature_key", type=str, default="", help="Defaults to feature_key in weights/config.json.")
    p.add_argument("--max_vision_tokens", type=int, default=0, help="Optional cap across all WSIs; 0 disables.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--min_new_tokens", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--output_json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = None
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16

    model, tokenizer, cfg = _load_release_model(Path(args.weights), device=device, dtype=dtype)
    feature_key = str(args.feature_key or cfg.get("feature_key", "conch_v15"))

    features: list[np.ndarray] = []
    counts: list[int] = []
    kept_paths: list[str] = []
    remaining = int(args.max_vision_tokens)
    for path in args.embeddings:
        arr = _read_h5_features(Path(path), feature_key)
        if int(args.max_vision_tokens) > 0:
            if remaining <= 0:
                break
            arr = arr[:remaining]
            remaining -= int(arr.shape[0])
        if int(arr.shape[0]) == 0:
            continue
        features.append(arr)
        counts.append(int(arr.shape[0]))
        kept_paths.append(str(path))

    if not features:
        raise RuntimeError("No non-empty embedding matrices were loaded.")

    vision_np = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    vision = torch.from_numpy(vision_np).unsqueeze(0).to(device)
    vision_mask = torch.ones((1, int(vision.shape[1])), dtype=torch.bool, device=device)
    wsi_patch_counts = torch.tensor([counts], dtype=torch.long, device=device)

    target_field = str(cfg.get("report_target_field", "conclusion"))
    target_label = str(cfg.get("report_target_label", "") or "") or None
    field_label = resolve_target_field_label(target_field, target_label)
    prompt_text = resolve_prompt_text(
        str(cfg.get("prompt_style", "single")),
        target_field_name=target_field,
        target_field_label=field_label,
    )
    markers = "\n".join([f"WSI #{i + 1}" for i in range(len(counts))])
    messages = [
        {"role": "system", "content": prompt_text.strip()},
        {
            "role": "user",
            "content": "Please analyze the provided WSIs (as visual tokens) and respond strictly in the requested format.\n"
            + markers,
        },
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            vision_embeddings=vision,
            vision_attention_mask=vision_mask,
            wsi_patch_counts=wsi_patch_counts,
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=int(args.max_new_tokens),
            min_new_tokens=int(args.min_new_tokens),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            num_beams=1,
        )

    prompt_len = int(prompt_ids.shape[1])
    gen_0 = generated[0]
    if int(gen_0.numel()) >= prompt_len and torch.equal(gen_0[:prompt_len], prompt_ids[0]):
        cont = gen_0[prompt_len:]
    else:
        cont = gen_0
    report = tokenizer.decode(cont.tolist(), skip_special_tokens=True).replace("<|endoftext|>", "").strip()

    payload = {
        "report": report,
        "slide_embeddings": kept_paths,
        "wsi_patch_counts": counts,
        "feature_key": feature_key,
        "target_field_label": field_label,
    }
    print(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
