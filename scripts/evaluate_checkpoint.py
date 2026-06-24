from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathosynvlm.alignment_dataset import create_train_val_dataloaders as create_3datasets_train_val_dataloaders
from pathosynvlm.alignment_dataset import load_tokenizer
from pathosynvlm.histai_dataset import create_train_val_dataloaders as create_histai_train_val_dataloaders
from pathosynvlm.metrics import summarize_field_accuracy
from pathosynvlm.model import VLM_MVP


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _extract_prompt_and_refs(
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    bsz = int(input_ids.shape[0])
    device = input_ids.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prompt_ids_list: list[torch.Tensor] = []
    refs: list[str] = []
    prompts: list[str] = []

    for i in range(bsz):
        valid_len = int(attention_mask[i].sum().item())
        ids_i = input_ids[i, :valid_len]
        labels_i = labels[i, :valid_len]

        target_ids = labels_i[labels_i != -100]
        refs.append(tokenizer.decode(target_ids.tolist(), skip_special_tokens=True).strip())

        tgt_pos = (labels_i != -100).nonzero(as_tuple=False)
        prompt_end = int(tgt_pos[0].item()) if tgt_pos.numel() > 0 else valid_len
        p_ids = ids_i[:prompt_end]
        prompts.append(tokenizer.decode(p_ids.tolist(), skip_special_tokens=True).strip())
        prompt_ids_list.append(p_ids)

    max_p = max((x.shape[0] for x in prompt_ids_list), default=1)
    out_ids = torch.full((bsz, max_p), pad_id, dtype=input_ids.dtype, device=device)
    out_mask = torch.zeros((bsz, max_p), dtype=attention_mask.dtype, device=device)

    for i, p_ids in enumerate(prompt_ids_list):
        l = int(p_ids.shape[0])
        if l > 0:
            out_ids[i, :l] = p_ids
            out_mask[i, :l] = 1

    return out_ids, out_mask, refs, prompts


@torch.no_grad()
def _generate_predictions_from_batch(
    *,
    model,
    tokenizer,
    batch: dict[str, Any],
    gen_max_new_tokens: int,
    gen_min_new_tokens: int,
    gen_do_sample: bool,
    gen_temperature: float,
    gen_top_p: float,
) -> tuple[list[str], list[str], list[str]]:
    prompt_ids, prompt_mask, refs, prompts = _extract_prompt_and_refs(
        tokenizer=tokenizer,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )

    gen_ids = model.generate(
        vision_embeddings=batch["vision_embeddings"],
        vision_attention_mask=batch["vision_attention_mask"],
        wsi_patch_counts=batch["wsi_patch_counts"],
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=int(gen_max_new_tokens),
        min_new_tokens=int(gen_min_new_tokens),
        do_sample=bool(gen_do_sample),
        temperature=float(gen_temperature),
        top_p=float(gen_top_p),
        num_beams=1,
    )

    preds: list[str] = []
    for i in range(int(gen_ids.shape[0])):
        gen_i = gen_ids[i]
        pm = prompt_mask[i].bool()
        p_1d = prompt_ids[i, pm]
        lp = int(p_1d.numel())

        if int(gen_i.numel()) >= lp and torch.equal(gen_i[:lp], p_1d):
            cont = gen_i[lp:]
        else:
            cont = gen_i

        pred = tokenizer.decode(cont.tolist(), skip_special_tokens=True).strip()
        pred = pred.replace("<|endoftext|>", "").strip()
        preds.append(pred)

    return preds, refs, prompts


def _compute_text_metrics(
    *,
    preds: list[str],
    refs: list[str],
    rouge_metric,
    meteor_metric,
    bleu_metric,
    bertscore_metric,
    bertscore_model_type: str,
) -> dict[str, float]:
    if not preds:
        return {
            "rougeL": 0.0,
            "meteor": 0.0,
            "bleu4": 0.0,
            "bertscore_f1": 0.0,
        }

    rouge = rouge_metric.compute(
        predictions=preds,
        references=refs,
        rouge_types=["rougeL"],
        use_stemmer=True,
    )
    meteor = meteor_metric.compute(predictions=preds, references=refs)
    bleu = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
    bs = bertscore_metric.compute(
        predictions=preds,
        references=refs,
        lang="en",
        model_type=str(bertscore_model_type),
        rescale_with_baseline=True,
        use_fast_tokenizer=True,
    )

    return {
        "rougeL": float(rouge.get("rougeL", 0.0)),
        "meteor": float(meteor.get("meteor", 0.0)),
        "bleu4": float(bleu.get("score", 0.0)),
        "bertscore_f1": float(sum(bs["f1"]) / max(1, len(bs["f1"]))),
    }


def _safe_batch_item(batch: dict[str, Any], key: str, idx: int, default: str = "") -> str:
    value = batch.get(key)
    if isinstance(value, (list, tuple)) and 0 <= idx < len(value):
        item = value[idx]
        if isinstance(item, (list, tuple)):
            return " || ".join(str(x) for x in item)
        return str(item)
    return default


@torch.no_grad()
def evaluate_split(
    *,
    model,
    dl: DataLoader,
    device: torch.device,
    tokenizer,
    max_batches: int,
    gen_max_new_tokens: int,
    gen_min_new_tokens: int,
    gen_do_sample: bool,
    gen_temperature: float,
    gen_top_p: float,
    rouge_metric,
    meteor_metric,
    bleu_metric,
    bertscore_metric,
    bertscore_model_type: str,
    third_field_name: str,
    third_field_label: str | None,
    certainty_percent_threshold: float,
    sample_count: int,
    prediction_jsonl_path: Path | None,
    prediction_group_name: str,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    model.eval()

    losses: list[float] = []
    all_preds: list[str] = []
    all_refs: list[str] = []
    sample_rows: list[dict[str, str]] = []
    prediction_handle = None
    if prediction_jsonl_path is not None:
        prediction_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_handle = prediction_jsonl_path.open("a", encoding="utf-8")

    try:
        for b_idx, batch in enumerate(dl):
            if int(max_batches) > 0 and b_idx >= int(max_batches):
                break

            batch = to_device(batch, device)

            out = model(
                vision_embeddings=batch["vision_embeddings"],
                vision_attention_mask=batch["vision_attention_mask"],
                wsi_patch_counts=batch["wsi_patch_counts"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            if out.loss is not None:
                losses.append(float(out.loss.detach().float().cpu().item()))

            preds, refs, prompts = _generate_predictions_from_batch(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                gen_max_new_tokens=int(gen_max_new_tokens),
                gen_min_new_tokens=int(gen_min_new_tokens),
                gen_do_sample=bool(gen_do_sample),
                gen_temperature=float(gen_temperature),
                gen_top_p=float(gen_top_p),
            )

            all_preds.extend(preds)
            all_refs.extend(refs)

            for i in range(len(preds)):
                case_value = _safe_batch_item(batch, "case_id", i)
                if not case_value:
                    case_value = _safe_batch_item(batch, "case_mapping", i, default=f"batch{b_idx}_i{i}")
                row = {
                    "eval_group": prediction_group_name,
                    "case": case_value,
                    "slide_path": _safe_batch_item(batch, "slide_paths", i),
                    "prompt": prompts[i],
                    "prediction": preds[i],
                    "reference": refs[i],
                }
                if prediction_handle is not None:
                    prediction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    prediction_handle.flush()
                if len(sample_rows) < int(sample_count):
                    sample_rows.append(row)
    finally:
        if prediction_handle is not None:
            prediction_handle.close()

    metrics = {
        "val_loss": float(sum(losses) / max(1, len(losses))),
        "val_n_samples": int(len(all_preds)),
    }
    metrics.update(
        _compute_text_metrics(
            preds=all_preds,
            refs=all_refs,
            rouge_metric=rouge_metric,
            meteor_metric=meteor_metric,
            bleu_metric=bleu_metric,
            bertscore_metric=bertscore_metric,
            bertscore_model_type=bertscore_model_type,
        )
    )
    field_summary = summarize_field_accuracy(
        predicted_texts=all_preds,
        reference_texts=all_refs,
        certainty_percent_threshold=float(certainty_percent_threshold),
        third_field_name=str(third_field_name),
        third_field_label=third_field_label,
        rouge_metric=rouge_metric,
        meteor_metric=meteor_metric,
        bleu_metric=bleu_metric,
    )
    for key, value in field_summary.items():
        if isinstance(value, (int, float)) or value is None:
            metrics[f"field_{key}"] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    metrics[f"field_{key}_{sub_key}"] = float(sub_value)
    return metrics, sample_rows


def _domain_from_paths(paths: tuple[Path, ...]) -> str:
    s = " || ".join(str(p) for p in paths)
    if "/REG_dataset/" in s:
        return "reg"
    if "/HistGen-" in s:
        return "histgen"
    if "/PathText/" in s:
        return "pathtext"
    return "unknown"


def _resolve_best_step(run_dir: Path, ckpt_step: int) -> int:
    if int(ckpt_step) >= 0:
        return int(ckpt_step)
    best_path = run_dir / "best_checkpoint_summary.json"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best_checkpoint_summary.json in {run_dir}")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    step = int(best.get("best_step", -1))
    if step < 0:
        raise ValueError(f"Invalid best_step in {best_path}")
    return step


def _load_train_args(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "train_args.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing train_args.json in {run_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_saved_llm_dir(run_dir: Path, step: int) -> Path:
    candidates = [
        run_dir / f"llm_step_{int(step)}",
        run_dir / f"lora_step_{int(step)}",
        run_dir / "best_llm",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"Missing saved LLM directory for step={step} in {run_dir}")


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    out: list[str] = []
    for tok in str(value).replace("+", ",").split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _normalize_case_token(value: str) -> str:
    s = str(value or "").strip().lower().replace("\\", "/")
    s = re.sub(r"/+", "/", s)
    s = re.sub(r"case_0*([0-9]+)", lambda m: f"case_{int(m.group(1))}", s)
    return s


def _extract_case_value_from_row(row: dict[str, Any], preferred_field: str) -> str:
    if preferred_field and str(row.get(preferred_field) or "").strip():
        return str(row.get(preferred_field) or "").strip()
    for k in ("case_id", "case_mapping", "id"):
        v = str(row.get(k) or "").strip()
        if v:
            return v
    return ""


def _load_case_filter_set(case_files: list[Path], preferred_field: str) -> tuple[set[str], dict[str, Any]]:
    case_set: set[str] = set()
    stats: dict[str, Any] = {
        "files": [],
        "rows_seen": 0,
        "rows_used": 0,
        "case_count": 0,
    }

    for p in case_files:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"case filter file not found: {p}")

        file_info = {"path": str(p), "rows_seen": 0, "rows_used": 0}
        suffix = p.suffix.lower()

        if suffix == ".jsonl":
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    file_info["rows_seen"] += 1
                    stats["rows_seen"] = int(stats["rows_seen"]) + 1
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        continue
                    val = _extract_case_value_from_row(row, preferred_field)
                    if not val:
                        continue
                    case_set.add(_normalize_case_token(val))
                    file_info["rows_used"] += 1
                    stats["rows_used"] = int(stats["rows_used"]) + 1

        elif suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            rows: list[Any]
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict) and isinstance(data.get("cases"), list):
                rows = data.get("cases", [])
            else:
                raise ValueError(f"Unsupported case json structure in {p}")

            for row in rows:
                file_info["rows_seen"] += 1
                stats["rows_seen"] = int(stats["rows_seen"]) + 1
                if isinstance(row, dict):
                    val = _extract_case_value_from_row(row, preferred_field)
                else:
                    val = str(row).strip()
                if not val:
                    continue
                case_set.add(_normalize_case_token(val))
                file_info["rows_used"] += 1
                stats["rows_used"] = int(stats["rows_used"]) + 1

        else:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    v = str(line).strip()
                    if not v:
                        continue
                    file_info["rows_seen"] += 1
                    stats["rows_seen"] = int(stats["rows_seen"]) + 1
                    case_set.add(_normalize_case_token(v))
                    file_info["rows_used"] += 1
                    stats["rows_used"] = int(stats["rows_used"]) + 1

        stats["files"].append(file_info)

    stats["case_count"] = int(len(case_set))
    return case_set, stats


def _filter_positions_by_cases(
    *,
    positions: list[int],
    case_getter: Callable[[int], str],
    case_filter: set[str],
) -> list[int]:
    if not case_filter:
        return list(positions)
    out: list[int] = []
    for pos in positions:
        case_val = _normalize_case_token(case_getter(pos))
        if case_val in case_filter:
            out.append(int(pos))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stage-2 checkpoint on Stage-1 (3datasets) or HistAI validation split.")
    repo_root = Path(__file__).resolve().parents[1]
    p.add_argument("--finetune_run_dir", type=Path, required=True)
    p.add_argument("--checkpoint_step", type=int, default=-1, help="-1: auto use best_step from best_checkpoint_summary.json")

    p.add_argument("--dataset_scope", type=str, default="3datasets", choices=["3datasets", "histai"])
    p.add_argument("--dataset_selection", type=str, default="all", help="3datasets selection: all/reg/histgen/pathtext/no_reg/...")
    p.add_argument("--eval_groups", type=str, default="", help="Comma-separated groups to report (scope-dependent)")

    p.add_argument(
        "--alignment_metadata_json",
        type=Path,
        default=repo_root / "data" / "stage1" / "merged_metadata_3datasets_filtered_conch_v15.json",
    )
    p.add_argument(
        "--histai_metadata_standardized_json",
        type=Path,
        default=repo_root / "data" / "histai" / "standardized_metadata_fixed_filtered_5x_512.json",
    )

    p.add_argument("--dataset_embeddings_root", type=Path, default=repo_root / "data" / "embeddings")
    p.add_argument("--feature_key", type=str, default="")
    p.add_argument("--patch_level", type=str, default="")
    p.add_argument("--val_size", type=str, default="0.2")
    p.add_argument("--split_seed", type=int, default=42)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_text_length", type=int, default=512)
    p.add_argument("--prompt_style", type=str, default="")

    p.add_argument("--gen_max_new_tokens", type=int, default=256)
    p.add_argument("--gen_min_new_tokens", type=int, default=0)
    p.add_argument("--gen_do_sample", action="store_true")
    p.add_argument("--gen_temperature", type=float, default=0.6)
    p.add_argument("--gen_top_p", type=float, default=0.95)
    p.add_argument("--certainty_percent_threshold", type=float, default=50.0)
    p.add_argument("--max_batches", type=int, default=-1)
    p.add_argument("--sample_count", type=int, default=20)
    p.add_argument("--bertscore_model_type", type=str, default="roberta-large")

    p.add_argument(
        "--case_filter_files",
        type=str,
        default="",
        help="Comma-separated files (.jsonl/.json/.txt) containing case_id/case_mapping to restrict evaluation",
    )
    p.add_argument("--case_filter_field", type=str, default="", help="Optional field name inside case filter jsonl/json rows")

    p.add_argument("--output_json", type=Path, default=None)
    p.add_argument(
        "--output_predictions_jsonl",
        type=Path,
        default=None,
        help="Optional streamed per-case prediction export; useful when long eval jobs hit time limit before summary JSON is written.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.finetune_run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"finetune_run_dir not found: {run_dir}")

    train_args = _load_train_args(run_dir)
    step = _resolve_best_step(run_dir, int(args.checkpoint_step))

    state_path = run_dir / f"trainer_state_step_{step}.pt"
    llm_dir = _resolve_saved_llm_dir(run_dir, step)
    if not state_path.exists():
        raise FileNotFoundError(f"Missing trainer state: {state_path}")

    llm = str(train_args.get("llm", "Qwen/Qwen2.5-3B-Instruct"))
    use_lora = bool(train_args.get("use_lora", True))
    vision_dim = int(train_args.get("vision_dim", 768))
    feature_key = str(args.feature_key or train_args.get("feature_key", "conch_v15"))
    patch_level = str(args.patch_level or train_args.get("patch_level", "5x_512"))
    prompt_style = str(args.prompt_style or train_args.get("prompt_style", "single"))
    report_target_field = str(train_args.get("report_target_field", "conclusion"))
    report_target_label = str(train_args.get("report_target_label", "")).strip() or None
    use_wsi_markers = bool(train_args.get("use_wsi_markers", True))
    use_wsi_index_emb = bool(train_args.get("use_wsi_index_emb", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device.type == "cuda" else None

    tokenizer_source = run_dir / "tokenizer"
    if not tokenizer_source.is_dir():
        tokenizer_source = Path(llm)
    print(f"[info] loading tokenizer: {tokenizer_source}")
    tokenizer = load_tokenizer(str(tokenizer_source), trust_remote_code=True, use_fast=True)

    group_positions: dict[str, list[int]] = {}
    group_base_counts: dict[str, int] = {}
    val_domain_counts: dict[str, int] = {}

    if str(args.dataset_scope) == "3datasets":
        print("[info] building 3datasets dataloaders with configured dataset_selection...")
        _train_dl, val_dl = create_3datasets_train_val_dataloaders(
            metadata_json=Path(args.alignment_metadata_json),
            dataset_embeddings_root=Path(args.dataset_embeddings_root),
            tokenizer=tokenizer,
            feature_key=feature_key,
            patch_level=patch_level,
            datasets=str(args.dataset_selection),
            prompt_style=prompt_style,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            max_text_length=int(args.max_text_length),
            missing_policy="skip",
            probe_h5_on_init=False,
            enforce_readable_h5=True,
            val_size=args.val_size,
            split_seed=int(args.split_seed),
            train_shuffle=False,
        )

        if not isinstance(val_dl.dataset, Subset):
            raise RuntimeError("Expected 3datasets val_dl.dataset to be torch.utils.data.Subset")
        val_subset = val_dl.dataset
        base_ds = val_subset.dataset
        subset_indices = [int(i) for i in val_subset.indices]

        domain_positions: dict[str, list[int]] = {"reg": [], "histgen": [], "pathtext": [], "unknown": []}
        for pos, base_idx in enumerate(subset_indices):
            sample = base_ds.samples[int(base_idx)]
            d = _domain_from_paths(sample.h5_paths)
            domain_positions.setdefault(d, []).append(pos)

        all_pos = list(range(len(subset_indices)))
        group_positions = {
            "all": all_pos,
            "reg": domain_positions.get("reg", []),
            "histgen": domain_positions.get("histgen", []),
            "pathtext": domain_positions.get("pathtext", []),
            "unknown": domain_positions.get("unknown", []),
            "combined_reg_histgen": sorted(domain_positions.get("reg", []) + domain_positions.get("histgen", [])),
        }

        group_base_counts = {k: int(len(v)) for k, v in group_positions.items()}
        val_domain_counts = {
            "total": int(len(subset_indices)),
            "reg": int(len(domain_positions.get("reg", []))),
            "histgen": int(len(domain_positions.get("histgen", []))),
            "pathtext": int(len(domain_positions.get("pathtext", []))),
            "unknown": int(len(domain_positions.get("unknown", []))),
        }

        def case_getter(pos: int) -> str:
            base_idx = subset_indices[int(pos)]
            return str(base_ds.samples[int(base_idx)].case_id)

        def make_group_loader(positions: list[int]) -> DataLoader:
            ds = Subset(val_subset, [int(x) for x in positions])
            return DataLoader(
                ds,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                collate_fn=val_dl.collate_fn,
            )

    else:
        print("[info] building HistAI dataloaders...")
        _train_dl, val_dl = create_histai_train_val_dataloaders(
            metadata_standardized_json=Path(args.histai_metadata_standardized_json),
            dataset_embeddings_root=Path(args.dataset_embeddings_root),
            tokenizer=tokenizer,
            feature_key=feature_key,
            patch_level=patch_level,
            conclusion_field=report_target_field,
            target_field_name=report_target_field,
            target_field_label=report_target_label,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            max_text_length=int(args.max_text_length),
            prompt_style=prompt_style,
            missing_policy="skip",
            probe_h5_on_init=False,
            val_size=args.val_size,
            split_seed=int(args.split_seed),
            train_shuffle=False,
        )

        val_ds = val_dl.dataset
        all_pos = list(range(len(val_ds)))
        group_positions = {"all": all_pos}
        group_base_counts = {"all": int(len(all_pos))}
        val_domain_counts = {"total": int(len(all_pos)), "all": int(len(all_pos))}

        def case_getter(pos: int) -> str:
            if hasattr(val_ds, "indices") and hasattr(val_ds, "records"):
                rec_idx = int(val_ds.indices[int(pos)])  # type: ignore[index]
                rec = val_ds.records[rec_idx]  # type: ignore[index]
                return str(rec.get("case_mapping") or "")
            item = val_ds[int(pos)]
            return str(getattr(item, "case_mapping", ""))

        def make_group_loader(positions: list[int]) -> DataLoader:
            ds = Subset(val_ds, [int(x) for x in positions])
            return DataLoader(
                ds,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                collate_fn=val_dl.collate_fn,
            )

    case_files = [Path(x) for x in _split_csv(args.case_filter_files)]
    case_filter: set[str] = set()
    case_filter_stats: dict[str, Any] | None = None
    if case_files:
        case_filter, case_filter_stats = _load_case_filter_set(case_files, str(args.case_filter_field))
        print(
            f"[case_filter] loaded {len(case_filter)} unique cases from {len(case_files)} files; "
            f"rows_seen={case_filter_stats['rows_seen']} rows_used={case_filter_stats['rows_used']}"
        )

        filtered_group_positions: dict[str, list[int]] = {}
        for g, pos in group_positions.items():
            filtered_group_positions[g] = _filter_positions_by_cases(
                positions=pos,
                case_getter=case_getter,
                case_filter=case_filter,
            )
        group_positions = filtered_group_positions

    eval_groups = _split_csv(args.eval_groups)
    if not eval_groups:
        if str(args.dataset_scope) == "3datasets":
            eval_groups = ["combined_reg_histgen", "reg", "histgen"]
        else:
            eval_groups = ["all"]

    missing_groups = [g for g in eval_groups if g not in group_positions]
    if missing_groups:
        raise ValueError(f"Unknown eval_groups={missing_groups}; available={sorted(group_positions.keys())}")

    from peft import PeftModel
    import evaluate

    print("[info] building model and loading checkpoint...")
    model = VLM_MVP(
        llm_name_or_path=(str(llm_dir) if not use_lora else llm),
        vision_dim=vision_dim,
        feature_key=feature_key,
        torch_dtype=torch_dtype,
        attn_implementation=None,
        use_wsi_markers=use_wsi_markers,
        use_index_emb=use_wsi_index_emb,
    )
    if use_lora:
        model.llm = PeftModel.from_pretrained(model.llm, str(llm_dir), is_trainable=False)

    state = torch.load(state_path, map_location="cpu")
    llm_state = state.get("llm_state_dict") if isinstance(state, dict) else None
    if isinstance(llm_state, dict):
        missing, unexpected = model.llm.load_state_dict(llm_state, strict=False)
        print(
            "[info] loaded full llm_state_dict from trainer state "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    elif bool(train_args.get("unfreeze_llm_base", False)):
        print(
            "[warn] train_args indicate unfreeze_llm_base=true, but this checkpoint has no "
            "llm_state_dict. Exact reruns may require a full-state or merged-model export."
        )
    aligner_sd = state.get("aligner")
    if isinstance(aligner_sd, dict):
        model.aligner.load_state_dict(aligner_sd, strict=True)

    if bool(getattr(model, "use_wsi_markers", False)):
        if bool(getattr(model, "use_index_emb", False)):
            idx_sd = state.get("wsi_index_emb")
            if isinstance(idx_sd, dict) and hasattr(model, "wsi_index_emb"):
                model.wsi_index_emb.load_state_dict(idx_sd, strict=True)
        wsi_sep = state.get("wsi_sep")
        if torch.is_tensor(wsi_sep) and hasattr(model, "wsi_sep"):
            model.wsi_sep.data.copy_(wsi_sep.to(model.wsi_sep.dtype))

    model = model.to(device)
    model.eval()

    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    bleu_metric = evaluate.load("sacrebleu")
    bertscore_metric = evaluate.load("bertscore")

    output_json = args.output_json
    if output_json is None:
        suffix = f"{args.dataset_scope}_{args.dataset_selection}".replace("/", "_").replace(",", "_")
        output_json = run_dir / f"stage2_baseline_eval_{suffix}_step{step}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    output_predictions_jsonl = args.output_predictions_jsonl
    if output_predictions_jsonl is None:
        output_predictions_jsonl = output_json.with_name(f"{output_json.stem}_predictions.jsonl")
    output_predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_predictions_jsonl.exists():
        output_predictions_jsonl.unlink()
    print(f"[info] streaming predictions jsonl to: {output_predictions_jsonl}")

    results: dict[str, Any] = {}
    for name in eval_groups:
        pos = group_positions[name]
        dl = make_group_loader(pos)
        metrics, samples = evaluate_split(
            model=model,
            dl=dl,
            device=device,
            tokenizer=tokenizer,
            max_batches=int(args.max_batches),
            gen_max_new_tokens=int(args.gen_max_new_tokens),
            gen_min_new_tokens=int(args.gen_min_new_tokens),
            gen_do_sample=bool(args.gen_do_sample),
            gen_temperature=float(args.gen_temperature),
            gen_top_p=float(args.gen_top_p),
            rouge_metric=rouge_metric,
            meteor_metric=meteor_metric,
            bleu_metric=bleu_metric,
            bertscore_metric=bertscore_metric,
            bertscore_model_type=str(args.bertscore_model_type),
            third_field_name=str(report_target_field),
            third_field_label=report_target_label,
            certainty_percent_threshold=float(args.certainty_percent_threshold),
            sample_count=int(args.sample_count),
            prediction_jsonl_path=output_predictions_jsonl,
            prediction_group_name=name,
        )
        results[name] = {
            "n_subset_rows_before_case_filter": int(group_base_counts.get(name, len(pos))),
            "n_subset_rows": int(len(pos)),
            "metrics": metrics,
            "sample_rows": samples,
        }
        print(
            f"[{name}] n={metrics['val_n_samples']} val_loss={metrics['val_loss']:.4f} "
            f"rougeL={metrics['rougeL']:.4f} meteor={metrics['meteor']:.4f} "
            f"bleu4={metrics['bleu4']:.4f} bertscore_f1={metrics['bertscore_f1']:.4f}"
        )

    out = {
        "config": {
            "finetune_run_dir": str(run_dir),
            "checkpoint_step": int(step),
            "state_path": str(state_path),
            "llm_dir": str(llm_dir),
            "use_lora": bool(use_lora),
            "dataset_scope": str(args.dataset_scope),
            "dataset_selection": str(args.dataset_selection),
            "eval_groups": eval_groups,
            "alignment_metadata_json": str(args.alignment_metadata_json),
            "histai_metadata_standardized_json": str(args.histai_metadata_standardized_json),
            "dataset_embeddings_root": str(args.dataset_embeddings_root),
            "feature_key": feature_key,
            "patch_level": patch_level,
            "val_size": str(args.val_size),
            "split_seed": int(args.split_seed),
            "prompt_style": prompt_style,
            "report_target_field": str(report_target_field),
            "report_target_label": (str(report_target_label) if report_target_label is not None else None),
            "gen_max_new_tokens": int(args.gen_max_new_tokens),
            "gen_min_new_tokens": int(args.gen_min_new_tokens),
            "gen_do_sample": bool(args.gen_do_sample),
            "gen_temperature": float(args.gen_temperature),
            "gen_top_p": float(args.gen_top_p),
            "certainty_percent_threshold": float(args.certainty_percent_threshold),
            "bertscore_model_type": str(args.bertscore_model_type),
            "max_batches": int(args.max_batches),
            "llm": llm,
            "vision_dim": int(vision_dim),
            "use_wsi_markers": bool(use_wsi_markers),
            "use_wsi_index_emb": bool(use_wsi_index_emb),
            "case_filter_files": [str(x) for x in case_files],
            "case_filter_field": str(args.case_filter_field),
        },
        "val_domain_counts": val_domain_counts,
        "case_filter": case_filter_stats,
        "results": results,
    }

    output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[done] wrote: {output_json}")


if __name__ == "__main__":
    main()
