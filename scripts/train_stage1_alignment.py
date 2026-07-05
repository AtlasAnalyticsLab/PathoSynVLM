from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathosynvlm.alignment_dataset import create_train_val_dataloaders, load_tokenizer
from pathosynvlm.alignment_model import build_alignment_model, freeze_for_alignment_only


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _unwrap_subset_dataset(ds: Any) -> Any:
    cur = ds
    while isinstance(cur, Subset):
        cur = cur.dataset
    return cur


def build_scheduler(
    *,
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    name = str(scheduler_name).strip().lower()
    if name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_steps),
            num_training_steps=int(total_steps),
        )
    if name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_steps),
            num_training_steps=int(total_steps),
        )
    if name == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_steps))
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def save_alignment_checkpoint(
    *,
    output_dir: Path,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    tokenizer,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    filename: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / (filename if filename else f"alignment_epoch_{epoch:03d}_step_{global_step}.pt")

    m = model
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "llm_name_or_path": str(args.llm),
        "vision_dim": int(args.vision_dim),
        "feature_key": str(args.feature_key),
        "patch_level": str(args.patch_level),
        "aligner_state_dict": m.aligner.state_dict() if hasattr(m, "aligner") else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
    }
    torch.save(payload, ckpt_path)

    tok_dir = output_dir / "tokenizer"
    if not tok_dir.exists():
        tokenizer.save_pretrained(tok_dir)

    return ckpt_path


def save_best_aligner_weights(
    *,
    output_dir: Path,
    model,
    epoch: int,
    global_step: int,
    val_metrics: Dict[str, float],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_aligner_weights.pt"
    m = model
    aligner_state = (
        {k: v.detach().cpu() for k, v in m.aligner.state_dict().items()}
        if hasattr(m, "aligner")
        else None
    )
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "aligner_state_dict": aligner_state,
    }
    torch.save(payload, best_path)
    return best_path


def _extract_prompt_and_refs(
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
):
    """
    Returns prompt-only ids/masks + decoded refs/prompts.
    """
    bsz = int(input_ids.shape[0])
    device = input_ids.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prompt_ids_list = []
    prompt_mask_list = []
    refs: List[str] = []
    prompts: List[str] = []

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
        prompt_mask_list.append(torch.ones(p_ids.shape[0], device=device, dtype=attention_mask.dtype))

    max_p = max((x.shape[0] for x in prompt_ids_list), default=1)
    out_ids = torch.full((bsz, max_p), pad_id, device=device, dtype=input_ids.dtype)
    out_mask = torch.zeros((bsz, max_p), device=device, dtype=attention_mask.dtype)

    for i in range(bsz):
        l = prompt_ids_list[i].shape[0]
        if l > 0:
            out_ids[i, :l] = prompt_ids_list[i]
            out_mask[i, :l] = 1

    return out_ids, out_mask, refs, prompts


@torch.no_grad()
def _generate_predictions_from_batch(
    *,
    model,
    tokenizer,
    batch: Dict[str, Any],
    gen_max_new_tokens: int,
    gen_do_sample: bool,
    gen_temperature: float,
    gen_top_p: float,
) -> tuple[List[str], List[str], List[str]]:
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
        do_sample=bool(gen_do_sample),
        temperature=float(gen_temperature),
        top_p=float(gen_top_p),
        num_beams=1,
    )

    preds: List[str] = []
    for i in range(gen_ids.shape[0]):
        gen_i = gen_ids[i]
        pm = prompt_mask[i].bool()
        p_1d = prompt_ids[i, pm]
        lp = int(p_1d.numel())

        if gen_i.numel() >= lp and torch.equal(gen_i[:lp], p_1d):
            cont = gen_i[lp:]
        else:
            cont = gen_i

        pred = tokenizer.decode(cont.tolist(), skip_special_tokens=True).strip()
        pred = pred.replace("<|endoftext|>", "").strip()
        preds.append(pred)

    return preds, refs, prompts


def _compute_text_metrics(
    *,
    preds: List[str],
    refs: List[str],
    rouge_metric,
    meteor_metric,
    bleu_metric,
    bertscore_metric,
    bertscore_model_type: str,
) -> Dict[str, float]:
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


def _basic_text_stats(
    *,
    preds: List[str],
    refs: List[str],
    prompts: List[str] | None = None,
) -> Dict[str, float]:
    def _stats(xs: List[str], prefix: str) -> Dict[str, float]:
        n = len(xs)
        if n <= 0:
            return {
                f"{prefix}_chars_mean": 0.0,
                f"{prefix}_chars_p95": 0.0,
                f"{prefix}_words_mean": 0.0,
                f"{prefix}_empty_rate": 0.0,
            }
        chars = sorted(len(x) for x in xs)
        words = [len(x.split()) for x in xs]
        p95_idx = min(n - 1, max(0, int(math.ceil(0.95 * n)) - 1))
        empty_n = sum(1 for x in xs if not str(x).strip())
        return {
            f"{prefix}_chars_mean": float(sum(chars) / n),
            f"{prefix}_chars_p95": float(chars[p95_idx]),
            f"{prefix}_words_mean": float(sum(words) / n),
            f"{prefix}_empty_rate": float(empty_n / n),
        }

    out: Dict[str, float] = {}
    out.update(_stats(preds, "pred"))
    out.update(_stats(refs, "ref"))
    if prompts is not None:
        out.update(_stats(prompts, "prompt"))
    return out


def _sample_rows_stats(rows: List[Dict[str, str]], prefix: str) -> Dict[str, float]:
    preds = [str(r.get("prediction", "")) for r in rows]
    refs = [str(r.get("reference", "")) for r in rows]
    prompts = [str(r.get("prompt", "")) for r in rows]
    stats = _basic_text_stats(preds=preds, refs=refs, prompts=prompts)
    out: Dict[str, float] = {f"{prefix}/n": float(len(rows))}
    for k, v in stats.items():
        out[f"{prefix}/{k}"] = float(v)
    return out


def _flatten_numeric_dict(prefix: str, d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(_flatten_numeric_dict(key, v))
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def _ensure_min_sample_rows(rows: List[Dict[str, str]], min_rows: int) -> List[Dict[str, str]]:
    """
    Ensure we have at least `min_rows` rows for visualization.
    If the source split has fewer rows, repeat existing rows with a suffix.
    """
    target = max(0, int(min_rows))
    if len(rows) >= target or target == 0 or len(rows) == 0:
        return rows[:target] if target > 0 else rows

    out = list(rows)
    dup_i = 1
    base_len = len(rows)
    while len(out) < target:
        src = rows[(len(out) - base_len) % base_len]
        r = dict(src)
        r["case_id"] = f"{r.get('case_id', 'sample')}_dup{dup_i}"
        out.append(r)
        dup_i += 1
    return out


@torch.no_grad()
def evaluate_validation(
    *,
    model,
    val_dl: DataLoader,
    device: torch.device,
    tokenizer,
    max_batches: int,
    gen_max_new_tokens: int,
    gen_do_sample: bool,
    gen_temperature: float,
    gen_top_p: float,
    rouge_metric,
    meteor_metric,
    bleu_metric,
    bertscore_metric,
    bertscore_model_type: str,
    sample_limit: int = 10,
) -> tuple[Dict[str, float], List[Dict[str, str]]]:
    model.eval()

    losses: List[float] = []
    all_preds: List[str] = []
    all_refs: List[str] = []
    all_prompts: List[str] = []
    sample_rows: List[Dict[str, str]] = []

    for b_idx, batch in enumerate(val_dl):
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
            gen_max_new_tokens=gen_max_new_tokens,
            gen_do_sample=gen_do_sample,
            gen_temperature=gen_temperature,
            gen_top_p=gen_top_p,
        )

        all_preds.extend(preds)
        all_refs.extend(refs)
        all_prompts.extend(prompts)

        for i in range(len(preds)):
            if len(sample_rows) >= int(sample_limit):
                break
            sample_rows.append(
                {
                    "case_id": str(batch.get("case_id", [f"batch{b_idx}_i{i}"])[i]),
                    "slide_path": str(batch.get("slide_paths", [""])[i]),
                    "prompt": prompts[i],
                    "prediction": preds[i],
                    "reference": refs[i],
                }
            )

    metrics = {
        "val_loss": float(sum(losses) / max(1, len(losses))),
        "val_n_samples": float(len(all_preds)),
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
    metrics.update(
        {
            f"val_{k}": float(v)
            for k, v in _basic_text_stats(
                preds=all_preds,
                refs=all_refs,
                prompts=all_prompts,
            ).items()
        }
    )

    model.train()
    return metrics, sample_rows


@torch.no_grad()
def collect_split_samples(
    *,
    split_name: str,
    model,
    dl: DataLoader,
    device: torch.device,
    tokenizer,
    n_samples: int,
    sample_seed: int,
    gen_max_new_tokens: int,
    gen_do_sample: bool,
    gen_temperature: float,
    gen_top_p: float,
) -> List[Dict[str, str]]:
    ds = getattr(dl, "dataset", None)
    collate_fn = getattr(dl, "collate_fn", None)
    if ds is None or collate_fn is None:
        return []

    n_total = len(ds)
    if n_total <= 0:
        return []

    take = min(int(n_samples), int(n_total))
    rng = random.Random(int(sample_seed))
    picked = rng.sample(range(n_total), k=take) if n_total >= take else list(range(n_total))

    rows: List[Dict[str, str]] = []
    model.eval()

    for idx in picked:
        item = ds[int(idx)]
        batch = collate_fn([item])
        batch = to_device(batch, device)

        preds, refs, prompts = _generate_predictions_from_batch(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            gen_max_new_tokens=gen_max_new_tokens,
            gen_do_sample=gen_do_sample,
            gen_temperature=gen_temperature,
            gen_top_p=gen_top_p,
        )

        rows.append(
            {
                "case_id": str(batch.get("case_id", [f"{split_name}_idx_{idx}"])[0]),
                "slide_path": str(batch.get("slide_paths", [""])[0]),
                "prompt": prompts[0] if prompts else "",
                "prediction": preds[0] if preds else "",
                "reference": refs[0] if refs else "",
            }
        )

    model.train()
    return rows


def log_samples_to_wandb(
    *,
    rows: List[Dict[str, str]],
    split_name: str,
    epoch: int,
    step: int,
    wandb_run,
) -> None:
    if wandb_run is None or not rows:
        return

    import wandb

    table = wandb.Table(columns=["case_id", "slide_path", "prompt", "prediction", "reference"])
    for r in rows:
        table.add_data(r["case_id"], r["slide_path"], r["prompt"], r["prediction"], r["reference"])

    wandb_run.log(
        {
            f"{split_name}/samples": table,
            f"{split_name}/samples_epoch_{epoch:03d}": table,
        },
        step=int(step),
    )


def save_loss_history_json(
    *,
    output_dir: Path,
    epochs: List[int],
    train_epoch_losses: List[float],
    val_losses: List[float],
) -> Path:
    rows = []
    for e, tr, va in zip(epochs, train_epoch_losses, val_losses):
        rows.append(
            {
                "epoch": int(e),
                "train_epoch_loss": float(tr),
                "val_loss": float(va),
            }
        )

    out_path = output_dir / "loss_by_epoch.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return out_path


def log_epoch_loss_curve_to_wandb(
    *,
    wandb_run,
    epochs: List[int],
    train_epoch_losses: List[float],
    val_losses: List[float],
    step: int,
) -> None:
    if wandb_run is None or not epochs:
        return

    import wandb

    wandb_run.log(
        {
            "plots/loss_by_epoch": wandb.plot.line_series(
                xs=[int(x) for x in epochs],
                ys=[
                    [float(x) for x in train_epoch_losses],
                    [float(x) for x in val_losses],
                ],
                keys=["train/epoch_loss", "val/loss"],
                title="Training vs Validation Loss by Epoch",
                xname="epoch",
            )
        },
        step=int(step),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 aligner-only training on WSI-text pairs")

    repo_root = Path(__file__).resolve().parents[1]

    parser.add_argument(
        "--metadata_json",
        type=str,
        default=str(repo_root / "data" / "stage1" / "merged_metadata_3datasets_filtered_conch_v15.json"),
        help="Filtered merged metadata JSON from merge_filter_3datasets_metadata.py",
    )
    parser.add_argument(
        "--dataset_embeddings_root",
        type=str,
        default=str(repo_root / "data" / "embeddings"),
    )
    parser.add_argument("--feature_key", type=str, default="conch_v15")
    parser.add_argument("--patch_level", type=str, default="5x_512")
    parser.add_argument(
        "--datasets",
        type=str,
        default="histgen,reg_dataset",
        help=(
            "Dataset selector: histgen,reg_dataset matches the paper default. "
            "Other optional selectors are kept for compatibility: all | reg | histgen | pathtext | no_reg | no_histgen | no_pathtext "
            "(comma separated also supported, e.g. histgen,pathtext or all,no_reg)."
        ),
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="single",
        choices=["single", "double"],
        help="Prompt variant for text instruction: single|double",
    )

    parser.add_argument("--llm", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--vision_dim", type=int, default=768)

    parser.add_argument("--output_dir", type=str, default="./runs/alignment_3datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_size", default="500")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_text_length", type=int, default=512)
    parser.add_argument("--probe_h5_on_init", action="store_true", help="Validate every candidate h5 during dataset init (slow on shared FS).")
    parser.add_argument(
        "--enforce_readable_h5",
        action="store_true",
        default=True,
        help="Drop samples without any readable/non-empty h5 feature file before creating train/val subsets.",
    )
    parser.add_argument("--no_enforce_readable_h5", dest="enforce_readable_h5", action="store_false")

    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--aligner_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--scheduler", choices=["cosine", "linear", "constant"], default="cosine")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--val_max_batches", type=int, default=-1, help="-1 means full validation set")
    parser.add_argument("--sample_count", type=int, default=10)
    parser.add_argument("--save_every_epoch", type=int, default=1)

    parser.add_argument("--gen_max_new_tokens", type=int, default=256)
    parser.add_argument("--gen_do_sample", action="store_true")
    parser.add_argument("--gen_temperature", type=float, default=0.6)
    parser.add_argument("--gen_top_p", type=float, default=0.95)

    parser.add_argument("--bertscore_model_type", type=str, default="roberta-large")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="PathoSynVLM")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_tags", type=str, default="alignment,3datasets")

    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = load_tokenizer(args.llm, trust_remote_code=bool(args.trust_remote_code), use_fast=True)

    print("Building dataloaders...")
    dataloader_build_start = time.perf_counter()
    train_dl, val_dl = create_train_val_dataloaders(
        metadata_json=Path(args.metadata_json),
        dataset_embeddings_root=Path(args.dataset_embeddings_root),
        tokenizer=tokenizer,
        feature_key=args.feature_key,
        patch_level=args.patch_level,
        datasets=args.datasets,
        prompt_style=args.prompt_style,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_text_length=int(args.max_text_length),
        probe_h5_on_init=bool(args.probe_h5_on_init),
        enforce_readable_h5=bool(args.enforce_readable_h5),
        val_size=args.val_size,
        split_seed=int(args.split_seed),
        train_shuffle=True,
    )
    print(f"Dataloader build time: {time.perf_counter() - dataloader_build_start:.1f}s")

    if len(train_dl.dataset) == 0:
        raise RuntimeError("Train dataset is empty. Build filtered metadata first and verify embeddings.")

    torch_dtype = None
    if args.precision == "bf16" and device.type == "cuda":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16" and device.type == "cuda":
        torch_dtype = torch.float16

    model = build_alignment_model(
        llm_name_or_path=args.llm,
        vision_dim=int(args.vision_dim),
        feature_key=str(args.feature_key),
        torch_dtype=torch_dtype,
        attn_implementation=None,
    ).to(device)

    freeze_stats = freeze_for_alignment_only(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Expected aligner parameters to be trainable.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(args.aligner_lr),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=float(args.weight_decay),
    )

    updates_per_epoch = math.ceil(len(train_dl) / max(1, int(args.grad_accum)))
    total_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(args.epochs) * updates_per_epoch
    warmup_steps = int(total_steps * float(args.warmup_ratio))
    if float(args.warmup_ratio) > 0.0 and warmup_steps == 0:
        warmup_steps = 1

    scheduler = build_scheduler(
        scheduler_name=args.scheduler,
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # Mandatory metrics for this strategy.
    try:
        import evaluate
    except Exception as e:
        raise RuntimeError(
            "Failed to import `evaluate` dependencies. "
            "On Compute Canada/FIR, load Arrow before activating the venv "
            "(e.g., `module load arrow/19.0.1`), then rerun."
        ) from e

    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    bleu_metric = evaluate.load("sacrebleu")
    bertscore_metric = evaluate.load("bertscore")

    wandb_run = None
    if bool(args.wandb):
        import wandb

        tags = [x.strip() for x in str(args.wandb_tags).split(",") if x.strip()]
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=(args.wandb_run_name or None),
            group=(args.wandb_group or None),
            tags=(tags if tags else None),
            config=vars(args),
        )
        wandb_run.config.update(freeze_stats, allow_val_change=True)
        wandb_run.config.update(
            {
                "updates_per_epoch": int(updates_per_epoch),
                "total_steps": int(total_steps),
                "warmup_steps": int(warmup_steps),
            },
            allow_val_change=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # Persist dataset construction stats, including invalid h5 files skipped by the loader.
    base_dataset = _unwrap_subset_dataset(train_dl.dataset)
    dataset_build_stats: Dict[str, Any] = {
        "train_size": int(len(train_dl.dataset)),
        "val_size": int(len(val_dl.dataset)),
    }
    if hasattr(base_dataset, "stats"):
        dataset_build_stats["dataset_stats"] = getattr(base_dataset, "stats")
    if hasattr(base_dataset, "invalid_h5_reason_counts"):
        dataset_build_stats["invalid_h5_reason_counts"] = getattr(base_dataset, "invalid_h5_reason_counts")
        rc = getattr(base_dataset, "invalid_h5_reason_counts") or {}
        missing_features_total = int(rc.get("missing_features_group", 0)) + int(rc.get("missing_feature_key", 0))
        dataset_build_stats["invalid_h5_missing_features_total"] = missing_features_total
        dataset_build_stats["invalid_h5_total_counted"] = int(sum(int(v) for v in rc.values()))

    dataset_stats_path = output_dir / "dataset_build_stats.json"
    dataset_stats_path.write_text(json.dumps(dataset_build_stats, indent=2), encoding="utf-8")

    invalid_h5_paths = list(getattr(base_dataset, "invalid_h5_paths", []))
    if invalid_h5_paths:
        invalid_h5_path = output_dir / "skipped_invalid_h5_files.txt"
        invalid_h5_path.write_text("\n".join(invalid_h5_paths) + "\n", encoding="utf-8")
        print(f"Skipped invalid h5 files: {len(invalid_h5_paths)} (saved: {invalid_h5_path})")
    print(f"Actual samples after h5 filtering -> train: {len(train_dl.dataset)}, val: {len(val_dl.dataset)}")
    if "invalid_h5_missing_features_total" in dataset_build_stats:
        print(
            "Invalid h5 missing-features count:",
            int(dataset_build_stats.get("invalid_h5_missing_features_total", 0)),
            "| total invalid counted:",
            int(dataset_build_stats.get("invalid_h5_total_counted", 0)),
        )
    print(f"Dataset build stats saved: {dataset_stats_path}")

    if wandb_run is not None:
        startup_log = {
            "setup/train_size": float(len(train_dl.dataset)),
            "setup/val_size": float(len(val_dl.dataset)),
            "setup/updates_per_epoch": float(updates_per_epoch),
            "setup/total_steps": float(total_steps),
            "setup/warmup_steps": float(warmup_steps),
            "model/trainable_params": float(freeze_stats.get("trainable_params", 0)),
            "model/total_params": float(freeze_stats.get("total_params", 0)),
            "model/trainable_pct": float(freeze_stats.get("trainable_pct", 0.0)),
        }
        startup_log.update(_flatten_numeric_dict("data", dataset_build_stats))
        wandb_run.log(startup_log, step=0)

    use_fp16 = (args.precision == "fp16" and device.type == "cuda")
    use_bf16 = (args.precision == "bf16" and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    except (AttributeError, TypeError):
        # Backward compatibility for older torch versions.
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    best_val_loss = float("inf")
    best_epoch = 0
    best_global_step = 0
    best_val_metrics: Dict[str, float] = {}
    best_ckpt_path: Path | None = None
    best_aligner_path: Path | None = None
    epoch_history: List[int] = []
    train_epoch_loss_history: List[float] = []
    val_loss_history: List[float] = []

    print("Train dataset size:", len(train_dl.dataset))
    print("Val dataset size:", len(val_dl.dataset))
    print("Freeze stats:", freeze_stats)
    print("updates_per_epoch=", updates_per_epoch, "total_steps=", total_steps, "warmup_steps=", warmup_steps)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_losses: List[float] = []
        epoch_started = time.perf_counter()
        samples_seen_epoch = 0
        vision_tokens_epoch = 0
        text_tokens_epoch = 0
        target_tokens_epoch = 0
        wsi_seen_epoch = 0

        log_window_started = time.perf_counter()
        window_updates = 0
        window_loss_sum = 0.0
        window_samples = 0
        window_vision_tokens = 0
        window_text_tokens = 0
        window_target_tokens = 0
        window_wsi_count = 0
        window_grad_norm_sum = 0.0
        window_grad_norm_count = 0

        pbar = tqdm(total=len(train_dl), desc=f"epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        for step_in_epoch, batch in enumerate(train_dl, start=1):
            if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
                break

            batch = to_device(batch, device)

            bsz = int(batch["vision_embeddings"].shape[0]) if "vision_embeddings" in batch else 0
            vision_tokens = int(batch["vision_attention_mask"].sum().item()) if "vision_attention_mask" in batch else 0
            text_tokens = int(batch["attention_mask"].sum().item()) if "attention_mask" in batch else 0
            target_tokens = int((batch["labels"] != -100).sum().item()) if "labels" in batch else 0
            wsi_count = int((batch["wsi_patch_counts"] > 0).sum().item()) if "wsi_patch_counts" in batch else bsz

            samples_seen_epoch += bsz
            vision_tokens_epoch += vision_tokens
            text_tokens_epoch += text_tokens
            target_tokens_epoch += target_tokens
            wsi_seen_epoch += wsi_count

            window_samples += bsz
            window_vision_tokens += vision_tokens
            window_text_tokens += text_tokens
            window_target_tokens += target_tokens
            window_wsi_count += wsi_count

            autocast_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
            if autocast_dtype is None:
                autocast_ctx = nullcontext()
            else:
                try:
                    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
                except (AttributeError, TypeError):
                    # Backward compatibility for older torch versions.
                    autocast_ctx = torch.cuda.amp.autocast(enabled=True, dtype=autocast_dtype)

            with autocast_ctx:
                out = model(
                    vision_embeddings=batch["vision_embeddings"],
                    vision_attention_mask=batch["vision_attention_mask"],
                    wsi_patch_counts=batch["wsi_patch_counts"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                if out.loss is None:
                    raise RuntimeError("Model returned no loss.")
                raw_loss = out.loss
                loss = raw_loss / max(1, int(args.grad_accum))

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            do_update = (step_in_epoch % int(args.grad_accum) == 0) or (step_in_epoch == len(train_dl))
            if do_update:
                grad_norm_value: float | None = None
                if float(args.clip_grad_norm) > 0:
                    if use_fp16:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(args.clip_grad_norm))
                    grad_norm_value = float(grad_norm.detach().float().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)

                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                loss_value = float(raw_loss.detach().float().cpu().item())
                epoch_losses.append(loss_value)
                window_updates += 1
                window_loss_sum += loss_value
                if grad_norm_value is not None and math.isfinite(grad_norm_value):
                    window_grad_norm_sum += grad_norm_value
                    window_grad_norm_count += 1

                if global_step % int(args.log_every) == 0:
                    lr = float(scheduler.get_last_lr()[0])
                    print(f"[epoch {epoch} step {global_step}] train_loss={loss_value:.4f} lr={lr:.3e}")
                    if wandb_run is not None:
                        elapsed = max(1e-9, time.perf_counter() - log_window_started)
                        avg_loss = float(window_loss_sum / max(1, window_updates))
                        log_data: Dict[str, float] = {
                            "train/loss": loss_value,
                            "train/loss_avg_window": avg_loss,
                            "train/lr": lr,
                            "train/step": float(global_step),
                            "train/epoch_progress": float(epoch - 1 + step_in_epoch / max(1, len(train_dl))),
                            "train/window_updates": float(window_updates),
                            "train/window_samples": float(window_samples),
                            "train/window_vision_tokens": float(window_vision_tokens),
                            "train/window_text_tokens": float(window_text_tokens),
                            "train/window_target_tokens": float(window_target_tokens),
                            "train/window_wsi_count": float(window_wsi_count),
                            "train/samples_per_sec": float(window_samples / elapsed),
                            "train/vision_tokens_per_sec": float(window_vision_tokens / elapsed),
                            "train/text_tokens_per_sec": float(window_text_tokens / elapsed),
                            "train/target_tokens_per_sec": float(window_target_tokens / elapsed),
                            "train/updates_per_sec": float(window_updates / elapsed),
                            "train/avg_vision_tokens_per_sample_window": float(window_vision_tokens / max(1, window_samples)),
                            "train/avg_text_tokens_per_sample_window": float(window_text_tokens / max(1, window_samples)),
                            "train/avg_target_tokens_per_sample_window": float(window_target_tokens / max(1, window_samples)),
                            "train/avg_wsi_per_sample_window": float(window_wsi_count / max(1, window_samples)),
                        }
                        if window_grad_norm_count > 0:
                            log_data["train/grad_norm_avg_window"] = float(window_grad_norm_sum / window_grad_norm_count)
                        if grad_norm_value is not None and math.isfinite(grad_norm_value):
                            log_data["train/grad_norm"] = float(grad_norm_value)
                        if device.type == "cuda":
                            log_data["train/gpu_mem_alloc_mb"] = float(torch.cuda.memory_allocated(device) / (1024 ** 2))
                            log_data["train/gpu_mem_reserved_mb"] = float(torch.cuda.memory_reserved(device) / (1024 ** 2))
                            log_data["train/gpu_max_mem_alloc_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

                        wandb_run.log(
                            log_data,
                            step=global_step,
                        )
                        log_window_started = time.perf_counter()
                        window_updates = 0
                        window_loss_sum = 0.0
                        window_samples = 0
                        window_vision_tokens = 0
                        window_text_tokens = 0
                        window_target_tokens = 0
                        window_wsi_count = 0
                        window_grad_norm_sum = 0.0
                        window_grad_norm_count = 0

            pbar.set_postfix({"loss": f"{float(raw_loss.detach().cpu().item()):.4f}"})
            pbar.update(1)

        pbar.close()

        train_loss_epoch = float(sum(epoch_losses) / max(1, len(epoch_losses)))

        sample_count_for_logging = max(10, int(args.sample_count))

        val_metrics, val_rows = evaluate_validation(
            model=model,
            val_dl=val_dl,
            device=device,
            tokenizer=tokenizer,
            max_batches=int(args.val_max_batches),
            gen_max_new_tokens=int(args.gen_max_new_tokens),
            gen_do_sample=bool(args.gen_do_sample),
            gen_temperature=float(args.gen_temperature),
            gen_top_p=float(args.gen_top_p),
            rouge_metric=rouge_metric,
            meteor_metric=meteor_metric,
            bleu_metric=bleu_metric,
            bertscore_metric=bertscore_metric,
            bertscore_model_type=str(args.bertscore_model_type),
            sample_limit=sample_count_for_logging,
        )

        train_rows = collect_split_samples(
            split_name="train",
            model=model,
            dl=train_dl,
            device=device,
            tokenizer=tokenizer,
            n_samples=sample_count_for_logging,
            sample_seed=int(args.seed) + epoch,
            gen_max_new_tokens=int(args.gen_max_new_tokens),
            gen_do_sample=bool(args.gen_do_sample),
            gen_temperature=float(args.gen_temperature),
            gen_top_p=float(args.gen_top_p),
        )
        train_rows = _ensure_min_sample_rows(train_rows, sample_count_for_logging)

        if len(val_rows) < sample_count_for_logging and len(val_dl.dataset) > 0:
            val_rows = collect_split_samples(
                split_name="val",
                model=model,
                dl=val_dl,
                device=device,
                tokenizer=tokenizer,
                n_samples=sample_count_for_logging,
                sample_seed=int(args.seed) + 10000 + epoch,
                gen_max_new_tokens=int(args.gen_max_new_tokens),
                gen_do_sample=bool(args.gen_do_sample),
                gen_temperature=float(args.gen_temperature),
                gen_top_p=float(args.gen_top_p),
            )
        val_rows = _ensure_min_sample_rows(val_rows, sample_count_for_logging)

        train_sample_preds = [str(r.get("prediction", "")) for r in train_rows]
        train_sample_refs = [str(r.get("reference", "")) for r in train_rows]
        val_sample_preds = [str(r.get("prediction", "")) for r in val_rows]
        val_sample_refs = [str(r.get("reference", "")) for r in val_rows]

        train_sample_metrics = _compute_text_metrics(
            preds=train_sample_preds,
            refs=train_sample_refs,
            rouge_metric=rouge_metric,
            meteor_metric=meteor_metric,
            bleu_metric=bleu_metric,
            bertscore_metric=bertscore_metric,
            bertscore_model_type=str(args.bertscore_model_type),
        )
        val_sample_metrics = _compute_text_metrics(
            preds=val_sample_preds,
            refs=val_sample_refs,
            rouge_metric=rouge_metric,
            meteor_metric=meteor_metric,
            bleu_metric=bleu_metric,
            bertscore_metric=bertscore_metric,
            bertscore_model_type=str(args.bertscore_model_type),
        )

        train_row_stats = _sample_rows_stats(train_rows, "train_sample")
        val_row_stats = _sample_rows_stats(val_rows, "val_sample")
        epoch_time_sec = float(time.perf_counter() - epoch_started)

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_loss_epoch:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"rougeL={val_metrics['rougeL']:.4f} "
            f"meteor={val_metrics['meteor']:.4f} "
            f"bleu4={val_metrics['bleu4']:.4f} "
            f"bertscore_f1={val_metrics['bertscore_f1']:.4f}"
        )

        epoch_log = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train/epoch_loss": float(train_loss_epoch),
            "train/epoch_time_sec": float(epoch_time_sec),
            "train/epoch_samples": float(samples_seen_epoch),
            "train/epoch_vision_tokens": float(vision_tokens_epoch),
            "train/epoch_text_tokens": float(text_tokens_epoch),
            "train/epoch_target_tokens": float(target_tokens_epoch),
            "train/epoch_wsi_count": float(wsi_seen_epoch),
            "train/epoch_samples_per_sec": float(samples_seen_epoch / max(1e-9, epoch_time_sec)),
            "train/epoch_updates": float(len(epoch_losses)),
            "train/avg_vision_tokens_per_sample_epoch": float(vision_tokens_epoch / max(1, samples_seen_epoch)),
            "train/avg_text_tokens_per_sample_epoch": float(text_tokens_epoch / max(1, samples_seen_epoch)),
            "train/avg_target_tokens_per_sample_epoch": float(target_tokens_epoch / max(1, samples_seen_epoch)),
            "train/avg_wsi_per_sample_epoch": float(wsi_seen_epoch / max(1, samples_seen_epoch)),
            "val/loss": float(val_metrics["val_loss"]),
            "val/rougeL": float(val_metrics["rougeL"]),
            "val/meteor": float(val_metrics["meteor"]),
            "val/bleu4": float(val_metrics["bleu4"]),
            "val/bertscore_f1": float(val_metrics["bertscore_f1"]),
            "val/n_samples": float(val_metrics["val_n_samples"]),
            "train_sample/rougeL": float(train_sample_metrics["rougeL"]),
            "train_sample/meteor": float(train_sample_metrics["meteor"]),
            "train_sample/bleu4": float(train_sample_metrics["bleu4"]),
            "train_sample/bertscore_f1": float(train_sample_metrics["bertscore_f1"]),
            "val_sample/rougeL": float(val_sample_metrics["rougeL"]),
            "val_sample/meteor": float(val_sample_metrics["meteor"]),
            "val_sample/bleu4": float(val_sample_metrics["bleu4"]),
            "val_sample/bertscore_f1": float(val_sample_metrics["bertscore_f1"]),
        }
        epoch_log.update(train_row_stats)
        epoch_log.update(val_row_stats)
        for k, v in val_metrics.items():
            if k in {"val_loss", "val_n_samples", "rougeL", "meteor", "bleu4", "bertscore_f1"}:
                continue
            if isinstance(v, (int, float)):
                if k.startswith("val_"):
                    epoch_log[f"val/{k[len('val_'):]}"] = float(v)
                else:
                    epoch_log[f"val/{k}"] = float(v)

        cur_val_loss = float(val_metrics["val_loss"])
        epoch_history.append(int(epoch))
        train_epoch_loss_history.append(float(train_loss_epoch))
        val_loss_history.append(float(cur_val_loss))
        save_loss_history_json(
            output_dir=output_dir,
            epochs=epoch_history,
            train_epoch_losses=train_epoch_loss_history,
            val_losses=val_loss_history,
        )
        is_best = cur_val_loss < float(best_val_loss)
        if is_best:
            best_val_loss = cur_val_loss
            best_epoch = int(epoch)
            best_global_step = int(global_step)
            best_val_metrics = {k: float(v) for k, v in val_metrics.items()}

            best_ckpt_path = save_alignment_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=tokenizer,
                epoch=epoch,
                global_step=global_step,
                args=args,
                filename="alignment_best.pt",
            )
            best_aligner_path = save_best_aligner_weights(
                output_dir=output_dir,
                model=model,
                epoch=epoch,
                global_step=global_step,
                val_metrics=val_metrics,
            )
            best_summary = {
                "best_epoch": int(best_epoch),
                "best_global_step": int(best_global_step),
                "best_val_loss": float(best_val_loss),
                "best_val_metrics": best_val_metrics,
                "best_checkpoint": str(best_ckpt_path),
                "best_aligner_weights": str(best_aligner_path),
            }
            (output_dir / "best_checkpoint_summary.json").write_text(
                json.dumps(best_summary, indent=2), encoding="utf-8"
            )
            print(
                "Updated best checkpoint:",
                best_ckpt_path,
                "| best aligner:",
                best_aligner_path,
                f"| val_loss={best_val_loss:.4f}",
            )

        epoch_log["val/is_best"] = float(1.0 if is_best else 0.0)
        epoch_log["best/val_loss_so_far"] = float(best_val_loss)
        epoch_log["best/epoch"] = float(best_epoch)

        if wandb_run is not None:
            log_epoch_loss_curve_to_wandb(
                wandb_run=wandb_run,
                epochs=epoch_history,
                train_epoch_losses=train_epoch_loss_history,
                val_losses=val_loss_history,
                step=global_step,
            )
            wandb_run.log(epoch_log, step=global_step)
            log_samples_to_wandb(
                rows=train_rows,
                split_name="train",
                epoch=epoch,
                step=global_step,
                wandb_run=wandb_run,
            )
            log_samples_to_wandb(
                rows=val_rows,
                split_name="val",
                epoch=epoch,
                step=global_step,
                wandb_run=wandb_run,
            )

        sample_dump = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_rows": train_rows,
            "val_rows": val_rows,
            "loss_history": [
                {
                    "epoch": int(e),
                    "train_epoch_loss": float(tr),
                    "val_loss": float(va),
                }
                for e, tr, va in zip(epoch_history, train_epoch_loss_history, val_loss_history)
            ],
            "train_sample_metrics": train_sample_metrics,
            "val_sample_metrics": val_sample_metrics,
            "train_sample_stats": train_row_stats,
            "val_sample_stats": val_row_stats,
            "val_metrics": val_metrics,
            "epoch_log": epoch_log,
        }
        (output_dir / f"epoch_{epoch:03d}_samples_and_metrics.json").write_text(
            json.dumps(sample_dump, indent=2), encoding="utf-8"
        )

        if int(args.save_every_epoch) > 0 and epoch % int(args.save_every_epoch) == 0:
            ckpt_path = save_alignment_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=tokenizer,
                epoch=epoch,
                global_step=global_step,
                args=args,
            )
            print("Saved checkpoint:", ckpt_path)

        if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
            break

    final_ckpt = save_alignment_checkpoint(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        epoch=int(epoch),
        global_step=global_step,
        args=args,
    )
    print("Final checkpoint:", final_ckpt)
    if best_ckpt_path is not None:
        print("Best checkpoint:", best_ckpt_path)
    if best_aligner_path is not None:
        print("Best aligner weights:", best_aligner_path)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
