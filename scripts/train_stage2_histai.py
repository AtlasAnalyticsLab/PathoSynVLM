from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, TaskType, get_peft_model

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathosynvlm.histai_dataset import create_train_val_dataloaders, normalize_target_field_name, resolve_target_field_label
from pathosynvlm.metrics import summarize_field_accuracy
from pathosynvlm.model import PathoSynVLM, load_aligner_from_checkpoint
from pathosynvlm.paths import get_path_defaults


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
    if name == "cosine_annealing":
        warmup_steps = max(0, int(warmup_steps))
        total_steps = max(1, int(total_steps))
        anneal_steps = max(1, int(total_steps) - int(warmup_steps))
        if warmup_steps > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=int(warmup_steps),
            )
            cosine = CosineAnnealingLR(optimizer, T_max=int(anneal_steps), eta_min=0.0)
            return SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[int(warmup_steps)],
            )
        return CosineAnnealingLR(optimizer, T_max=int(anneal_steps), eta_min=0.0)
    if name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_steps),
            num_training_steps=int(total_steps),
        )
    if name == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(warmup_steps))
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def _resolve_transformer_layers(llm: torch.nn.Module) -> list[torch.nn.Module]:
    candidates: list[tuple[int, int, torch.nn.ModuleList]] = []
    for name, module in llm.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            score = 2 if str(name).endswith("layers") else 1
            candidates.append((score, int(len(module)), module))
    if not candidates:
        raise RuntimeError("Could not locate transformer layers in the LLM module.")
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return list(candidates[0][2])


def _resolve_llm_train_scope(args: argparse.Namespace) -> str:
    scope = str(getattr(args, "llm_train_scope", "") or "").strip().lower()
    if scope:
        return scope
    return "full" if bool(args.unfreeze_llm_base) else "none"


def _unfreeze_llm_norms(llm: torch.nn.Module) -> int:
    trainable = 0
    for _module_name, module in llm.named_modules():
        cls_name = type(module).__name__.lower()
        if "norm" not in cls_name:
            continue
        for param in module.parameters(recurse=False):
            param.requires_grad = True
            trainable += int(param.numel())
    return trainable


def _apply_llm_train_scope(model: PathoSynVLM, *, scope: str, use_lora: bool) -> dict[str, Any]:
    llm = model.llm
    layers: list[torch.nn.Module] | None = None
    scope = str(scope).strip().lower()

    if scope == "none":
        return {"scope": scope, "trainable_llm_params": 0, "num_llm_layers": None}

    if scope == "full":
        for name, param in llm.named_parameters():
            if (not use_lora) or ("lora_" not in name):
                param.requires_grad = True
        return {
            "scope": scope,
            "trainable_llm_params": int(sum(p.numel() for p in llm.parameters() if p.requires_grad)),
            "num_llm_layers": None,
        }

    if scope == "second_half":
        layers = _resolve_transformer_layers(llm)
        first_trainable_layer = int(len(layers) // 2)
        for layer in layers[first_trainable_layer:]:
            for param in layer.parameters():
                param.requires_grad = True
        for module_name, module in llm.named_modules():
            cls_name = type(module).__name__.lower()
            if "norm" not in cls_name:
                continue
            if str(module_name) not in {"norm", "model.norm", "base_model.model.norm"} and not str(module_name).endswith(".norm"):
                continue
            for param in module.parameters(recurse=False):
                param.requires_grad = True
        if hasattr(llm, "lm_head"):
            for param in llm.lm_head.parameters():
                param.requires_grad = True
        return {
            "scope": scope,
            "trainable_llm_params": int(sum(p.numel() for p in llm.parameters() if p.requires_grad)),
            "num_llm_layers": int(len(layers)),
            "first_trainable_layer": int(first_trainable_layer),
        }

    if scope == "norm_only":
        norm_params = _unfreeze_llm_norms(llm)
        return {
            "scope": scope,
            "trainable_llm_params": int(sum(p.numel() for p in llm.parameters() if p.requires_grad)),
            "norm_only_params": int(norm_params),
            "num_llm_layers": None,
        }

    raise ValueError(f"Unsupported llm_train_scope: {scope}")


class AnchorRegularizer:
    def __init__(
        self,
        model: PathoSynVLM,
        *,
        llm_weight: float,
        aligner_weight: float,
        marker_weight: float,
    ) -> None:
        self.llm_weight = max(0.0, float(llm_weight))
        self.aligner_weight = max(0.0, float(aligner_weight))
        self.marker_weight = max(0.0, float(marker_weight))
        self._groups: dict[str, list[tuple[torch.nn.Parameter, torch.Tensor]]] = {
            "llm": [],
            "aligner": [],
            "markers": [],
        }

        for name, param in model.named_parameters():
            if not param.requires_grad or not param.is_floating_point():
                continue
            if name.startswith("llm.") and "lora_" not in name and self.llm_weight > 0.0:
                self._groups["llm"].append((param, param.detach().to(device=param.device, dtype=torch.float32).clone()))
            elif name.startswith("aligner.") and self.aligner_weight > 0.0:
                self._groups["aligner"].append((param, param.detach().to(device=param.device, dtype=torch.float32).clone()))
            elif (name.startswith("wsi_index_emb.") or name == "wsi_sep") and self.marker_weight > 0.0:
                self._groups["markers"].append((param, param.detach().to(device=param.device, dtype=torch.float32).clone()))

    @property
    def enabled(self) -> bool:
        return any(self._groups.values())

    def summary(self) -> dict[str, Any]:
        return {
            "llm_weight": float(self.llm_weight),
            "aligner_weight": float(self.aligner_weight),
            "marker_weight": float(self.marker_weight),
            "llm_tensors": int(len(self._groups["llm"])),
            "aligner_tensors": int(len(self._groups["aligner"])),
            "marker_tensors": int(len(self._groups["markers"])),
        }

    def compute(self) -> tuple[torch.Tensor | None, dict[str, float]]:
        total_loss: torch.Tensor | None = None
        metrics: dict[str, float] = {}
        group_cfg = {
            "llm": float(self.llm_weight),
            "aligner": float(self.aligner_weight),
            "markers": float(self.marker_weight),
        }

        for group_name, weight in group_cfg.items():
            pairs = self._groups[group_name]
            if weight <= 0.0 or not pairs:
                continue

            sq_sum = 0.0
            numel = 0
            for param, anchor_cpu in pairs:
                diff = param.float() - anchor_cpu
                sq_sum = sq_sum + diff.pow(2).sum()
                numel += int(param.numel())

            if numel <= 0:
                continue
            group_loss = sq_sum / float(numel)
            metrics[f"{group_name}_anchor_reg_loss"] = float(group_loss.detach().cpu().item())
            weighted = group_loss * float(weight)
            total_loss = weighted if total_loss is None else (total_loss + weighted)

        return total_loss, metrics


def _try_print_nvidia_smi() -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as e:
        print(f"[warn] nvidia-smi unavailable: {e}")
        return

    if out.stdout:
        print("[gpu] nvidia-smi:")
        print(out.stdout.rstrip())
    elif out.stderr:
        print("[warn] nvidia-smi stderr:", out.stderr.strip())


def print_runtime_device_info(device: torch.device) -> None:
    print(f"Using device: {device}")
    print(f"Torch: {torch.__version__} | Torch CUDA build: {torch.version.cuda}")

    if device.type != "cuda":
        print("[warn] CUDA is not available. Training will run on CPU.")
        return

    try:
        n_gpu = int(torch.cuda.device_count())
        cur = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(cur)
        print(
            "[gpu] "
            f"count={n_gpu} current={cur} name={props.name} "
            f"total_mem_gb={float(props.total_memory) / (1024 ** 3):.1f}"
        )
        print(
            "[gpu] "
            f"mem_alloc_mb={torch.cuda.memory_allocated(cur) / (1024 ** 2):.1f} "
            f"mem_reserved_mb={torch.cuda.memory_reserved(cur) / (1024 ** 2):.1f}"
        )
    except Exception as e:
        print(f"[warn] failed to query torch cuda device props: {e}")

    _try_print_nvidia_smi()


def init_distributed_if_needed() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if not is_distributed:
        return {
            "is_distributed": False,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
        }

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed launch requested (WORLD_SIZE>1) but CUDA is not available.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    return {
        "is_distributed": True,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
    }


def cleanup_distributed_if_needed(is_distributed: bool) -> None:
    if is_distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class ExponentialMovingAverage:
    """EMA over floating trainable params, with swap-in/swap-out for eval/checkpoint."""

    def __init__(self, model: torch.nn.Module, *, decay: float) -> None:
        d = float(decay)
        if not (0.0 < d < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1), got {decay!r}")

        tracked: list[tuple[str, torch.nn.Parameter]] = []
        shadow: list[torch.Tensor] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not param.is_floating_point():
                continue
            tracked.append((str(name), param))
            shadow.append(param.detach().to(device="cpu", dtype=torch.float32).clone())

        self.decay = d
        self._tracked = tracked
        self._shadow = shadow
        self._is_swapped = False
        self._numel = int(sum(int(p.numel()) for _, p in tracked))
        self._bytes = int(sum(int(t.numel()) * int(t.element_size()) for t in shadow))

    @property
    def tracked_tensors(self) -> int:
        return int(len(self._tracked))

    @property
    def tracked_numel(self) -> int:
        return int(self._numel)

    @property
    def shadow_bytes(self) -> int:
        return int(self._bytes)

    @property
    def is_swapped(self) -> bool:
        return bool(self._is_swapped)

    def update(self) -> None:
        if self._is_swapped:
            raise RuntimeError("Cannot update EMA while EMA weights are swapped into the model.")
        if not self._tracked:
            return
        one_minus = float(1.0 - self.decay)
        with torch.no_grad():
            for (_, param), shadow in zip(self._tracked, self._shadow):
                shadow.mul_(self.decay).add_(param.detach().to(device="cpu", dtype=torch.float32), alpha=one_minus)

    def swap_parameters(self) -> None:
        if not self._tracked:
            self._is_swapped = not self._is_swapped
            return
        with torch.no_grad():
            for (_, param), shadow in zip(self._tracked, self._shadow):
                model_copy = param.detach().to(device="cpu", dtype=torch.float32)
                param.copy_(shadow.to(device=param.device, dtype=param.dtype))
                shadow.copy_(model_copy)
        self._is_swapped = not self._is_swapped


def save_loss_history_json(
    *,
    output_dir: Path,
    epochs: List[int],
    train_epoch_losses: List[float],
    val_losses: List[float],
) -> Path:
    rows: List[Dict[str, Any]] = []
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


def save_loss_curve_plot(
    *,
    output_dir: Path,
    epochs: List[int],
    train_epoch_losses: List[float],
    val_losses: List[float],
) -> Path | None:
    if not epochs:
        return None

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"[warn] matplotlib unavailable; skip loss curve png: {e}")
        return None

    out_path = output_dir / "loss_by_epoch.png"
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_epoch_losses, marker="o", label="train/epoch_loss")
        plt.plot(epochs, val_losses, marker="o", label="val/loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training vs Validation Loss by Epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path
    except Exception as e:
        print(f"[warn] failed to save loss curve png: {e}")
        return None


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


def save_checkpoint(
    *,
    save_dir: Path,
    model: PathoSynVLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    tokenizer,
    extra: Optional[Dict[str, Any]] = None,
    save_full_llm_state: bool = False,
    include_optimizer_state: bool = True,
    include_scheduler_state: bool = True,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)

    llm_save_dir = save_dir / f"llm_step_{int(step)}"
    llm_save_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model.llm, "save_pretrained"):
        model.llm.save_pretrained(llm_save_dir)

    ckpt: Dict[str, Any] = {
        "step": int(step),
        "aligner": model.aligner.state_dict() if hasattr(model, "aligner") else None,
        "use_wsi_markers": bool(getattr(model, "use_wsi_markers", False)),
        "wsi_index_emb": model.wsi_index_emb.state_dict() if hasattr(model, "wsi_index_emb") else None,
        "wsi_sep": model.wsi_sep.detach().cpu() if hasattr(model, "wsi_sep") else None,
        "optimizer": (optimizer.state_dict() if bool(include_optimizer_state) else None),
        "scheduler": (scheduler.state_dict() if (scheduler is not None and bool(include_scheduler_state)) else None),
        "extra": extra or {},
    }
    if bool(save_full_llm_state):
        ckpt["llm_state_dict"] = model.llm.state_dict()

    state_path = save_dir / f"trainer_state_step_{int(step)}.pt"
    torch.save(ckpt, state_path)

    tok_dir = save_dir / "tokenizer"
    if not tok_dir.exists():
        tokenizer.save_pretrained(tok_dir)

    return state_path


def _metric_float(metrics: Dict[str, Any], key: str, default: float) -> float:
    try:
        v = float(metrics.get(key, default))
        if not math.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def build_checkpoint_rank(
    metrics: Dict[str, Any],
    *,
    report_target_field_name: str = "conclusion",
) -> tuple[tuple[float, float, float, float, float], Dict[str, float]]:
    # Higher is better for all entries in rank tuple.
    field_name = normalize_target_field_name(report_target_field_name)
    meteor = _metric_float(metrics, f"val_{field_name}_meteor", float("-inf"))
    bleu4 = _metric_float(metrics, f"val_{field_name}_bleu4", float("-inf"))
    rouge_l = _metric_float(metrics, f"val_{field_name}_rougeL", float("-inf"))
    bert_f1 = _metric_float(metrics, "val_bertscore_f1", float("-inf"))
    val_loss = _metric_float(metrics, "val_loss", float("inf"))

    rank_metrics = {
        f"val_{field_name}_meteor": float(meteor),
        f"val_{field_name}_bleu4": float(bleu4),
        f"val_{field_name}_rougeL": float(rouge_l),
        "val_bertscore_f1": float(bert_f1),
        "val_loss": float(val_loss),
    }
    # Lower loss is better, so use -loss.
    rank_tuple = (float(meteor), float(bleu4), float(rouge_l), float(bert_f1), float(-val_loss))
    return rank_tuple, rank_metrics


def remove_checkpoint_step_artifacts(output_dir: Path, step: int) -> bool:
    removed = False
    state_path = output_dir / f"trainer_state_step_{int(step)}.pt"
    llm_dir = output_dir / f"llm_step_{int(step)}"
    if state_path.exists():
        state_path.unlink()
        removed = True
    if llm_dir.exists():
        shutil.rmtree(llm_dir, ignore_errors=True)
        removed = True
    return removed


def update_best_checkpoint_aliases(output_dir: Path, *, step: int) -> None:
    aliases = [
        (output_dir / f"trainer_state_step_{int(step)}.pt", output_dir / "best_trainer_state.pt", False),
        (output_dir / f"llm_step_{int(step)}", output_dir / "best_llm", True),
    ]

    for src, dst, is_dir in aliases:
        if not src.exists():
            continue

        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst, ignore_errors=True)

        try:
            os.symlink(src.name, dst, target_is_directory=is_dir)
        except OSError:
            if is_dir:
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)


def _extract_prompt_and_refs(
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    bsz = int(input_ids.shape[0])
    device = input_ids.device

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prompt_id_list: List[torch.Tensor] = []
    refs: List[str] = []
    prompts: List[str] = []

    for i in range(bsz):
        valid_len = int(attention_mask[i].sum().item())
        ids_i = input_ids[i, :valid_len]
        labels_i = labels[i, :valid_len]

        ref_token_ids = labels_i[labels_i != -100]
        ref_text = tokenizer.decode(ref_token_ids.tolist(), skip_special_tokens=True).strip()
        refs.append(ref_text)

        tgt_pos = (labels_i != -100).nonzero(as_tuple=False)
        prompt_end = int(tgt_pos[0].item()) if tgt_pos.numel() > 0 else valid_len

        p_ids = ids_i[:prompt_end]
        p_text = tokenizer.decode(p_ids.tolist(), skip_special_tokens=True).strip()
        prompts.append(p_text)
        prompt_id_list.append(p_ids)

    max_p = max((x.shape[0] for x in prompt_id_list), default=1)
    prompt_input_ids = torch.full((bsz, max_p), pad_id, device=device, dtype=input_ids.dtype)
    prompt_attention_mask = torch.zeros((bsz, max_p), device=device, dtype=attention_mask.dtype)

    for i in range(bsz):
        l = int(prompt_id_list[i].shape[0])
        if l == 0:
            continue
        if getattr(tokenizer, "padding_side", "right") == "left":
            prompt_input_ids[i, -l:] = prompt_id_list[i]
            prompt_attention_mask[i, -l:] = 1
        else:
            prompt_input_ids[i, :l] = prompt_id_list[i]
            prompt_attention_mask[i, :l] = 1

    return prompt_input_ids, prompt_attention_mask, refs, prompts


@torch.no_grad()
def run_validation(
    *,
    model: PathoSynVLM,
    val_dl: DataLoader,
    device: torch.device,
    tokenizer,
    max_batches: int,
    gen_max_new_tokens: int,
    gen_min_new_tokens: int,
    gen_do_sample: bool,
    gen_temperature: float,
    gen_top_p: float,
    certainty_percent_threshold: float,
    rouge_metric=None,
    meteor_metric=None,
    bleu_metric=None,
    bertscore_metric=None,
    bertscore_model_type: str = "roberta-large",
    report_target_field_name: str = "conclusion",
    report_target_field_label: str | None = None,
    wandb_run=None,
    global_step: int = 0,
    log_n_samples: int = 16,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    target_field_name = normalize_target_field_name(report_target_field_name)
    model.eval()
    losses: List[float] = []
    all_preds: List[str] = []
    all_refs: List[str] = []
    sample_rows: List[Dict[str, Any]] = []
    samples_seen = 0
    val_vision_tokens = 0
    val_text_tokens = 0
    val_target_tokens = 0
    val_wsi_count = 0

    for b_idx, batch in enumerate(val_dl):
        if int(max_batches) > 0 and b_idx >= int(max_batches):
            break

        batch = to_device(batch, device)
        bsz = int(batch["vision_embeddings"].shape[0]) if "vision_embeddings" in batch else 0
        samples_seen += bsz
        if "vision_attention_mask" in batch:
            val_vision_tokens += int(batch["vision_attention_mask"].sum().item())
        if "attention_mask" in batch:
            val_text_tokens += int(batch["attention_mask"].sum().item())
        if "labels" in batch:
            val_target_tokens += int((batch["labels"] != -100).sum().item())
        if "wsi_patch_counts" in batch:
            val_wsi_count += int((batch["wsi_patch_counts"] > 0).sum().item())

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

        preds: List[str] = []
        pred_gen_token_counts: List[int] = []
        prompt_token_counts: List[int] = []
        for i in range(gen_ids.shape[0]):
            gen_i = gen_ids[i]
            pm = prompt_mask[i].bool()
            prompt_1d = prompt_ids[i, pm]
            lp = int(prompt_1d.numel())
            prompt_token_counts.append(lp)

            if gen_i.numel() >= lp and torch.equal(gen_i[:lp], prompt_1d):
                cont_ids = gen_i[lp:]
            else:
                cont_ids = gen_i

            pred = tokenizer.decode(cont_ids.tolist(), skip_special_tokens=True).strip()
            pred = pred.replace("<|endoftext|>", "").strip()
            preds.append(pred)
            pred_gen_token_counts.append(int(cont_ids.numel()))

        all_preds.extend(preds)
        all_refs.extend(refs)

        if len(sample_rows) < int(log_n_samples):
            case_ids = [str(x) for x in batch.get("case_mapping", [])] if "case_mapping" in batch else []
            if not case_ids:
                case_ids = [f"batch{b_idx}_i{i}" for i in range(len(preds))]

            for i in range(len(preds)):
                if len(sample_rows) >= int(log_n_samples):
                    break
                slide_paths_any = ""
                if "slide_paths" in batch and i < len(batch["slide_paths"]):
                    raw_slide_paths = batch["slide_paths"][i]
                    if isinstance(raw_slide_paths, (list, tuple)):
                        slide_paths_any = "; ".join(str(x) for x in raw_slide_paths)
                    else:
                        slide_paths_any = str(raw_slide_paths)
                vision_tokens_i = (
                    int(batch["vision_attention_mask"][i].sum().item())
                    if "vision_attention_mask" in batch
                    else 0
                )
                text_tokens_i = int(batch["attention_mask"][i].sum().item()) if "attention_mask" in batch else 0
                target_tokens_i = int((batch["labels"][i] != -100).sum().item()) if "labels" in batch else 0
                wsi_count_i = int((batch["wsi_patch_counts"][i] > 0).sum().item()) if "wsi_patch_counts" in batch else 0
                sample_rows.append(
                    {
                        "case": case_ids[i],
                        "slide_paths": slide_paths_any,
                        "prompt": prompts[i],
                        "prediction": preds[i],
                        "reference": refs[i],
                        "prompt_token_count": int(prompt_token_counts[i]),
                        "generated_token_count": int(pred_gen_token_counts[i]),
                        "text_token_count": int(text_tokens_i),
                        "target_token_count": int(target_tokens_i),
                        "vision_token_count": int(vision_tokens_i),
                        "wsi_count": int(wsi_count_i),
                    }
                )

    metrics: Dict[str, Any] = {
        "val_loss": float(sum(losses) / max(1, len(losses))),
        "val_n_samples": int(len(all_preds)),
        "val/vision_tokens_total": int(val_vision_tokens),
        "val/text_tokens_total": int(val_text_tokens),
        "val/target_tokens_total": int(val_target_tokens),
        "val/wsi_count_total": int(val_wsi_count),
        "val/avg_vision_tokens_per_sample": float(val_vision_tokens / max(1, samples_seen)),
        "val/avg_text_tokens_per_sample": float(val_text_tokens / max(1, samples_seen)),
        "val/avg_target_tokens_per_sample": float(val_target_tokens / max(1, samples_seen)),
        "val/avg_wsi_per_sample": float(val_wsi_count / max(1, samples_seen)),
    }

    if bertscore_metric is not None and all_preds and all_refs:
        try:
            bs = bertscore_metric.compute(
                predictions=all_preds,
                references=all_refs,
                lang="en",
                model_type=str(bertscore_model_type),
                rescale_with_baseline=True,
                use_fast_tokenizer=True,
            )
            f1_list = bs.get("f1", []) if isinstance(bs, dict) else []
            if f1_list:
                metrics["val_bertscore_f1"] = float(sum(float(x) for x in f1_list) / max(1, len(f1_list)))
        except Exception as e:
            metrics["val/bertscore_error"] = str(e)

    try:
        field_summary = summarize_field_accuracy(
            predicted_texts=all_preds,
            reference_texts=all_refs,
            certainty_percent_threshold=float(certainty_percent_threshold),
            third_field_name=target_field_name,
            third_field_label=report_target_field_label,
            rouge_metric=rouge_metric,
            meteor_metric=meteor_metric,
            bleu_metric=bleu_metric,
        )

        metrics["val/field_eval_n"] = int(field_summary.get("n", 0))
        metrics["val_diagnosis_relaxed_match_rate"] = float(field_summary.get("diagnosis_relaxed_match_rate", 0.0))
        metrics["val_diagnosis_exact_match_rate"] = float(field_summary.get("diagnosis_exact_match_rate", 0.0))
        metrics["val_certainty_match_rate"] = float(field_summary.get("certainty_match_rate", 0.0))
        metrics["val_certainty_exact_match_rate"] = float(field_summary.get("certainty_exact_match_rate", 0.0))
        metrics[f"val_{target_field_name}_exact_match_rate"] = float(
            field_summary.get(f"{target_field_name}_exact_match_rate", 0.0)
        )

        if field_summary.get(f"{target_field_name}_rougeL") is not None:
            metrics[f"val_{target_field_name}_rougeL"] = float(field_summary[f"{target_field_name}_rougeL"])
        if field_summary.get(f"{target_field_name}_meteor") is not None:
            metrics[f"val_{target_field_name}_meteor"] = float(field_summary[f"{target_field_name}_meteor"])
        if field_summary.get(f"{target_field_name}_bleu4") is not None:
            metrics[f"val_{target_field_name}_bleu4"] = float(field_summary[f"{target_field_name}_bleu4"])

        pred_counts = field_summary.get("pred_format_score_counts", {}) or {}
        ref_counts = field_summary.get("ref_format_score_counts", {}) or {}
        for s in ("0", "1", "2", "3"):
            metrics[f"val/pred_format_score_{s}_count"] = int(pred_counts.get(s, 0))
            metrics[f"val/ref_format_score_{s}_count"] = int(ref_counts.get(s, 0))

        metrics["val/pred_diagnosis_present_rate"] = float(field_summary.get("pred_diagnosis_present_rate", 0.0))
        metrics["val/pred_certainty_present_rate"] = float(field_summary.get("pred_certainty_present_rate", 0.0))
        metrics[f"val/pred_{target_field_name}_present_rate"] = float(
            field_summary.get(f"pred_{target_field_name}_present_rate", 0.0)
        )
    except Exception as e:
        metrics["val/field_eval_error"] = str(e)

    if wandb_run is not None:
        import wandb

        log_data: Dict[str, Any] = {"val/loss": metrics["val_loss"], "val/n_samples": metrics["val_n_samples"]}
        for k in (
            "val_diagnosis_relaxed_match_rate",
            "val_diagnosis_exact_match_rate",
            "val_certainty_match_rate",
            "val_certainty_exact_match_rate",
            f"val_{target_field_name}_exact_match_rate",
            f"val_{target_field_name}_rougeL",
            f"val_{target_field_name}_meteor",
            f"val_{target_field_name}_bleu4",
            "val_bertscore_f1",
        ):
            if k in metrics:
                log_data[k.replace("val_", "val/")] = metrics[k]

        for k, v in metrics.items():
            if k.startswith("val/pred_format_score_") or k.startswith("val/ref_format_score_"):
                log_data[k] = v
            if k.startswith("val/pred_") and k.endswith("_present_rate"):
                log_data[k] = v
            if k.startswith("val/") and isinstance(v, (int, float)):
                log_data[k] = float(v)

        wandb_run.log(log_data, step=int(global_step))

        if sample_rows:
            table = wandb.Table(
                columns=[
                    "case",
                    "slide_paths",
                    "prompt",
                    "prediction",
                    "reference",
                    "prompt_token_count",
                    "generated_token_count",
                    "text_token_count",
                    "target_token_count",
                    "vision_token_count",
                    "wsi_count",
                ]
            )
            for r in sample_rows:
                table.add_data(
                    r["case"],
                    r.get("slide_paths", ""),
                    r["prompt"],
                    r["prediction"],
                    r["reference"],
                    int(r.get("prompt_token_count", 0)),
                    int(r.get("generated_token_count", 0)),
                    int(r.get("text_token_count", 0)),
                    int(r.get("target_token_count", 0)),
                    int(r.get("vision_token_count", 0)),
                    int(r.get("wsi_count", 0)),
                )
            wandb_run.log({"val/samples": table}, step=int(global_step))

    model.train()
    return metrics, sample_rows


def _param_groups(
    model: PathoSynVLM,
    *,
    lr_lora: float,
    lr_aligner: float,
    lr_llm_base: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    lora_params = []
    aligner_params = []
    marker_params = []
    llm_base_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "llm" in n and "lora_" in n:
            lora_params.append(p)
        elif "aligner" in n:
            aligner_params.append(p)
        elif "wsi_index_emb" in n or "wsi_sep" in n:
            marker_params.append(p)
        elif "llm" in n:
            llm_base_params.append(p)
        else:
            marker_params.append(p)

    groups: list[dict[str, Any]] = []
    if lora_params:
        groups.append({"name": "lora", "params": lora_params, "lr": float(lr_lora), "weight_decay": float(weight_decay)})
    if aligner_params:
        groups.append(
            {"name": "aligner", "params": aligner_params, "lr": float(lr_aligner), "weight_decay": float(weight_decay)}
        )
    if marker_params:
        groups.append({"name": "markers", "params": marker_params, "lr": float(lr_aligner), "weight_decay": float(weight_decay)})
    if llm_base_params:
        groups.append(
            {
                "name": "llm_base",
                "params": llm_base_params,
                "lr": float(lr_llm_base),
                "weight_decay": float(weight_decay),
            }
        )
    return groups


def _build_optimizer(
    *,
    optimizer_name: str,
    param_groups: list[dict[str, Any]],
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    name = str(optimizer_name).strip().lower()
    if name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=float(lr),
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=float(weight_decay),
            foreach=False,
        )
    if name != "muon":
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}")

    muon_cls: Any = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        try:
            from muon import Muon as muon_cls  # type: ignore
        except Exception:
            try:
                from muon_optimizer import Muon as muon_cls  # type: ignore
            except Exception as e_muon_optimizer:
                raise RuntimeError(
                    "optimizer='muon' requested but Muon is unavailable. "
                    "Install Muon in the environment or switch to --optimizer adamw."
                ) from e_muon_optimizer

    init_sig = inspect.signature(muon_cls.__init__)
    kwargs: dict[str, Any] = {"params": param_groups, "lr": float(lr)}
    if "weight_decay" in init_sig.parameters:
        kwargs["weight_decay"] = float(weight_decay)
    if "betas" in init_sig.parameters:
        kwargs["betas"] = (0.9, 0.95)
    if "eps" in init_sig.parameters:
        kwargs["eps"] = 1e-8
    if "foreach" in init_sig.parameters:
        kwargs["foreach"] = False
    return muon_cls(**kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HistAI finetuning (all HISTAI domains) with alignment init + LoRA/full-FT")
    repo_root = Path(__file__).resolve().parents[1]

    def _parse_bool(x: Any) -> bool:
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {x!r}")

    # Data
    paths = get_path_defaults(repo_root)
    p.add_argument(
        "--metadata_standardized_json",
        type=str,
        default=str(paths.histai_metadata_dir / "standardized_metadata_fixed_filtered_5x_512.json"),
    )
    p.add_argument(
        "--dataset_embeddings_root",
        type=str,
        default=str(paths.embeddings_root),
        help="Root containing dataset embedding folders. Defaults to PATHOSYNVLM_EMBEDDINGS_ROOT or data/embeddings.",
    )
    p.add_argument(
        "--alignment_metadata_json",
        type=str,
        default=str(paths.stage1_metadata_dir / "merged_metadata_3datasets_filtered_conch_v15.json"),
    )
    p.add_argument("--feature_key", type=str, default="conch_v15")
    p.add_argument("--patch_level", type=str, default="5x_512", choices=["1x_512", "5x_512"])
    p.add_argument("--alignment_dataset_selection", type=str, default="all")
    p.add_argument(
        "--alignment_mix_fraction",
        type=float,
        default=0.0,
        help="Optional fraction of stage-1 alignment train data to mix into HistAI training.",
    )
    p.add_argument("--val_size", default="0.2")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_text_length", type=int, default=512)
    p.add_argument("--max_vision_tokens", type=int, default=0, help="Cap vision tokens per sample (0 disables cap).")
    p.add_argument(
        "--vision_token_dropout",
        type=float,
        default=0.0,
        help="Train-time token dropout on vision tokens (0 disables; tokens are zeroed, shape unchanged).",
    )
    p.add_argument("--missing_policy", choices=["skip", "error"], default="skip")
    p.add_argument("--probe_h5_on_init", action="store_true")

    # Model
    p.add_argument("--llm", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--vision_dim", type=int, default=768)
    p.add_argument("--use_wsi_markers", action="store_true", default=True)
    p.add_argument("--no_use_wsi_markers", dest="use_wsi_markers", action="store_false")
    p.add_argument(
        "--use_wsi_index_emb",
        type=_parse_bool,
        default=True,
        help="Enable learned WSI index embeddings when WSI markers are enabled.",
    )
    # Backward-compatible alias for older command templates.
    p.add_argument("--wsi_index_emb", dest="use_wsi_index_emb", type=_parse_bool, help=argparse.SUPPRESS)
    p.add_argument(
        "--no_use_wsi_index_emb",
        dest="use_wsi_index_emb",
        action="store_false",
        help="Disable learned WSI index embeddings and use only shared separator marker.",
    )
    p.add_argument("--prompt_style", type=str, default="single", choices=["single", "double"])
    p.add_argument("--alignment_prompt_style", type=str, default="single", choices=["single", "double"])
    p.add_argument("--report_target_field", type=str, default="conclusion", choices=["conclusion", "micro_protocol"])
    p.add_argument("--report_target_label", type=str, default="")

    # Alignment init
    p.add_argument("--aligner_init", type=str, default="", help="Alignment checkpoint path (file or run dir)")
    p.add_argument("--skip_aligner_init", action="store_true", help="Do not load aligner init checkpoint.")
    p.add_argument("--strict_aligner_load", action="store_true", default=True)
    p.add_argument("--no_strict_aligner_load", dest="strict_aligner_load", action="store_false")

    # Train
    p.add_argument("--output_dir", type=str, default=str(paths.runs_root / "stage2_main"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5, help="LoRA LR")
    p.add_argument("--aligner_lr", type=float, default=1e-4)
    p.add_argument("--llm_base_lr", type=float, default=5e-6, help="Used whenever LLM base parameters are trainable.")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--scheduler", choices=["cosine", "cosine_annealing", "linear", "constant"], default="cosine")
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpoint", action="store_true")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--ema", action="store_true", help="Enable exponential moving average (EMA) for trainable weights.")
    p.add_argument("--EMA", dest="ema", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay in (0,1), used when --ema is enabled.")
    p.add_argument("--EMA_DECAY", dest="ema_decay", type=float, help=argparse.SUPPRESS)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target", type=str, default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--use_lora", type=_parse_bool, default=True, help="Enable LoRA adapters for the LLM.")
    p.add_argument("--no_use_lora", dest="use_lora", action="store_false", help="Disable LoRA adapters.")
    p.add_argument(
        "--freeze_aligner",
        action="store_true",
        help="Freeze the vision aligner (MLP) module.",
    )
    p.add_argument(
        "--unfreeze_llm_base",
        action="store_true",
        default=False,
        help="Train full LLM base weights in addition to LoRA adapters (default: disabled, LoRA-only).",
    )
    p.add_argument(
        "--freeze_llm_base",
        dest="unfreeze_llm_base",
        action="store_false",
        help="Freeze LLM base weights and train LoRA adapters only (default behavior).",
    )
    p.add_argument(
        "--llm_train_scope",
        type=str,
        default="",
        choices=["", "none", "full", "second_half", "norm_only"],
        help="Explicit LLM base trainability policy. Empty keeps legacy behavior from --unfreeze_llm_base.",
    )
    p.add_argument("--aligner_anchor_reg_weight", type=float, default=0.0)
    p.add_argument("--llm_anchor_reg_weight", type=float, default=0.0)
    p.add_argument("--marker_anchor_reg_weight", type=float, default=0.0)

    # Eval/log/ckpt
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument(
        "--checkpoint_policy",
        choices=["meaningful", "all"],
        default="meaningful",
        help="Checkpoint retention policy: keep meaningful top-k checkpoints or all periodic checkpoints.",
    )
    p.add_argument(
        "--save_first_n",
        type=int,
        default=3,
        help="When checkpoint_policy=meaningful, always keep the first N evaluated checkpoints.",
    )
    p.add_argument(
        "--save_top_k",
        type=int,
        default=10,
        help="When checkpoint_policy=meaningful, keep top-K evaluated checkpoints ranked by METEOR/BLEU/ROUGE/BERTScore/loss.",
    )
    p.add_argument(
        "--max_saved_checkpoints",
        type=int,
        default=10,
        help="Hard cap on the number of saved step checkpoints kept on disk.",
    )
    p.add_argument("--val_max_batches", type=int, default=30)
    p.add_argument("--gen_max_new_tokens", type=int, default=256)
    p.add_argument("--gen_min_new_tokens", type=int, default=0)
    p.add_argument("--gen_do_sample", action="store_true")
    p.add_argument("--gen_temperature", type=float, default=0.6)
    p.add_argument("--gen_top_p", type=float, default=0.95)
    p.add_argument("--certainty_percent_threshold", type=float, default=50.0)
    p.add_argument("--sample_count", type=int, default=16, help="Number of validation sample rows to log/save per eval.")
    p.add_argument("--bertscore_model_type", type=str, default="roberta-large")

    # Wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_project", type=str, default="PathoSynVLM")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")
    p.add_argument("--wandb_tags", type=str, default="histai,finetune,lora")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    dist_info = init_distributed_if_needed()
    is_distributed = bool(dist_info["is_distributed"])
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    local_rank = int(dist_info["local_rank"])
    is_main = rank == 0

    def rprint(*xs: Any) -> None:
        if is_main:
            print(*xs)

    out_dir = Path(args.output_dir)
    wandb_run = None
    train_sampler: Optional[DistributedSampler] = None

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        if is_main:
            (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        if is_distributed:
            dist.barrier()

        if is_distributed:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(int(args.seed) + int(rank))
        if is_main:
            print_runtime_device_info(device)
        else:
            print(f"[rank {rank}] using device {device}", flush=True)

        if bool(args.wandb) and is_main:
            import wandb

            tags = [x.strip() for x in str(args.wandb_tags).split(",") if x.strip()]
            run_name = str(args.wandb_run_name or "").strip() or None
            wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=run_name,
                group=(str(args.wandb_group).strip() or None),
                tags=(tags if tags else None),
                config=vars(args),
            )

        rprint("Loading tokenizer:", args.llm)
        tokenizer = AutoTokenizer.from_pretrained(args.llm, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        report_target_field_name = normalize_target_field_name(str(args.report_target_field))
        report_target_field_label = resolve_target_field_label(
            report_target_field_name,
            (str(args.report_target_label).strip() or None),
        )

        rprint("Building dataloaders...")
        t0 = time.perf_counter()
        mixed_data_info: Dict[str, Any] | None = None
        if float(args.alignment_mix_fraction) > 0.0:
            raise ValueError(
                "--alignment_mix_fraction is an experimental extension and is not part of "
                "the paper-default path. Leave it at 0.0."
            )
        else:
            train_dl, val_dl = create_train_val_dataloaders(
                metadata_standardized_json=Path(args.metadata_standardized_json),
                dataset_embeddings_root=Path(args.dataset_embeddings_root),
                tokenizer=tokenizer,
                feature_key=args.feature_key,
                patch_level=args.patch_level,
                diagnosis_field="standardized_diagnosis",
                conclusion_field=report_target_field_name,
                target_field_name=report_target_field_name,
                target_field_label=report_target_field_label,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                max_text_length=int(args.max_text_length),
                max_vision_tokens=int(args.max_vision_tokens),
                vision_token_dropout=float(args.vision_token_dropout),
                prompt_style=str(args.prompt_style),
                missing_policy=str(args.missing_policy),
                probe_h5_on_init=bool(args.probe_h5_on_init),
                val_size=args.val_size,
                split_seed=int(args.split_seed),
                train_shuffle=True,
            )

        if is_distributed:
            train_sampler = DistributedSampler(
                train_dl.dataset,
                num_replicas=int(world_size),
                rank=int(rank),
                shuffle=True,
                drop_last=False,
            )
            train_dl = DataLoader(
                train_dl.dataset,
                batch_size=int(args.batch_size),
                shuffle=False,
                sampler=train_sampler,
                num_workers=int(args.num_workers),
                collate_fn=train_dl.collate_fn,
            )

        rprint(f"Dataloader build time: {time.perf_counter() - t0:.1f}s")
        if mixed_data_info is not None and is_main:
            (out_dir / "mixed_data_info.json").write_text(json.dumps(mixed_data_info, indent=2), encoding="utf-8")
        if len(train_dl.dataset) == 0:
            raise RuntimeError("Train dataset is empty. Check metadata/embedding coverage and patch level.")

        torch_dtype = None
        if args.precision == "bf16" and device.type == "cuda":
            torch_dtype = torch.bfloat16
        elif args.precision == "fp16" and device.type == "cuda":
            torch_dtype = torch.float16

        rprint("Building model...")
        model_t0 = time.perf_counter()
        base_model = PathoSynVLM(
            llm_name_or_path=args.llm,
            vision_dim=int(args.vision_dim),
            feature_key=str(args.feature_key),
            torch_dtype=torch_dtype,
            use_wsi_markers=bool(args.use_wsi_markers),
            use_index_emb=bool(args.use_wsi_index_emb),
        ).to(device)
        rprint(f"Model build time: {time.perf_counter() - model_t0:.1f}s")
        if device.type == "cuda" and is_main:
            print(
                "[gpu] after model load "
                f"mem_alloc_mb={torch.cuda.memory_allocated(device) / (1024 ** 2):.1f} "
                f"mem_reserved_mb={torch.cuda.memory_reserved(device) / (1024 ** 2):.1f}"
            )

        aligner_load_info: Dict[str, Any]
        if bool(args.skip_aligner_init):
            rprint("Skipping aligner init load; keeping random aligner initialization.")
            aligner_load_info = {
                "loaded": False,
                "status": "skipped",
                "reason": "skip_aligner_init",
                "checkpoint_path": "",
            }
        else:
            if not str(args.aligner_init).strip():
                raise ValueError("--aligner_init is required unless --skip_aligner_init is set.")
            rprint("Loading aligner init:", args.aligner_init)
            aligner_load_info = load_aligner_from_checkpoint(
                base_model,
                checkpoint_path=args.aligner_init,
                strict=bool(args.strict_aligner_load),
                map_location="cpu",
            )
        if is_main:
            (out_dir / "aligner_init_info.json").write_text(json.dumps(aligner_load_info, indent=2), encoding="utf-8")

        for _, p in base_model.named_parameters():
            p.requires_grad = False

        llm_train_scope = _resolve_llm_train_scope(args)
        if bool(args.use_lora):
            rprint("Applying LoRA...")
            target_modules = [x.strip() for x in str(args.lora_target).split(",") if x.strip()]
            if not target_modules:
                raise ValueError("LoRA is enabled but --lora_target resolved to an empty module list.")
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(args.lora_r),
                lora_alpha=int(args.lora_alpha),
                lora_dropout=float(args.lora_dropout),
                target_modules=target_modules,
                bias="none",
            )
            base_model.llm = get_peft_model(base_model.llm, lora_cfg)

            for n, p in base_model.llm.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
        else:
            rprint("LoRA disabled; using base LLM weights directly.")

        if hasattr(base_model, "aligner") and (not bool(args.freeze_aligner)):
            for p in base_model.aligner.parameters():
                p.requires_grad = True
        elif hasattr(base_model, "aligner") and bool(args.freeze_aligner):
            rprint("Freezing aligner (MLP) module.")

        if bool(getattr(base_model, "use_wsi_markers", False)):
            if hasattr(base_model, "wsi_index_emb"):
                for p in base_model.wsi_index_emb.parameters():
                    p.requires_grad = True
            if hasattr(base_model, "wsi_sep") and isinstance(base_model.wsi_sep, torch.Tensor):
                base_model.wsi_sep.requires_grad_(True)

        llm_scope_info = _apply_llm_train_scope(
            base_model,
            scope=llm_train_scope,
            use_lora=bool(args.use_lora),
        )
        if llm_train_scope == "none" and not bool(args.use_lora):
            rprint("LLM base is frozen.")
        else:
            rprint(f"LLM train scope: {llm_scope_info}")

        n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in base_model.parameters())
        rprint(f"Trainable params: {n_trainable} / {n_total} ({100.0 * n_trainable / max(1, n_total):.2f}%)")

        if bool(args.gradient_checkpoint):
            try:
                base_model.llm.gradient_checkpointing_enable()
                base_model.llm.config.use_cache = False
            except Exception as e:
                rprint(f"[warn] failed to enable gradient checkpointing: {e}")

        param_groups = _param_groups(
            base_model,
            lr_lora=float(args.lr),
            lr_aligner=float(args.aligner_lr),
            lr_llm_base=float(args.llm_base_lr),
            weight_decay=float(args.weight_decay),
        )
        if not param_groups:
            raise RuntimeError("No trainable parameters found after applying trainability policy.")

        anchor_regularizer = AnchorRegularizer(
            base_model,
            llm_weight=float(args.llm_anchor_reg_weight),
            aligner_weight=float(args.aligner_anchor_reg_weight),
            marker_weight=float(args.marker_anchor_reg_weight),
        )
        if anchor_regularizer.enabled:
            rprint(f"Anchor regularization: {anchor_regularizer.summary()}")
        else:
            rprint("Anchor regularization: disabled")

        optimizer = _build_optimizer(
            optimizer_name=str(args.optimizer),
            param_groups=param_groups,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        rprint(
            f"Optimizer: {str(args.optimizer).strip().lower()} | "
            f"use_lora={bool(args.use_lora)} | llm_train_scope={llm_train_scope}"
        )

        updates_per_epoch = math.ceil(len(train_dl) / max(1, int(args.grad_accum)))
        total_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(args.epochs) * updates_per_epoch
        warmup_steps = int(total_steps * float(args.warmup_ratio))
        if float(args.warmup_ratio) > 0 and warmup_steps == 0:
            warmup_steps = 1

        scheduler = build_scheduler(
            scheduler_name=str(args.scheduler),
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        rouge_metric = None
        meteor_metric = None
        bleu_metric = None
        bertscore_metric = None
        if is_main:
            try:
                import evaluate

                rouge_metric = evaluate.load("rouge")
                meteor_metric = evaluate.load("meteor")
                bleu_metric = evaluate.load("sacrebleu")
                bertscore_metric = evaluate.load("bertscore")
            except Exception as e:
                rprint(f"[warn] text metrics partially disabled: {e}")

        if is_distributed:
            model = DDP(
                base_model,
                device_ids=[int(local_rank)],
                output_device=int(local_rank),
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            model = base_model
        model_for_eval = base_model
        ema_tracker: ExponentialMovingAverage | None = None
        if bool(args.ema):
            ema_tracker = ExponentialMovingAverage(base_model, decay=float(args.ema_decay))
            rprint(
                "[ema] enabled "
                f"decay={float(args.ema_decay):.6f} "
                f"tracked_tensors={ema_tracker.tracked_tensors} "
                f"tracked_numel={ema_tracker.tracked_numel} "
                f"shadow_mem_gb={float(ema_tracker.shadow_bytes) / (1024 ** 3):.2f}"
            )
        else:
            rprint("[ema] disabled")

        @contextmanager
        def ema_eval_scope():
            if ema_tracker is None:
                yield
                return
            ema_tracker.swap_parameters()
            try:
                yield
            finally:
                if ema_tracker.is_swapped:
                    ema_tracker.swap_parameters()

        if wandb_run is not None:
            wandb_run.config.update(
                {
                    "trainable_params": int(n_trainable),
                    "total_params": int(n_total),
                    "trainable_pct": float(100.0 * n_trainable / max(1, n_total)),
                    "updates_per_epoch": int(updates_per_epoch),
                    "total_steps": int(total_steps),
                    "warmup_steps": int(warmup_steps),
                    "aligner_init_info": aligner_load_info,
                    "train_size": int(len(train_dl.dataset)),
                    "val_size": int(len(val_dl.dataset)),
                    "world_size": int(world_size),
                    "checkpoint_policy": str(args.checkpoint_policy),
                    "save_first_n": int(args.save_first_n),
                    "save_top_k": int(args.save_top_k),
                    "max_saved_checkpoints": int(args.max_saved_checkpoints),
                    "ema_enabled": bool(args.ema),
                    "ema_decay": float(args.ema_decay),
                    "ema_tracked_tensors": (int(ema_tracker.tracked_tensors) if ema_tracker is not None else 0),
                    "ema_tracked_numel": (int(ema_tracker.tracked_numel) if ema_tracker is not None else 0),
                    "llm_train_scope": str(llm_train_scope),
                    "llm_scope_info": llm_scope_info,
                    "report_target_field_name": report_target_field_name,
                    "report_target_field_label": report_target_field_label,
                    "alignment_mix_fraction": float(args.alignment_mix_fraction),
                    "alignment_dataset_selection": str(args.alignment_dataset_selection),
                    "anchor_regularizer": anchor_regularizer.summary(),
                },
                allow_val_change=True,
            )
            if mixed_data_info is not None:
                wandb_run.config.update({"mixed_data_info": mixed_data_info}, allow_val_change=True)

        use_fp16 = bool(args.precision == "fp16" and device.type == "cuda")
        use_bf16 = bool(args.precision == "bf16" and device.type == "cuda")
        has_fp16_trainable_params = any(
            p.requires_grad and p.is_floating_point() and p.dtype == torch.float16 for p in base_model.parameters()
        )
        scaler_enabled = bool(use_fp16 and (not has_fp16_trainable_params))
        if use_fp16 and has_fp16_trainable_params:
            rprint(
                "[warn] fp16 trainable parameters detected; disabling GradScaler to avoid "
                "unscale errors on pure-fp16 grads."
            )
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        use_fp16_scaler = bool(use_fp16 and scaler.is_enabled())

        global_step = 0
        optimizer.zero_grad(set_to_none=True)

        best_val_loss = float("inf")
        best_step = -1
        best_epoch = -1
        best_metrics: Dict[str, Any] = {}
        best_source = ""
        epoch_history: List[int] = []
        train_epoch_loss_history: List[float] = []
        val_loss_history: List[float] = []
        loss_history_json_path: Path | None = None
        loss_curve_png_path: Path | None = None
        checkpoint_policy = str(args.checkpoint_policy).strip().lower()
        save_first_n = max(0, int(args.save_first_n))
        save_top_k = max(0, int(args.save_top_k))
        max_saved_checkpoints = max(1, int(args.max_saved_checkpoints))
        checkpoint_candidates_by_step: Dict[int, Dict[str, Any]] = {}
        checkpoint_seen_steps: List[int] = []
        saved_checkpoint_steps: set[int] = set()

        def write_checkpoint_retention_state(keep_steps: set[int]) -> None:
            if not is_main:
                return
            candidates_ordered = [checkpoint_candidates_by_step[s] for s in checkpoint_seen_steps if s in checkpoint_candidates_by_step]
            payload = {
                "checkpoint_policy": str(checkpoint_policy),
                "save_first_n": int(save_first_n),
                "save_top_k": int(save_top_k),
                "max_saved_checkpoints": int(max_saved_checkpoints),
                "best_step": int(best_step),
                "best_val_loss": (float(best_val_loss) if math.isfinite(best_val_loss) else None),
                "kept_steps": sorted(int(x) for x in keep_steps),
                "saved_steps": sorted(int(x) for x in saved_checkpoint_steps),
                "candidates_seen": int(len(candidates_ordered)),
                "candidates": candidates_ordered,
            }
            (out_dir / "checkpoint_retention_state.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

        def compute_checkpoint_keep_steps() -> set[int]:
            ordered_keep: list[int] = []

            def add_steps(steps: list[int]) -> None:
                for step_i in steps:
                    if step_i in ordered_keep:
                        continue
                    ordered_keep.append(int(step_i))

            if int(best_step) >= 0:
                add_steps([int(best_step)])

            if str(checkpoint_policy) == "all":
                add_steps(list(reversed([int(s) for s in checkpoint_seen_steps])))
            else:
                if int(save_top_k) > 0 and checkpoint_candidates_by_step:
                    ranked = sorted(
                        checkpoint_candidates_by_step.values(),
                        key=lambda c: tuple(c.get("rank_tuple", [])),
                        reverse=True,
                    )
                    add_steps([int(c["step"]) for c in ranked[: int(save_top_k)]])
                if int(save_first_n) > 0:
                    add_steps([int(s) for s in checkpoint_seen_steps[: int(save_first_n)]])

            if int(max_saved_checkpoints) > 0:
                ordered_keep = ordered_keep[: int(max_saved_checkpoints)]

            return set(int(x) for x in ordered_keep)

        def maybe_save_meaningful_checkpoint(metrics: Dict[str, Any], *, step: int, epoch_idx: int, source: str) -> None:
            if not is_main:
                return

            step_i = int(step)
            rank_tuple, rank_metrics = build_checkpoint_rank(
                metrics,
                report_target_field_name=report_target_field_name,
            )

            existing = checkpoint_candidates_by_step.get(step_i)
            if existing is None:
                candidate: Dict[str, Any] = {
                    "step": int(step_i),
                    "epoch": int(epoch_idx),
                    "source": str(source),
                    "sources": [str(source)],
                    "val_loss": _metric_float(metrics, "val_loss", float("inf")),
                    "rank_metrics": rank_metrics,
                    "rank_tuple": [float(x) for x in rank_tuple],
                }
                checkpoint_candidates_by_step[step_i] = candidate
                checkpoint_seen_steps.append(step_i)
            else:
                existing_sources = {str(x) for x in existing.get("sources", []) if str(x)}
                existing_sources.add(str(source))
                existing["sources"] = sorted(existing_sources)
                existing_rank = tuple(float(x) for x in existing.get("rank_tuple", []))
                if rank_tuple > existing_rank:
                    existing["epoch"] = int(epoch_idx)
                    existing["source"] = str(source)
                    existing["val_loss"] = _metric_float(metrics, "val_loss", float("inf"))
                    existing["rank_metrics"] = rank_metrics
                    existing["rank_tuple"] = [float(x) for x in rank_tuple]

            keep_steps = compute_checkpoint_keep_steps()
            if step_i in keep_steps and step_i not in saved_checkpoint_steps:
                with ema_eval_scope():
                    save_checkpoint(
                        save_dir=out_dir,
                        model=model_for_eval,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step_i,
                        tokenizer=tokenizer,
                        extra={
                            "epoch": int(epoch_idx),
                            "is_best": bool(step_i == int(best_step)),
                            "source": str(source),
                            "metrics": metrics,
                            "checkpoint_policy": str(checkpoint_policy),
                            "rank_metrics": rank_metrics,
                            "ema_enabled": bool(ema_tracker is not None),
                        },
                        save_full_llm_state=False,
                        include_optimizer_state=False,
                        include_scheduler_state=False,
                    )
                saved_checkpoint_steps.add(step_i)
                rprint(f"[ckpt] kept step={step_i} source={source}")

            stale_steps = [int(s) for s in sorted(saved_checkpoint_steps) if int(s) not in keep_steps]
            for stale_step in stale_steps:
                if remove_checkpoint_step_artifacts(out_dir, stale_step):
                    rprint(f"[ckpt] pruned step={stale_step}")
                saved_checkpoint_steps.discard(int(stale_step))

            if int(best_step) in saved_checkpoint_steps:
                update_best_checkpoint_aliases(out_dir, step=int(best_step))

            write_checkpoint_retention_state(keep_steps)

        def maybe_update_best(metrics: Dict[str, Any], *, step: int, epoch_idx: int, source: str) -> None:
            nonlocal best_val_loss, best_step, best_epoch, best_metrics, best_source
            if not is_main:
                return
            cur_val_loss = float(metrics.get("val_loss", float("inf")))
            if not math.isfinite(cur_val_loss):
                return
            if cur_val_loss >= float(best_val_loss):
                return

            best_val_loss = cur_val_loss
            best_step = int(step)
            best_epoch = int(epoch_idx)
            best_source = str(source)
            best_metrics = dict(metrics)

            best_summary = {
                "best_step": int(best_step),
                "best_epoch": int(best_epoch),
                "best_source": best_source,
                "best_val_loss": float(best_val_loss),
                "metrics": best_metrics,
            }
            (out_dir / "best_checkpoint_summary.json").write_text(
                json.dumps(best_summary, indent=2),
                encoding="utf-8",
            )
            rprint(
                f"[best] source={best_source} epoch={best_epoch} step={best_step} "
                f"val_loss={best_val_loss:.4f}"
            )

        rprint("Starting training...")
        rprint(f"train_size={len(train_dl.dataset)} val_size={len(val_dl.dataset)}")
        rprint(f"updates_per_epoch={updates_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps}")

        model.train()
        for epoch in range(int(args.epochs)):
            epoch_idx = int(epoch + 1)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_idx)

            epoch_started = time.perf_counter()
            epoch_losses: List[float] = []
            samples_seen_epoch = 0
            vision_tokens_epoch = 0
            text_tokens_epoch = 0
            target_tokens_epoch = 0
            wsi_seen_epoch = 0

            log_window_started = time.perf_counter()
            window_updates = 0
            window_loss_sum = 0.0
            window_task_loss_sum = 0.0
            window_reg_loss_sum = 0.0
            window_samples = 0
            window_vision_tokens = 0
            window_text_tokens = 0
            window_target_tokens = 0
            window_wsi_count = 0

            if device.type == "cuda":
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    pass

            pbar = tqdm(total=len(train_dl), desc=f"epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True) if is_main else None

            for step_in_epoch, batch in enumerate(train_dl, start=1):
                if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
                    break

                batch = to_device(batch, device)
                bsz = int(batch["vision_embeddings"].shape[0]) if "vision_embeddings" in batch else 0
                vision_tokens = int(batch["vision_attention_mask"].sum().item()) if "vision_attention_mask" in batch else 0
                text_tokens = int(batch["attention_mask"].sum().item()) if "attention_mask" in batch else 0
                target_tokens = int((batch["labels"] != -100).sum().item()) if "labels" in batch else 0
                wsi_count = int((batch["wsi_patch_counts"] > 0).sum().item()) if "wsi_patch_counts" in batch else 0

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
                        raise RuntimeError("Model did not return loss")
                    task_loss = out.loss
                    reg_loss, reg_loss_metrics = anchor_regularizer.compute()
                    raw_loss = task_loss if reg_loss is None else (task_loss + reg_loss)
                    loss = raw_loss / max(1, int(args.grad_accum))

                if use_fp16_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                do_update = ((step_in_epoch % int(args.grad_accum)) == 0) or (step_in_epoch == len(train_dl))
                if do_update:
                    if float(args.clip_grad_norm) > 0:
                        if use_fp16_scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))

                    if use_fp16_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if ema_tracker is not None:
                        ema_tracker.update()

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    task_loss_value = float(task_loss.detach().float().cpu().item())
                    reg_loss_value = float(reg_loss.detach().float().cpu().item()) if reg_loss is not None else 0.0
                    raw_loss_value = float(raw_loss.detach().float().cpu().item())
                    epoch_losses.append(raw_loss_value)
                    window_updates += 1
                    window_loss_sum += raw_loss_value
                    window_task_loss_sum += task_loss_value
                    window_reg_loss_sum += reg_loss_value

                    if global_step % int(args.log_every) == 0:
                        last_lrs = scheduler.get_last_lr() if scheduler is not None else [pg["lr"] for pg in optimizer.param_groups]
                        lr_by_group: Dict[str, float] = {}
                        for gi, (pg, lr_v) in enumerate(zip(optimizer.param_groups, last_lrs)):
                            lr_by_group[str(pg.get("name") or f"group{gi}")] = float(lr_v)

                        elapsed_window = max(1e-9, time.perf_counter() - log_window_started)
                        if is_main:
                            loss_avg_window = float(window_loss_sum / max(1, window_updates))
                            task_loss_avg_window = float(window_task_loss_sum / max(1, window_updates))
                            reg_loss_avg_window = float(window_reg_loss_sum / max(1, window_updates))
                            log_data: Dict[str, Any] = {
                                "train/loss": float(raw_loss_value),
                                "train/task_loss": float(task_loss_value),
                                "train/reg_loss": float(reg_loss_value),
                                "train/loss_avg_window": float(loss_avg_window),
                                "train/task_loss_avg_window": float(task_loss_avg_window),
                                "train/reg_loss_avg_window": float(reg_loss_avg_window),
                                "train/step": int(global_step),
                                "train/epoch": float(epoch + (step_in_epoch / max(1, len(train_dl)))),
                                "train/lr_lora": float(lr_by_group.get("lora", 0.0)),
                                "train/lr_aligner": float(lr_by_group.get("aligner", 0.0)),
                                "train/lr_markers": float(lr_by_group.get("markers", 0.0)),
                                "train/lr_llm_base": float(lr_by_group.get("llm_base", 0.0)),
                                "train/window_samples": float(window_samples),
                                "train/window_vision_tokens": float(window_vision_tokens),
                                "train/window_text_tokens": float(window_text_tokens),
                                "train/window_target_tokens": float(window_target_tokens),
                                "train/window_wsi_count": float(window_wsi_count),
                                "train/samples_per_sec": float(window_samples / elapsed_window),
                                "train/vision_tokens_per_sec": float(window_vision_tokens / elapsed_window),
                                "train/text_tokens_per_sec": float(window_text_tokens / elapsed_window),
                                "train/target_tokens_per_sec": float(window_target_tokens / elapsed_window),
                                "train/updates_per_sec": float(window_updates / elapsed_window),
                                "train/avg_vision_tokens_per_sample_window": float(window_vision_tokens / max(1, window_samples)),
                                "train/avg_text_tokens_per_sample_window": float(window_text_tokens / max(1, window_samples)),
                                "train/avg_target_tokens_per_sample_window": float(window_target_tokens / max(1, window_samples)),
                                "train/avg_wsi_per_sample_window": float(window_wsi_count / max(1, window_samples)),
                            }
                            if device.type == "cuda":
                                log_data["train/gpu_mem_alloc_mb"] = float(torch.cuda.memory_allocated(device) / (1024 ** 2))
                                log_data["train/gpu_mem_reserved_mb"] = float(torch.cuda.memory_reserved(device) / (1024 ** 2))
                                log_data["train/gpu_max_mem_alloc_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                            for reg_name, reg_value in reg_loss_metrics.items():
                                log_data[f"train/{reg_name}"] = float(reg_value)
                            rprint(
                                f"[step {global_step}] loss={log_data['train/loss']:.4f} "
                                f"task_loss={log_data['train/task_loss']:.4f} "
                                f"reg_loss={log_data['train/reg_loss']:.4f} "
                                f"loss_avg_window={log_data['train/loss_avg_window']:.4f} "
                                f"lr_lora={log_data['train/lr_lora']:.3e} "
                                f"lr_aligner={log_data['train/lr_aligner']:.3e} "
                                f"lr_llm_base={log_data['train/lr_llm_base']:.3e}"
                            )
                            if wandb_run is not None:
                                wandb_run.log(log_data, step=int(global_step))

                        log_window_started = time.perf_counter()
                        window_updates = 0
                        window_loss_sum = 0.0
                        window_task_loss_sum = 0.0
                        window_reg_loss_sum = 0.0
                        window_samples = 0
                        window_vision_tokens = 0
                        window_text_tokens = 0
                        window_target_tokens = 0
                        window_wsi_count = 0

                    if global_step % int(args.eval_every) == 0:
                        if is_distributed:
                            dist.barrier()
                        if is_main:
                            rprint(f"[step {global_step}] running validation...")
                            with ema_eval_scope():
                                metrics, _ = run_validation(
                                    model=model_for_eval,
                                    val_dl=val_dl,
                                    device=device,
                                    tokenizer=tokenizer,
                                    max_batches=int(args.val_max_batches),
                                    gen_max_new_tokens=int(args.gen_max_new_tokens),
                                    gen_min_new_tokens=int(args.gen_min_new_tokens),
                                    gen_do_sample=bool(args.gen_do_sample),
                                    gen_temperature=float(args.gen_temperature),
                                    gen_top_p=float(args.gen_top_p),
                                    certainty_percent_threshold=float(args.certainty_percent_threshold),
                                    rouge_metric=rouge_metric,
                                    meteor_metric=meteor_metric,
                                    bleu_metric=bleu_metric,
                                    bertscore_metric=bertscore_metric,
                                    bertscore_model_type=str(args.bertscore_model_type),
                                    report_target_field_name=report_target_field_name,
                                    report_target_field_label=report_target_field_label,
                                    wandb_run=wandb_run,
                                    global_step=int(global_step),
                                    log_n_samples=int(args.sample_count),
                                )
                            msg = f"[step {global_step}] val_loss={metrics['val_loss']:.4f}"
                            if "val_bertscore_f1" in metrics:
                                msg += f" bertscore_f1={float(metrics['val_bertscore_f1']):.4f}"
                            rprint(msg)
                            maybe_update_best(metrics, step=int(global_step), epoch_idx=epoch_idx, source="step_eval")
                            maybe_save_meaningful_checkpoint(
                                metrics,
                                step=int(global_step),
                                epoch_idx=epoch_idx,
                                source="step_eval",
                            )
                        if is_distributed:
                            dist.barrier()

                    if (
                        str(checkpoint_policy) == "all"
                        and int(args.save_every) > 0
                        and global_step % int(args.save_every) == 0
                        and is_main
                    ):
                        with ema_eval_scope():
                            save_checkpoint(
                                save_dir=out_dir,
                                model=model_for_eval,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                step=global_step,
                                tokenizer=tokenizer,
                                extra={
                                    "epoch": int(epoch),
                                    "is_best": False,
                                    "ema_enabled": bool(ema_tracker is not None),
                                },
                                save_full_llm_state=False,
                                include_optimizer_state=False,
                                include_scheduler_state=False,
                            )
                        rprint(f"[step {global_step}] checkpoint saved")

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{float(raw_loss.detach().float().cpu().item()):.4f}"})

            if pbar is not None:
                pbar.close()

            if is_distributed:
                dist.barrier()

            epoch_time_sec = float(time.perf_counter() - epoch_started)
            train_loss_epoch = float(sum(epoch_losses) / max(1, len(epoch_losses)))

            if is_main:
                rprint(f"[epoch {epoch_idx}] running end-of-epoch validation...")
                with ema_eval_scope():
                    epoch_val_metrics, epoch_val_rows = run_validation(
                        model=model_for_eval,
                        val_dl=val_dl,
                        device=device,
                        tokenizer=tokenizer,
                        max_batches=int(args.val_max_batches),
                        gen_max_new_tokens=int(args.gen_max_new_tokens),
                        gen_min_new_tokens=int(args.gen_min_new_tokens),
                        gen_do_sample=bool(args.gen_do_sample),
                        gen_temperature=float(args.gen_temperature),
                        gen_top_p=float(args.gen_top_p),
                        certainty_percent_threshold=float(args.certainty_percent_threshold),
                        rouge_metric=rouge_metric,
                        meteor_metric=meteor_metric,
                        bleu_metric=bleu_metric,
                        bertscore_metric=bertscore_metric,
                        bertscore_model_type=str(args.bertscore_model_type),
                        report_target_field_name=report_target_field_name,
                        report_target_field_label=report_target_field_label,
                        wandb_run=wandb_run,
                        global_step=int(global_step),
                        log_n_samples=int(args.sample_count),
                    )
                msg = (
                    f"[epoch {epoch_idx}] train_loss={train_loss_epoch:.4f} "
                    f"val_loss={float(epoch_val_metrics.get('val_loss', 0.0)):.4f}"
                )
                if "val_bertscore_f1" in epoch_val_metrics:
                    msg += f" bertscore_f1={float(epoch_val_metrics['val_bertscore_f1']):.4f}"
                rprint(msg)

                maybe_update_best(epoch_val_metrics, step=int(global_step), epoch_idx=epoch_idx, source="epoch_end")
                maybe_save_meaningful_checkpoint(
                    epoch_val_metrics,
                    step=int(global_step),
                    epoch_idx=epoch_idx,
                    source="epoch_end",
                )

                epoch_history.append(epoch_idx)
                train_epoch_loss_history.append(float(train_loss_epoch))
                val_loss_history.append(float(epoch_val_metrics.get("val_loss", 0.0)))
                loss_history_json_path = save_loss_history_json(
                    output_dir=out_dir,
                    epochs=epoch_history,
                    train_epoch_losses=train_epoch_loss_history,
                    val_losses=val_loss_history,
                )
                loss_curve_png_path = save_loss_curve_plot(
                    output_dir=out_dir,
                    epochs=epoch_history,
                    train_epoch_losses=train_epoch_loss_history,
                    val_losses=val_loss_history,
                )
                if wandb_run is not None:
                    log_epoch_loss_curve_to_wandb(
                        wandb_run=wandb_run,
                        epochs=epoch_history,
                        train_epoch_losses=train_epoch_loss_history,
                        val_losses=val_loss_history,
                        step=int(global_step),
                    )

                epoch_log: Dict[str, Any] = {
                    "epoch": int(epoch_idx),
                    "global_step": int(global_step),
                    "train/epoch_loss": float(train_loss_epoch),
                    "train/epoch_time_sec": float(epoch_time_sec),
                    "train/epoch_updates": float(len(epoch_losses)),
                    "train/epoch_samples": float(samples_seen_epoch),
                    "train/epoch_vision_tokens": float(vision_tokens_epoch),
                    "train/epoch_text_tokens": float(text_tokens_epoch),
                    "train/epoch_target_tokens": float(target_tokens_epoch),
                    "train/epoch_wsi_count": float(wsi_seen_epoch),
                    "train/avg_vision_tokens_per_sample_epoch": float(vision_tokens_epoch / max(1, samples_seen_epoch)),
                    "train/avg_text_tokens_per_sample_epoch": float(text_tokens_epoch / max(1, samples_seen_epoch)),
                    "train/avg_target_tokens_per_sample_epoch": float(target_tokens_epoch / max(1, samples_seen_epoch)),
                    "train/avg_wsi_per_sample_epoch": float(wsi_seen_epoch / max(1, samples_seen_epoch)),
                    "train/epoch_samples_per_sec": float(samples_seen_epoch / max(1e-9, epoch_time_sec)),
                    "best/val_loss_so_far": float(best_val_loss) if math.isfinite(best_val_loss) else None,
                    "best/step": int(best_step),
                    "best/epoch": int(best_epoch),
                }
                if device.type == "cuda":
                    epoch_log["train/epoch_gpu_max_mem_alloc_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

                for k, v in epoch_val_metrics.items():
                    if not isinstance(v, (int, float)):
                        continue
                    if k.startswith("val/"):
                        epoch_log[k] = float(v)
                    elif k.startswith("val_"):
                        epoch_log[f"val/{k[len('val_'):]}"] = float(v)
                    else:
                        epoch_log[f"val/{k}"] = float(v)

                if wandb_run is not None:
                    wandb_run.log(epoch_log, step=int(global_step))

                sample_dump = {
                    "epoch": int(epoch_idx),
                    "global_step": int(global_step),
                    "train_epoch_loss": float(train_loss_epoch),
                    "val_metrics": epoch_val_metrics,
                    "epoch_log": epoch_log,
                    "val_rows": epoch_val_rows,
                    "loss_history": [
                        {
                            "epoch": int(e),
                            "train_epoch_loss": float(tr),
                            "val_loss": float(va),
                        }
                        for e, tr, va in zip(epoch_history, train_epoch_loss_history, val_loss_history)
                    ],
                }
                (out_dir / f"epoch_{epoch_idx:03d}_samples_and_metrics.json").write_text(
                    json.dumps(sample_dump, indent=2),
                    encoding="utf-8",
                )

            if is_distributed:
                dist.barrier()

            if int(args.max_steps) > 0 and global_step >= int(args.max_steps):
                break

        if is_main:
            final_ckpt = None
            should_save_final = (
                int(global_step) in saved_checkpoint_steps
                or int(global_step) == int(best_step)
                or len(saved_checkpoint_steps) < int(max_saved_checkpoints)
            )
            if should_save_final:
                with ema_eval_scope():
                    final_ckpt = save_checkpoint(
                        save_dir=out_dir,
                        model=model_for_eval,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        tokenizer=tokenizer,
                        extra={
                            "final": True,
                            "best_step": int(best_step),
                            "best_epoch": int(best_epoch),
                            "best_source": str(best_source),
                            "best_val_loss": (float(best_val_loss) if math.isfinite(best_val_loss) else None),
                            "ema_enabled": bool(ema_tracker is not None),
                            "ema_decay": (float(args.ema_decay) if ema_tracker is not None else None),
                        },
                        save_full_llm_state=False,
                        include_optimizer_state=False,
                        include_scheduler_state=False,
                    )
                if int(global_step) not in saved_checkpoint_steps:
                    saved_checkpoint_steps.add(int(global_step))
                rprint(f"[final] saved checkpoint: {final_ckpt}")
            else:
                rprint("[final] skipped extra final checkpoint to respect max_saved_checkpoints cap")
            if int(best_step) >= 0 and (out_dir / f"trainer_state_step_{int(best_step)}.pt").exists():
                update_best_checkpoint_aliases(out_dir, step=int(best_step))
            write_checkpoint_retention_state(set(saved_checkpoint_steps))

            final_summary = {
                "final_step": int(global_step),
                "best_step": int(best_step),
                "best_epoch": int(best_epoch),
                "best_source": str(best_source),
                "best_val_loss": (float(best_val_loss) if math.isfinite(best_val_loss) else None),
                "best_metrics": best_metrics,
                "aligner_init_info": aligner_load_info,
                "train_size": int(len(train_dl.dataset)),
                "val_size": int(len(val_dl.dataset)),
                "world_size": int(world_size),
                "checkpoint_policy": str(checkpoint_policy),
                "save_first_n": int(save_first_n),
                "save_top_k": int(save_top_k),
                "max_saved_checkpoints": int(max_saved_checkpoints),
                "ema_enabled": bool(ema_tracker is not None),
                "ema_decay": (float(args.ema_decay) if ema_tracker is not None else None),
                "ema_tracked_tensors": (int(ema_tracker.tracked_tensors) if ema_tracker is not None else 0),
                "ema_tracked_numel": (int(ema_tracker.tracked_numel) if ema_tracker is not None else 0),
                "llm_train_scope": str(llm_train_scope),
                "llm_scope_info": llm_scope_info,
                "report_target_field_name": report_target_field_name,
                "report_target_field_label": report_target_field_label,
                "mixed_data_info": mixed_data_info,
                "best_llm_alias": str(out_dir / "best_llm"),
                "best_trainer_state_alias": str(out_dir / "best_trainer_state.pt"),
                "saved_checkpoint_steps": sorted(int(x) for x in saved_checkpoint_steps),
                "checkpoint_retention_state": str(out_dir / "checkpoint_retention_state.json"),
                "loss_history_json": (str(loss_history_json_path) if loss_history_json_path is not None else None),
                "loss_curve_png": (str(loss_curve_png_path) if loss_curve_png_path is not None else None),
                "loss_history": [
                    {
                        "epoch": int(e),
                        "train_epoch_loss": float(tr),
                        "val_loss": float(va),
                    }
                    for e, tr, va in zip(epoch_history, train_epoch_loss_history, val_loss_history)
                ],
            }
            (out_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "final/step": int(global_step),
                        "final/best_step": int(best_step),
                        "final/best_val_loss": (float(best_val_loss) if math.isfinite(best_val_loss) else None),
                    },
                    step=int(global_step),
                )
                wandb_run.finish()
    finally:
        cleanup_distributed_if_needed(is_distributed)


if __name__ == "__main__":
    main()
