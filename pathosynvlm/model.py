"""
VLM model used for HistAI finetuning.

- Vision inputs are precomputed patch embeddings (B, N, D).
- Vision embeddings are projected into LLM hidden size via `VisionAligner`.
- Projected vision tokens are prepended before text tokens.
- Includes helpers to load aligner initialization from prior alignment checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class VLMOutputs:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    llm_outputs: Any


class VisionAligner(nn.Module):
    """2-layer MLP to map patch embedding dim -> LLM hidden size."""

    def __init__(
        self,
        vision_dim: int,
        hidden_size: int,
        mlp_hidden_mult: int = 4,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        inner = hidden_size * mlp_hidden_mult
        self.pre_ln = nn.LayerNorm(vision_dim) if use_layernorm else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_size),
        )
        self.post_ln = nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_ln(x)
        x = self.mlp(x)
        x = self.post_ln(x)
        return x


class PathoSynVLM(nn.Module):
    """
    Minimal VLM = [VisionAligner] + [CausalLM]

    Forward:
      - Concatenate vision prefix embeddings with text token embeddings.
      - Concatenate attention masks.
      - Pad labels with -100 for vision prefix positions.
    """

    def __init__(
        self,
        llm_name_or_path: str,
        vision_dim: int,
        feature_key: str = "conch_v15",
        aligner_mlp_hidden_mult: int = 4,
        aligner_dropout: float = 0.0,
        aligner_use_layernorm: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        use_wsi_markers: bool = True,
        use_index_emb: bool = True,
    ) -> None:
        super().__init__()

        self.llm_name_or_path = llm_name_or_path
        self.vision_dim = vision_dim
        self.feature_key = feature_key
        self.use_index_emb = use_index_emb

        cfg = AutoConfig.from_pretrained(llm_name_or_path)
        hidden_size = getattr(cfg, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(cfg, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Could not infer LLM hidden size from config.")

        model_kwargs: Dict[str, Any] = {}
        if torch_dtype is not None:
            model_kwargs["dtype"] = torch_dtype
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path, **model_kwargs)

        self.aligner = VisionAligner(
            vision_dim=vision_dim,
            hidden_size=hidden_size,
            mlp_hidden_mult=aligner_mlp_hidden_mult,
            dropout=aligner_dropout,
            use_layernorm=aligner_use_layernorm,
        )
        self.aligner.to(dtype=self.llm.dtype)

        self.use_wsi_markers = use_wsi_markers
        self.max_wsis = 32
        hidden_size = self.llm.config.hidden_size

        if self.use_index_emb:
            self.wsi_index_emb = nn.Embedding(self.max_wsis, hidden_size)
        self.wsi_sep = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.wsi_sep, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _build_prefix_inputs(
        self,
        vision_embeddings: Optional[torch.Tensor],
        vision_attention_mask: Optional[torch.Tensor],
        wsi_patch_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if vision_embeddings is None:
            return None, None

        if vision_embeddings.ndim != 3:
            raise ValueError(f"vision_embeddings must be (B, N, D), got {vision_embeddings.shape}")

        bsz, n_tok, _ = vision_embeddings.shape

        if vision_attention_mask is None:
            vision_attention_mask = torch.ones((bsz, n_tok), dtype=torch.bool, device=vision_embeddings.device)
        else:
            vision_attention_mask = vision_attention_mask.to(dtype=torch.bool)

        vision_embeddings = vision_embeddings.to(dtype=self.llm.dtype)
        proj = self.aligner(vision_embeddings)

        if (not getattr(self, "use_wsi_markers", False)) or (wsi_patch_counts is None):
            return proj, vision_attention_mask

        wsi_patch_counts = wsi_patch_counts.to(device=proj.device)

        per_sample_embeds = []
        per_sample_masks = []
        max_len = 0

        for b in range(bsz):
            counts = wsi_patch_counts[b]
            counts = counts[counts > 0].to(dtype=torch.long)
            if counts.numel() == 0:
                counts = torch.tensor([int(vision_attention_mask[b].sum().item())], device=proj.device)

            if counts.numel() > self.max_wsis:
                counts = counts[: self.max_wsis]

            valid_n = int(vision_attention_mask[b].sum().item())
            patches_b = proj[b, :valid_n, :]

            chunks = []
            start = 0
            for c in counts.tolist():
                if start >= valid_n:
                    break
                end = min(start + int(c), valid_n)
                chunks.append(patches_b[start:end, :])
                start = end
            if start < valid_n:
                chunks.append(patches_b[start:valid_n, :])

            seq = []
            for k, chunk in enumerate(chunks):
                if k >= self.max_wsis:
                    break
                if self.use_index_emb:
                    idx = torch.tensor(k, device=proj.device, dtype=torch.long)
                    marker = self.wsi_index_emb(idx) + self.wsi_sep
                else:
                    marker = self.wsi_sep
                seq.append(marker.unsqueeze(0))
                seq.append(chunk)

            seq_emb = torch.cat(seq, dim=0)
            seq_mask = torch.ones(seq_emb.shape[0], dtype=torch.bool, device=proj.device)

            per_sample_embeds.append(seq_emb)
            per_sample_masks.append(seq_mask)
            max_len = max(max_len, seq_emb.shape[0])

        hsz = proj.shape[-1]
        out_emb = proj.new_zeros((bsz, max_len, hsz))
        out_msk = torch.zeros((bsz, max_len), dtype=torch.bool, device=proj.device)

        for b in range(bsz):
            l = per_sample_embeds[b].shape[0]
            out_emb[b, :l, :] = per_sample_embeds[b]
            out_msk[b, :l] = per_sample_masks[b]

        return out_emb, out_msk

    def _concat_text_and_vision(
        self,
        *,
        vision_prefix_embeds: Optional[torch.Tensor],
        vision_prefix_mask: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be (B, T), got {input_ids.shape}")

        tok_emb = self.llm.get_input_embeddings()
        text_embeds = tok_emb(input_ids)

        bsz, tlen, hsz = text_embeds.shape

        if attention_mask is None:
            attention_mask = torch.ones((bsz, tlen), dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        if vision_prefix_embeds is None:
            out: Dict[str, torch.Tensor] = {
                "inputs_embeds": text_embeds,
                "attention_mask": attention_mask,
            }
            if labels is not None:
                if labels.shape != (bsz, tlen):
                    raise ValueError(f"labels must be (B, T)={(bsz, tlen)}, got {labels.shape}")
                out["labels"] = labels
            return out

        if vision_prefix_embeds.ndim != 3:
            raise ValueError("vision_prefix_embeds must be (B, N, H)")
        if vision_prefix_embeds.shape[0] != bsz or vision_prefix_embeds.shape[2] != hsz:
            raise ValueError(
                f"vision_prefix_embeds shape mismatch: got {vision_prefix_embeds.shape}, "
                f"expected (B={bsz}, N, H={hsz})"
            )

        if vision_prefix_mask is None:
            vision_prefix_mask = torch.ones(
                (bsz, vision_prefix_embeds.shape[1]),
                dtype=torch.bool,
                device=input_ids.device,
            )
        else:
            vision_prefix_mask = vision_prefix_mask.to(dtype=torch.bool)

        inputs_embeds = torch.cat([vision_prefix_embeds, text_embeds], dim=1)
        attn = torch.cat([vision_prefix_mask, attention_mask], dim=1)

        out = {"inputs_embeds": inputs_embeds, "attention_mask": attn}

        if labels is not None:
            if labels.shape != (bsz, tlen):
                raise ValueError(f"labels must be (B, T)={(bsz, tlen)}, got {labels.shape}")
            n_vis = vision_prefix_embeds.shape[1]
            pad = torch.full((bsz, n_vis), -100, dtype=labels.dtype, device=labels.device)
            out["labels"] = torch.cat([pad, labels], dim=1)

        return out

    def forward(
        self,
        *,
        vision_embeddings: Optional[torch.Tensor],
        vision_attention_mask: Optional[torch.Tensor] = None,
        wsi_patch_counts: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **llm_kwargs: Any,
    ) -> VLMOutputs:
        vision_prefix_embeds, vision_prefix_mask = self._build_prefix_inputs(
            vision_embeddings=vision_embeddings,
            vision_attention_mask=vision_attention_mask,
            wsi_patch_counts=wsi_patch_counts,
        )

        packed = self._concat_text_and_vision(
            vision_prefix_embeds=vision_prefix_embeds,
            vision_prefix_mask=vision_prefix_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        llm_out = self.llm(**packed, **llm_kwargs)
        return VLMOutputs(
            loss=getattr(llm_out, "loss", None),
            logits=llm_out.logits,
            llm_outputs=llm_out,
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        vision_embeddings: Optional[torch.Tensor],
        vision_attention_mask: Optional[torch.Tensor] = None,
        wsi_patch_counts: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        min_new_tokens: int = 96,
        do_sample: bool = True,
        temperature: float = 0.6,
        top_p: float = 0.95,
        num_beams: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, list[int]]] = None,
        **gen_kwargs: Any,
    ) -> torch.Tensor:
        vision_prefix_embeds, vision_prefix_mask = self._build_prefix_inputs(
            vision_embeddings=vision_embeddings,
            vision_attention_mask=vision_attention_mask,
            wsi_patch_counts=wsi_patch_counts,
        )

        packed = self._concat_text_and_vision(
            vision_prefix_embeds=vision_prefix_embeds,
            vision_prefix_mask=vision_prefix_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )

        if pad_token_id is None:
            pad_token_id = getattr(self.llm.config, "pad_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.llm.config, "eos_token_id", None)

        generate_kwargs: dict[str, Any] = {
            "inputs_embeds": packed["inputs_embeds"],
            "attention_mask": packed["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            **gen_kwargs,
        }
        if bool(do_sample):
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        generated = self.llm.generate(**generate_kwargs)
        return generated


def _extract_aligner_state_dict(payload: Any) -> dict[str, torch.Tensor] | None:
    if not isinstance(payload, dict):
        return None

    # Common alignment payload format.
    if isinstance(payload.get("aligner_state_dict"), dict):
        return payload["aligner_state_dict"]

    # Legacy main2 trainer save format.
    if isinstance(payload.get("aligner"), dict):
        return payload["aligner"]

    # Full model state dict with aligner prefix.
    keys = list(payload.keys())
    if keys and all(isinstance(k, str) for k in keys):
        aligner_prefixed = [k for k in keys if str(k).startswith("aligner.")]
        if aligner_prefixed:
            out: dict[str, torch.Tensor] = {}
            for k in aligner_prefixed:
                out[str(k)[len("aligner.") :]] = payload[k]
            return out

    # Raw aligner-only state dict.
    if "pre_ln.weight" in payload or "mlp.0.weight" in payload:
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload  # type: ignore[return-value]

    return None


def resolve_aligner_checkpoint_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_file():
        return p

    if p.is_dir():
        candidates = [
            p / "best_aligner_weights.pt",
            p / "alignment_best.pt",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand

        epoch_ckpts = sorted(p.glob("alignment_epoch_*.pt"))
        if epoch_ckpts:
            return epoch_ckpts[-1]

    raise FileNotFoundError(f"Could not resolve aligner checkpoint from: {path_like}")


def load_aligner_from_checkpoint(
    model: PathoSynVLM,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    ckpt_path = resolve_aligner_checkpoint_path(checkpoint_path)

    payload = torch.load(ckpt_path, map_location=map_location)
    aligner_sd = _extract_aligner_state_dict(payload)
    if aligner_sd is None:
        raise ValueError(f"No aligner state dict found in checkpoint: {ckpt_path}")

    missing, unexpected = model.aligner.load_state_dict(aligner_sd, strict=bool(strict))

    out: dict[str, Any] = {
        "checkpoint_path": str(ckpt_path),
        "strict": bool(strict),
        "aligner_num_tensors": int(len(aligner_sd)),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }

    if isinstance(payload, dict):
        if "epoch" in payload:
            out["source_epoch"] = int(payload["epoch"])
        if "global_step" in payload:
            out["source_global_step"] = int(payload["global_step"])
        if "val_metrics" in payload and isinstance(payload["val_metrics"], dict):
            out["source_val_metrics"] = {
                str(k): float(v) for k, v in payload["val_metrics"].items() if isinstance(v, (int, float))
            }

    return out
