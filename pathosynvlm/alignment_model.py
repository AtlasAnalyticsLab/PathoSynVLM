from __future__ import annotations

from typing import Any

import torch

from .model import VLM_MVP


def build_alignment_model(
    *,
    llm_name_or_path: str,
    vision_dim: int,
    feature_key: str = "conch_v15",
    torch_dtype: torch.dtype | None = None,
    attn_implementation: str | None = None,
) -> VLM_MVP:
    """
    Build an alignment-only model:
    - uses precomputed vision embeddings
    - disables WSI markers for one-WSI caption training
    """
    model = VLM_MVP(
        llm_name_or_path=llm_name_or_path,
        vision_dim=vision_dim,
        feature_key=feature_key,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        use_wsi_markers=False,
    )
    return model


def freeze_for_alignment_only(model: VLM_MVP) -> dict[str, Any]:
    """
    Freeze everything except the projection module (`aligner`).

    Notes:
    - There is no trainable vision encoder here because vision tokens are precomputed.
    - LLM is fully frozen.
    """
    for _, p in model.named_parameters():
        p.requires_grad = False

    if hasattr(model, "aligner"):
        for p in model.aligner.parameters():
            p.requires_grad = True

    # Markers are not used in alignment training and stay frozen.
    if hasattr(model, "wsi_index_emb"):
        for p in model.wsi_index_emb.parameters():
            p.requires_grad = False
    if hasattr(model, "wsi_sep") and isinstance(model.wsi_sep, torch.Tensor):
        model.wsi_sep.requires_grad_(False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": int(n_trainable),
        "total_params": int(n_total),
        "trainable_pct": float(100.0 * n_trainable / max(1, n_total)),
    }
