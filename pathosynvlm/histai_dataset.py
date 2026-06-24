from __future__ import annotations

"""
HistAI all-domain dataset loader for finetuning.

Key features:
- Scans all HISTAI embedding folders under:
  data/embeddings/HISTAI-*/<feature>/<patch>/patches
- Uses standardized metadata only:
  - data/histai/standardized_metadata_fixed_filtered_5x_512.json
- Matches records to embeddings by canonical case key: (group, case_number)
- Supports robust h5 probing/filtering so unreadable files are excluded
- Builds chat-template-supervised inputs in the target output format:
  Diagnosis / Certainty / Conclusion
"""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


MissingPolicy = Literal["skip", "error"]
PromptStyle = Literal["single", "double"]

TARGET_FIELD_LABELS: dict[str, str] = {
    "conclusion": "Conclusion",
    "micro_protocol": "Micro protocol",
}


def normalize_target_field_name(target_field_name: str) -> str:
    name = str(target_field_name or "conclusion").strip().lower()
    if not name:
        return "conclusion"
    return name


def resolve_target_field_label(target_field_name: str, target_field_label: str | None = None) -> str:
    if target_field_label is not None and str(target_field_label).strip():
        return str(target_field_label).strip()
    name = normalize_target_field_name(target_field_name)
    return TARGET_FIELD_LABELS.get(name, name.replace("_", " ").strip().title())


def _build_prompt_text_once(target_field_label: str) -> str:
    return (
        "You are a pathology assistant. You will receive multiple WSIs as visual tokens.\n"
        "The visual tokens are separated by WSI markers in order: WSI #1, WSI #2, ...\n"
        "\n"
        "Task:\n"
        "- Use evidence across WSIs to write:\n"
        "Diagnosis: ...\n"
        "Certainty: ...\n"
        f"{target_field_label}: ...\n"
        "\n"
        "Return exactly in the format below:\n"
        "Diagnosis: ...\n"
        "Certainty: ...\n"
        f"{target_field_label}: ...\n"
    )


def resolve_prompt_text(
    prompt_style: str = "single",
    *,
    target_field_name: str = "conclusion",
    target_field_label: str | None = None,
) -> str:
    style = str(prompt_style).strip().lower()
    field_label = resolve_target_field_label(target_field_name, target_field_label)
    prompt_once = _build_prompt_text_once(field_label)
    if style == "single":
        return prompt_once
    if style == "double":
        return prompt_once + prompt_once
    raise ValueError(f"Unsupported prompt_style={prompt_style}. Use: single|double")


def load_tokenizer(
    model_id: str,
    *,
    trust_remote_code: bool = True,
    **kwargs: Any,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _extract_input_ids_1d(tokenized_output: Any) -> torch.Tensor:
    obj = tokenized_output

    if isinstance(obj, dict) and "input_ids" in obj:
        obj = obj["input_ids"]
    elif hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        try:
            if "input_ids" in obj:
                obj = obj["input_ids"]
        except Exception:
            pass

    if hasattr(obj, "ids"):
        obj = obj.ids

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return torch.empty(0, dtype=torch.long)
        first = obj[0]
        if hasattr(first, "ids"):
            obj = first.ids
        elif isinstance(first, (list, tuple)):
            obj = first
        elif isinstance(first, int):
            obj = list(obj)
        else:
            raise TypeError(f"Unsupported token list element type: {type(first).__name__}")
        return torch.tensor(obj, dtype=torch.long)

    if torch.is_tensor(obj):
        t = obj.to(dtype=torch.long)
        if t.ndim == 2:
            if int(t.shape[0]) != 1:
                raise ValueError(f"Expected batch size 1 for chat template tensor, got shape={tuple(t.shape)}")
            return t[0]
        if t.ndim == 1:
            return t
        raise ValueError(f"Expected 1D or 2D token tensor, got shape={tuple(t.shape)}")

    raise TypeError(f"Unsupported tokenized output type: {type(tokenized_output).__name__}")


def _parse_case_key_from_mapping(case_mapping: str) -> tuple[str, int] | None:
    s = str(case_mapping or "").strip()
    if not s:
        return None

    parts = [p for p in re.split(r"[\\/]+", s) if p]
    if len(parts) < 2:
        return None

    group_raw = parts[-2].strip().lower()
    if group_raw.startswith("histai-"):
        group = group_raw[len("histai-") :]
    else:
        group = group_raw

    case_raw = parts[-1].strip().lower()
    m = re.match(r"^case_(\d+)$", case_raw)
    if not m:
        return None

    case_num = int(m.group(1))
    return group, case_num


def _parse_case_key_from_h5_name(filename: str) -> tuple[str, int] | None:
    # e.g. skin-b2_case_00001_slide_H&E_0.h5
    s = str(filename).strip().lower()
    m = re.match(r"^(?P<group>.+?)_case_(?P<num>\d+)(?:_|$)", s)
    if not m:
        return None
    return m.group("group"), int(m.group("num"))


def _canonical_case_mapping(group: str, case_num: int) -> str:
    # keep canonical no-zero-pad to remain stable with mixed metadata styles
    return f"histai/HISTAI-{group}/case_{case_num}"


def _normalize_certainty(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in {"confirmed", "suspected"}:
        return s

    # best-effort normalization for legacy labels
    if s in {"certain", "definite", "yes", "true", "1"}:
        return "confirmed"
    if s in {"uncertain", "possible", "maybe", "no", "false", "0"}:
        return "suspected"

    return "confirmed"


def _read_h5_features(path: Path, feature_key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        features_obj = f.get("features")
        if features_obj is None:
            raise KeyError(f"Missing /features in {path}")
        if not isinstance(features_obj, h5py.Group):
            raise TypeError(f"Expected /features to be a Group in {path}, got {type(features_obj).__name__}")

        g = features_obj
        dataset_obj = g.get(feature_key)
        if dataset_obj is not None:
            if not isinstance(dataset_obj, h5py.Dataset):
                raise TypeError(
                    f"Expected /features/{feature_key} to be a Dataset in {path}, got {type(dataset_obj).__name__}"
                )
            feats = dataset_obj[:]
        else:
            keys = list(g.keys())
            if len(keys) == 1:
                only_obj = g.get(keys[0])
                if not isinstance(only_obj, h5py.Dataset):
                    raise TypeError(
                        f"Expected /features/{keys[0]} to be a Dataset in {path}, got {type(only_obj).__name__}"
                    )
                feats = only_obj[:]
            else:
                raise KeyError(f"Missing /features/{feature_key} in {path}; available={keys}")

    if feats.ndim != 2:
        raise ValueError(f"Expected 2D features (N, D) in {path}, got {feats.shape}")
    if int(feats.shape[0]) <= 0 or int(feats.shape[1]) <= 0:
        raise ValueError(f"Expected non-empty 2D features in {path}, got {feats.shape}")

    return feats.astype(np.float32, copy=False)


def _probe_h5_feature(path: Path, feature_key: str) -> tuple[bool, str]:
    try:
        with h5py.File(path, "r") as f:
            features_obj = f.get("features")
            if features_obj is None:
                return False, "missing_features_group"
            if not isinstance(features_obj, h5py.Group):
                return False, "features_not_group"

            g = features_obj
            dataset_obj = g.get(feature_key)
            if dataset_obj is not None:
                if not isinstance(dataset_obj, h5py.Dataset):
                    return False, "feature_not_dataset"
            else:
                keys = list(g.keys())
                if len(keys) != 1:
                    return False, "missing_feature_key"
                dataset_obj = g.get(keys[0])
                if not isinstance(dataset_obj, h5py.Dataset):
                    return False, "feature_not_dataset"

            shp = getattr(dataset_obj, "shape", None)
            if shp is None or len(shp) != 2:
                return False, "invalid_shape"
            if int(shp[0]) <= 0 or int(shp[1]) <= 0:
                return False, "empty_shape"
            return True, "ok"
    except OSError:
        return False, "open_error"
    except Exception:
        return False, "other_error"


def _classify_h5_exception(exc: Exception) -> str:
    msg = str(exc).lower()
    if isinstance(exc, OSError):
        return "open_error"
    if isinstance(exc, KeyError):
        if "missing /features in" in msg:
            return "missing_features_group"
        if "missing /features/" in msg:
            return "missing_feature_key"
        return "key_error"
    if isinstance(exc, ValueError):
        if "2d features" in msg or "shape" in msg:
            return "invalid_shape"
        return "value_error"
    if isinstance(exc, TypeError):
        return "type_error"
    return "other_error"


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    out: list[dict[str, Any]] = []
    for row in data:
        if isinstance(row, dict):
            out.append(row)
    return out


def _prepare_standardized_records(
    standardized_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Normalize and deduplicate standardized metadata by canonical case key.
    """
    dedup_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    invalid_case_mapping_rows = 0

    for row in standardized_records:
        k = _parse_case_key_from_mapping(str(row.get("case_mapping") or ""))
        if k is None:
            invalid_case_mapping_rows += 1
            continue
        if k in dedup_by_key:
            continue
        dedup_by_key[k] = dict(row)

    final_rows: list[dict[str, Any]] = []
    for row in standardized_records:
        k = _parse_case_key_from_mapping(str(row.get("case_mapping") or ""))
        if k is None or k not in dedup_by_key:
            continue
        cur = dedup_by_key.pop(k)
        group, case_num = k

        out = dict(cur)
        out["case_mapping"] = str(out.get("case_mapping") or _canonical_case_mapping(group, case_num))
        if not str(out.get("standardized_diagnosis") or "").strip():
            out["standardized_diagnosis"] = str(out.get("diagnosis") or "")
        if not str(out.get("diagnosis") or "").strip():
            out["diagnosis"] = str(out.get("standardized_diagnosis") or "")
        out["certainty"] = _normalize_certainty(out.get("certainty"))
        out["_metadata_source"] = "standardized"
        out["_case_group"] = group
        out["_case_num"] = case_num
        final_rows.append(out)

    stats = {
        "standardized_rows": int(len(standardized_records)),
        "standardized_unique_cases": int(len(final_rows)),
        "invalid_case_mapping_rows": int(invalid_case_mapping_rows),
    }
    return final_rows, stats


def _collect_histai_patch_dirs(
    dataset_embeddings_root: Path,
    feature_key: str,
    patch_level: str,
) -> list[Path]:
    root = Path(dataset_embeddings_root)
    dirs: list[Path] = []

    for histai_dir in sorted(root.glob("HISTAI-*")):
        patch_dir = histai_dir / feature_key / patch_level / "patches"
        if patch_dir.is_dir():
            dirs.append(patch_dir)

    return dirs


def _build_h5_index(patch_dirs: Iterable[Path]) -> tuple[dict[tuple[str, int], list[Path]], int]:
    index: dict[tuple[str, int], list[Path]] = {}
    n_files = 0

    for d in patch_dirs:
        for p in d.glob("*.h5"):
            n_files += 1
            k = _parse_case_key_from_h5_name(p.name)
            if k is None:
                continue
            index.setdefault(k, []).append(p)

    for k in list(index.keys()):
        index[k] = sorted(set(index[k]), key=lambda x: str(x))

    return index, n_files


def _val_size_to_count(val_size: int | float | str, n: int) -> int:
    if n <= 0:
        return 0

    if isinstance(val_size, str):
        s = val_size.strip()
        if s.endswith("%"):
            frac = float(s[:-1].strip()) / 100.0
            return max(0, min(n, int(round(frac * n))))
        if "." in s:
            f = float(s)
            if 0.0 <= f <= 1.0:
                return max(0, min(n, int(round(f * n))))
            return max(0, min(n, int(round(f))))
        return max(0, min(n, int(s)))

    if isinstance(val_size, float):
        if 0.0 <= val_size <= 1.0:
            return max(0, min(n, int(round(val_size * n))))
        return max(0, min(n, int(round(val_size))))

    return max(0, min(n, int(val_size)))


def make_train_val_split(
    case_mappings: list[str],
    *,
    val_size: int | float | str = 500,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    n = len(case_mappings)
    val_n = _val_size_to_count(val_size, n)
    if val_n <= 0:
        return list(range(n)), []

    scored: list[tuple[str, int]] = []
    for i, cm in enumerate(case_mappings):
        h = hashlib.sha1(f"{seed}:{cm}".encode("utf-8")).hexdigest()
        scored.append((h, i))
    scored.sort(key=lambda t: t[0])

    val_idx = sorted(i for _, i in scored[:val_n])
    train_idx = sorted(i for _, i in scored[val_n:])
    return train_idx, val_idx


@dataclass(frozen=True)
class CaseItem:
    vision_embeddings: torch.Tensor
    wsi_patch_counts: torch.Tensor
    standardized_diagnosis: str
    certainty: str
    conclusion: str
    case_mapping: str
    slide_paths: tuple[str, ...]


class HistAIAllCaseDataset(Dataset):
    def __init__(
        self,
        *,
        metadata_standardized_json: Path,
        dataset_embeddings_root: Path,
        feature_key: str = "conch_v15",
        patch_level: str = "5x_512",
        diagnosis_field: str = "standardized_diagnosis",
        certainty_field: str = "certainty",
        conclusion_field: str = "conclusion",
        missing_policy: MissingPolicy = "skip",
        subset_indices: list[int] | None = None,
        probe_h5_on_init: bool = False,
        max_vision_tokens: int = 0,
        vision_token_dropout: float = 0.0,
    ) -> None:
        self.metadata_standardized_json = Path(metadata_standardized_json)
        self.dataset_embeddings_root = Path(dataset_embeddings_root)
        self.feature_key = str(feature_key)
        self.patch_level = str(patch_level)
        self.diagnosis_field = str(diagnosis_field)
        self.certainty_field = str(certainty_field)
        self.conclusion_field = str(conclusion_field)
        self.missing_policy = missing_policy
        self.probe_h5_on_init = bool(probe_h5_on_init)
        self.max_vision_tokens = max(0, int(max_vision_tokens))
        self.vision_token_dropout = float(vision_token_dropout)
        if not (0.0 <= self.vision_token_dropout <= 1.0):
            raise ValueError(f"vision_token_dropout must be in [0, 1], got {self.vision_token_dropout}")

        standardized_records = _load_json_records(self.metadata_standardized_json)
        normalized_records, normalize_stats = _prepare_standardized_records(standardized_records)
        self.records: list[dict[str, Any]] = normalized_records

        patch_dirs = _collect_histai_patch_dirs(
            self.dataset_embeddings_root,
            self.feature_key,
            self.patch_level,
        )
        self.patch_dirs = patch_dirs
        self.h5_index, n_h5_files = _build_h5_index(patch_dirs)

        self.stats: dict[str, Any] = {
            **normalize_stats,
            "feature_key": self.feature_key,
            "patch_level": self.patch_level,
            "max_vision_tokens": int(self.max_vision_tokens),
            "vision_token_dropout": float(self.vision_token_dropout),
            "patch_dirs": int(len(patch_dirs)),
            "h5_files_scanned": int(n_h5_files),
            "embedding_unique_cases": int(len(self.h5_index)),
            "missing_embedding": 0,
            "empty_text": 0,
            "matched_rows": 0,
            "dropped_invalid_h5": 0,
        }

        self.indices: list[int] = []
        self.case_key_by_record_idx: dict[int, tuple[str, int]] = {}
        self.record_h5_paths: dict[int, tuple[Path, ...]] = {}

        self.invalid_h5_paths: list[str] = []
        self.invalid_h5_reason_counts: dict[str, int] = {}
        invalid_h5_unique: set[str] = set()
        reason_counts: dict[str, int] = {}

        probe_cache: dict[str, tuple[bool, str]] = {}

        for i, r in enumerate(self.records):
            cm = str(r.get("case_mapping") or "")
            key = _parse_case_key_from_mapping(cm)
            if key is None:
                continue

            diag = str(r.get(self.diagnosis_field) or r.get("standardized_diagnosis") or r.get("diagnosis") or "").strip()
            conc = str(r.get(self.conclusion_field) or r.get("conclusion") or "").strip()
            if not diag and not conc:
                self.stats["empty_text"] = int(self.stats.get("empty_text", 0)) + 1
                continue

            h5_paths = list(self.h5_index.get(key, []))
            if not h5_paths:
                self.stats["missing_embedding"] = int(self.stats.get("missing_embedding", 0)) + 1
                if self.missing_policy == "error":
                    raise FileNotFoundError(f"No embeddings for case_mapping={cm}")
                continue

            if self.probe_h5_on_init:
                valid_paths: list[Path] = []
                for p in h5_paths:
                    cache_key = str(p)
                    pr = probe_cache.get(cache_key)
                    if pr is None:
                        pr = _probe_h5_feature(p, self.feature_key)
                        probe_cache[cache_key] = pr
                    ok, reason = pr
                    if ok:
                        valid_paths.append(p)
                    else:
                        invalid_h5_unique.add(cache_key)
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1

                if not valid_paths:
                    self.stats["dropped_invalid_h5"] = int(self.stats.get("dropped_invalid_h5", 0)) + 1
                    if self.missing_policy == "error":
                        raise RuntimeError(f"No valid embeddings after probe for case_mapping={cm}")
                    continue
                h5_paths = valid_paths

            self.indices.append(i)
            self.case_key_by_record_idx[i] = key
            self.record_h5_paths[i] = tuple(h5_paths)
            self.stats["matched_rows"] = int(self.stats.get("matched_rows", 0)) + 1

        if subset_indices is not None:
            keep = set(int(x) for x in subset_indices)
            self.indices = [i for i in self.indices if i in keep]
        self.stats["selected_rows_after_subset"] = int(len(self.indices))

        self.invalid_h5_paths = sorted(invalid_h5_unique)
        self.invalid_h5_reason_counts = dict(sorted(reason_counts.items(), key=lambda kv: kv[0]))
        self.stats["invalid_h5_files_unique"] = int(len(self.invalid_h5_paths))
        for reason, cnt in self.invalid_h5_reason_counts.items():
            self.stats[f"invalid_h5_{reason}"] = int(cnt)

        if len(self.indices) == 0 and self.missing_policy == "error":
            raise RuntimeError("No matched HistAI records with embeddings.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> CaseItem:
        rec_idx = self.indices[int(idx)]
        r = self.records[rec_idx]

        h5_paths = list(self.record_h5_paths.get(rec_idx, ()))
        if not h5_paths:
            key = self.case_key_by_record_idx.get(rec_idx)
            if key is not None:
                h5_paths = list(self.h5_index.get(key, []))

        slide_paths: list[str] = []
        features_list: list[np.ndarray] = []
        wsi_patch_counts: list[int] = []

        for p in h5_paths:
            try:
                feats = _read_h5_features(p, self.feature_key)
            except Exception as exc:
                reason = _classify_h5_exception(exc if isinstance(exc, Exception) else Exception(str(exc)))
                self.invalid_h5_reason_counts[reason] = self.invalid_h5_reason_counts.get(reason, 0) + 1
                self.stats[f"invalid_h5_{reason}"] = int(self.invalid_h5_reason_counts[reason])
                self.stats["dropped_invalid_h5"] = int(self.stats.get("dropped_invalid_h5", 0)) + 1
                self.invalid_h5_paths = sorted(set(self.invalid_h5_paths) | {str(p)})
                continue

            features_list.append(feats)
            wsi_patch_counts.append(int(feats.shape[0]))
            slide_paths.append(str(p))

        if not features_list:
            # avoid worker crash: try a nearby sample
            n = len(self.indices)
            for off in range(1, n):
                j = (int(idx) + off) % n
                if j == int(idx):
                    break
                try:
                    return self.__getitem__(j)
                except Exception:
                    continue
            if self.missing_policy == "error":
                raise RuntimeError("No readable h5 features available for any sample")
            raise RuntimeError("All candidate h5 files failed to load")

        if self.max_vision_tokens > 0:
            remaining = int(self.max_vision_tokens)
            trimmed_features: list[np.ndarray] = []
            trimmed_wsi_patch_counts: list[int] = []
            trimmed_slide_paths: list[str] = []
            for feats, path in zip(features_list, slide_paths):
                if remaining <= 0:
                    break
                n = int(feats.shape[0])
                take = min(n, remaining)
                if take <= 0:
                    break
                if take < n:
                    feats = feats[:take]
                trimmed_features.append(feats)
                trimmed_wsi_patch_counts.append(int(take))
                trimmed_slide_paths.append(path)
                remaining -= int(take)

            if not trimmed_features:
                raise RuntimeError("max_vision_tokens too small; no vision tokens kept for sample")
            features_list = trimmed_features
            wsi_patch_counts = trimmed_wsi_patch_counts
            slide_paths = trimmed_slide_paths

        vision_np = np.concatenate(features_list, axis=0)
        vision = torch.from_numpy(vision_np.astype(np.float32, copy=False))

        # Token dropout augmentation: zero entire token rows, keep tensor shape unchanged.
        if self.vision_token_dropout > 0.0 and vision.ndim == 2 and int(vision.shape[0]) > 0:
            drop_mask = torch.rand(int(vision.shape[0])) < float(self.vision_token_dropout)
            if bool(drop_mask.any()):
                vision[drop_mask, :] = 0.0


        diagnosis = str(r.get(self.diagnosis_field) or r.get("standardized_diagnosis") or r.get("diagnosis") or "")
        certainty = _normalize_certainty(r.get(self.certainty_field) or r.get("certainty"))
        conclusion = str(r.get(self.conclusion_field) or r.get("conclusion") or "")
        case_mapping = str(r.get("case_mapping") or "")

        return CaseItem(
            vision_embeddings=vision,
            wsi_patch_counts=torch.tensor(wsi_patch_counts, dtype=torch.long),
            standardized_diagnosis=diagnosis,
            certainty=certainty,
            conclusion=conclusion,
            case_mapping=case_mapping,
            slide_paths=tuple(slide_paths),
        )


class VLMDataCollator:
    def __init__(
        self,
        *,
        tokenizer: Any | None,
        max_text_length: int = 512,
        vision_pad_value: float = 0.0,
        prompt_style: PromptStyle = "single",
        target_field_name: str = "conclusion",
        target_field_label: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max(0, int(max_text_length))
        self.vision_pad_value = float(vision_pad_value)
        self.prompt_style = str(prompt_style).strip().lower()
        self.target_field_name = normalize_target_field_name(target_field_name)
        self.target_field_label = resolve_target_field_label(self.target_field_name, target_field_label)
        self.prompt_text = resolve_prompt_text(
            self.prompt_style,
            target_field_name=self.target_field_name,
            target_field_label=self.target_field_label,
        )

    def __call__(self, batch: list[CaseItem]) -> dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch")

        lengths = [int(x.vision_embeddings.shape[0]) for x in batch]
        max_n = max(lengths)
        d = int(batch[0].vision_embeddings.shape[1]) if max_n > 0 else 0

        vision = torch.full((len(batch), max_n, d), self.vision_pad_value, dtype=torch.float32)
        vision_mask = torch.zeros((len(batch), max_n), dtype=torch.bool)
        for i, item in enumerate(batch):
            n = int(item.vision_embeddings.shape[0])
            if n == 0:
                continue
            if int(item.vision_embeddings.shape[1]) != d:
                raise ValueError("Embedding dimension mismatch within batch")
            vision[i, :n, :] = item.vision_embeddings
            vision_mask[i, :n] = True

        max_wsi = max(len(x.wsi_patch_counts) for x in batch)
        wsi_patch_counts = torch.zeros((len(batch), max_wsi), dtype=torch.long)
        for i, item in enumerate(batch):
            counts = item.wsi_patch_counts
            wsi_patch_counts[i, : counts.numel()] = counts.to(dtype=torch.long)

        raw_diagnosis_texts = [x.standardized_diagnosis for x in batch]
        raw_certainty_texts = [x.certainty for x in batch]
        raw_conclusion_texts = [x.conclusion for x in batch]

        if self.tokenizer is None:
            return {
                "vision_embeddings": vision,
                "vision_attention_mask": vision_mask,
                "wsi_patch_counts": wsi_patch_counts,
                "total_patch_counts": lengths,
                "raw_diagnosis_text": raw_diagnosis_texts,
                "certainty": raw_certainty_texts,
                "raw_conclusion_text": raw_conclusion_texts,
                "raw_target_text": raw_conclusion_texts,
                "target_field_name": self.target_field_name,
                "target_field_label": self.target_field_label,
                "case_mapping": [x.case_mapping for x in batch],
                "slide_paths": [x.slide_paths for x in batch],
            }

        if not hasattr(self.tokenizer, "apply_chat_template") or self.tokenizer.chat_template is None:
            raise ValueError(
                "Tokenizer has no chat_template/apply_chat_template. "
                "Use an instruct tokenizer (e.g., Qwen2.5-Instruct) or set tokenizer.chat_template."
            )

        pad_id = int(self.tokenizer.pad_token_id)
        per_sample_input_ids: list[torch.Tensor] = []
        per_sample_labels: list[torch.Tensor] = []

        for item, diag, cert, conc in zip(batch, raw_diagnosis_texts, raw_certainty_texts, raw_conclusion_texts):
            wsi_counts = [int(x) for x in item.wsi_patch_counts.tolist() if int(x) > 0]
            n_wsi = max(1, len(wsi_counts))
            wsi_markers = "\n".join([f"WSI #{i + 1}" for i in range(n_wsi)])

            user_msg = (
                "Please analyze the provided WSIs (as visual tokens) and respond strictly in the requested format.\n"
                f"{wsi_markers}"
            )
            assistant_msg = (
                f"Diagnosis: {diag.strip()}\n"
                f"Certainty: {cert.strip()}\n"
                f"{self.target_field_label}: {conc.strip()}"
            )

            messages_full = [
                {"role": "system", "content": self.prompt_text.strip()},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            messages_prompt = [
                {"role": "system", "content": self.prompt_text.strip()},
                {"role": "user", "content": user_msg},
            ]

            full_ids = self.tokenizer.apply_chat_template(
                messages_full,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )
            prompt_ids = self.tokenizer.apply_chat_template(
                messages_prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            input_ids_1d = _extract_input_ids_1d(full_ids)
            prompt_ids_1d = _extract_input_ids_1d(prompt_ids)

            labels_1d = input_ids_1d.clone()
            prompt_len = min(int(prompt_ids_1d.numel()), int(labels_1d.numel()))
            labels_1d[:prompt_len] = -100

            if self.max_text_length > 0 and input_ids_1d.numel() > self.max_text_length:
                input_ids_1d = input_ids_1d[: self.max_text_length]
                labels_1d = labels_1d[: self.max_text_length]

            per_sample_input_ids.append(input_ids_1d)
            per_sample_labels.append(labels_1d)

        max_t = max(int(x.numel()) for x in per_sample_input_ids)
        bsz = len(per_sample_input_ids)

        input_ids = torch.full((bsz, max_t), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_t), dtype=torch.long)
        labels = torch.full((bsz, max_t), -100, dtype=torch.long)

        for i, (ids_1d, labs_1d) in enumerate(zip(per_sample_input_ids, per_sample_labels)):
            t = int(ids_1d.numel())
            input_ids[i, :t] = ids_1d
            attention_mask[i, :t] = 1
            labels[i, :t] = labs_1d

        return {
            "vision_embeddings": vision,
            "vision_attention_mask": vision_mask,
            "wsi_patch_counts": wsi_patch_counts,
            "total_patch_counts": lengths,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "raw_diagnosis_text": raw_diagnosis_texts,
            "certainty": raw_certainty_texts,
            "raw_conclusion_text": raw_conclusion_texts,
            "raw_target_text": raw_conclusion_texts,
            "target_field_name": self.target_field_name,
            "target_field_label": self.target_field_label,
            "case_mapping": [x.case_mapping for x in batch],
            "slide_paths": [x.slide_paths for x in batch],
        }


def create_train_val_dataloaders(
    *,
    metadata_standardized_json: Path,
    dataset_embeddings_root: Path,
    tokenizer: Any | None,
    feature_key: str = "conch_v15",
    patch_level: str = "5x_512",
    diagnosis_field: str = "standardized_diagnosis",
    conclusion_field: str = "conclusion",
    target_field_name: str | None = None,
    target_field_label: str | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    max_text_length: int = 512,
    max_vision_tokens: int = 0,
    vision_token_dropout: float = 0.0,
    prompt_style: PromptStyle = "single",
    missing_policy: MissingPolicy = "skip",
    probe_h5_on_init: bool = False,
    val_size: int | float | str = 500,
    split_seed: int = 42,
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    base = HistAIAllCaseDataset(
        metadata_standardized_json=Path(metadata_standardized_json),
        dataset_embeddings_root=Path(dataset_embeddings_root),
        feature_key=feature_key,
        patch_level=patch_level,
        diagnosis_field=diagnosis_field,
        conclusion_field=conclusion_field,
        missing_policy=missing_policy,
        probe_h5_on_init=bool(probe_h5_on_init),
    )

    case_mappings = [str(base.records[i].get("case_mapping") or "") for i in base.indices]
    train_pos, val_pos = make_train_val_split(case_mappings, val_size=val_size, seed=int(split_seed))

    train_indices = [base.indices[p] for p in train_pos]
    val_indices = [base.indices[p] for p in val_pos]

    train_ds = HistAIAllCaseDataset(
        metadata_standardized_json=Path(metadata_standardized_json),
        dataset_embeddings_root=Path(dataset_embeddings_root),
        feature_key=feature_key,
        patch_level=patch_level,
        diagnosis_field=diagnosis_field,
        conclusion_field=conclusion_field,
        missing_policy=missing_policy,
        subset_indices=train_indices,
        probe_h5_on_init=bool(probe_h5_on_init),
        max_vision_tokens=int(max_vision_tokens),
        vision_token_dropout=float(vision_token_dropout),
    )
    val_ds = HistAIAllCaseDataset(
        metadata_standardized_json=Path(metadata_standardized_json),
        dataset_embeddings_root=Path(dataset_embeddings_root),
        feature_key=feature_key,
        patch_level=patch_level,
        diagnosis_field=diagnosis_field,
        conclusion_field=conclusion_field,
        missing_policy=missing_policy,
        subset_indices=val_indices,
        probe_h5_on_init=bool(probe_h5_on_init),
        max_vision_tokens=int(max_vision_tokens),
        vision_token_dropout=0.0,
    )

    collator = VLMDataCollator(
        tokenizer=tokenizer,
        max_text_length=max_text_length,
        prompt_style=prompt_style,
        target_field_name=(target_field_name or conclusion_field),
        target_field_label=target_field_label,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=bool(train_shuffle),
        num_workers=int(num_workers),
        collate_fn=collator,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collator,
    )
    return train_dl, val_dl


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HistAI all-dataset loader sanity check")
    p.add_argument("--metadata-standardized-json", type=Path, required=True)
    p.add_argument("--dataset-embeddings-root", type=Path, required=True)
    p.add_argument("--feature-key", type=str, default="conch_v15")
    p.add_argument("--patch-level", type=str, default="5x_512")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-text-length", type=int, default=512)
    p.add_argument("--max-vision-tokens", type=int, default=0)
    p.add_argument("--vision-token-dropout", type=float, default=0.0)
    p.add_argument("--prompt-style", type=str, default="single", choices=["single", "double"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-size", default="0.2")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--tokenizer-id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--no-tokenizer", action="store_true")
    p.add_argument("--probe-h5-on-init", action="store_true")
    p.add_argument("--limit", type=int, default=2)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    tokenizer = None
    if not args.no_tokenizer:
        tokenizer = load_tokenizer(args.tokenizer_id, trust_remote_code=True, use_fast=True)

    train_dl, val_dl = create_train_val_dataloaders(
        metadata_standardized_json=args.metadata_standardized_json,
        dataset_embeddings_root=args.dataset_embeddings_root,
        tokenizer=tokenizer,
        feature_key=args.feature_key,
        patch_level=args.patch_level,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_text_length=args.max_text_length,
        max_vision_tokens=args.max_vision_tokens,
        vision_token_dropout=args.vision_token_dropout,
        prompt_style=args.prompt_style,
        val_size=args.val_size,
        split_seed=args.split_seed,
        probe_h5_on_init=args.probe_h5_on_init,
    )

    print("train_size:", len(train_dl.dataset))
    print("val_size:", len(val_dl.dataset))

    base = train_dl.dataset
    if hasattr(base, "stats"):
        print("dataset_stats:", json.dumps(getattr(base, "stats"), indent=2))

    for name, dl in (("train", train_dl), ("val", val_dl)):
        print(f"\\n[{name}] sample batches")
        for i, b in enumerate(dl):
            print("batch", i)
            print("vision_embeddings:", tuple(b["vision_embeddings"].shape))
            print("vision_attention_mask:", tuple(b["vision_attention_mask"].shape))
            print("wsi_patch_counts:", tuple(b["wsi_patch_counts"].shape))
            if "input_ids" in b:
                print("input_ids:", tuple(b["input_ids"].shape))
                print("attention_mask:", tuple(b["attention_mask"].shape))
                print("labels:", tuple(b["labels"].shape))
                print("case_mapping[0]:", b["case_mapping"][0])
            if i + 1 >= int(args.limit):
                break
