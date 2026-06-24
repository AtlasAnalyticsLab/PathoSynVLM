from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, cast

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer


MissingPolicy = Literal["skip", "error"]
DatasetKey = Literal["histgen", "pathtext", "reg_dataset"]
ALL_DATASETS: tuple[DatasetKey, ...] = ("histgen", "pathtext", "reg_dataset")

PROMPT_PREFIX = (
    "You are a pathology assistant. You will receive one WSI as visual tokens.\n"
    "Use the visual evidence to write the report/caption for that WSI.\n"
    "Summarize key histologic findings and give an impression for the given whole-slide image."
)

PROMPT_DOUBLE = (
    "You are a pathology assistant. You will receive one WSI as visual tokens.\n"
    "Use the visual evidence to write the report/caption for that WSI.\n"
    "Summarize key histologic findings and give an impression for the given whole-slide image.\n"
    "You are a pathology assistant. You will receive one WSI as visual tokens.\n"
    "Use the visual evidence to write the report/caption for that WSI.\n"
    "Summarize key histologic findings and give an impression for the given whole-slide image."
)

PromptStyle = Literal["single", "double"]


def resolve_prompt_text(prompt_style: str = "single") -> str:
    style = str(prompt_style).strip().lower()
    if style == "single":
        return PROMPT_PREFIX
    if style == "double":
        return PROMPT_DOUBLE
    raise ValueError(f"Unsupported prompt_style={prompt_style}. Use: single|double")


def _normalize_dataset_key(token: str) -> DatasetKey:
    t = str(token).strip().lower()
    if t == "reg":
        t = "reg_dataset"
    if t in ALL_DATASETS:
        return cast(DatasetKey, t)
    raise ValueError(
        f"Unsupported dataset token: {token}. "
        "Supported: all, histgen, pathtext, reg (or reg_dataset), "
        "no_reg, no_histgen, no_pathtext."
    )


def parse_dataset_selection(selection: str | None) -> set[DatasetKey]:
    """
    Parses dataset selector string.

    Examples:
      all
      reg
      no_reg
      histgen,pathtext
      all,no_reg
    """
    if selection is None:
        return set(ALL_DATASETS)

    raw = str(selection).strip().lower()
    if raw in {"", "all", "*"}:
        return set(ALL_DATASETS)

    tokens = [x.strip() for x in raw.replace("+", ",").split(",") if x.strip()]
    out: set[DatasetKey] = set()
    all_set = set(ALL_DATASETS)

    for tok in tokens:
        if tok in {"all", "*"}:
            out |= all_set
            continue
        if tok.startswith("no_"):
            key = _normalize_dataset_key(tok[len("no_") :])
            if not out:
                out = set(all_set)
            out.discard(key)
            continue
        out.add(_normalize_dataset_key(tok))

    if not out:
        raise ValueError(
            f"Dataset selection '{selection}' resolves to empty set. "
            "Use one of: all, histgen, pathtext, reg, no_reg, no_histgen, no_pathtext."
        )
    return out


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
    """
    Normalize tokenizer/apply_chat_template outputs to a 1D LongTensor of token ids.
    Handles HF version differences:
    - Tensor / BatchEncoding / list[int] / list[list[int]] / tokenizers.Encoding
    """
    obj = tokenized_output

    # BatchEncoding/dict path
    if isinstance(obj, dict) and "input_ids" in obj:
        obj = obj["input_ids"]
    elif hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        try:
            if "input_ids" in obj:
                obj = obj["input_ids"]
        except Exception:
            pass

    # tokenizers.Encoding path
    if hasattr(obj, "ids"):
        obj = obj.ids

    # list/tuple path
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

    # tensor path
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


def _pathtext_case_prefix(text: str) -> str:
    s = Path(str(text).strip()).stem
    if not s:
        return ""
    m = re.match(r"^(TCGA-[^-]+-[^-]+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return s.lower()


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
        raise ValueError(f"Expected non-empty 2D features (N>0, D>0) in {path}, got {feats.shape}")
    return feats.astype(np.float32, copy=False)


def _probe_h5_feature(path: Path, feature_key: str) -> tuple[bool, str]:
    """
    Lightweight probe to validate that an h5 file contains usable 2D features.
    Returns:
      (True, "ok") on success
      (False, <reason>) on failure
    """
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


def _collect_patch_dirs(
    dataset_embeddings_root: Path,
    feature_key: str,
    patch_level: str,
    datasets: set[DatasetKey] | None = None,
) -> list[Path]:
    root = Path(dataset_embeddings_root)
    out: list[Path] = []
    selected = set(ALL_DATASETS) if datasets is None else set(datasets)
    candidates: list[Path] = []

    if "histgen" in selected:
        candidates.extend(
            [
                root / "HistGen-train" / feature_key / patch_level / "patches",
                root / "HistGen-val" / feature_key / patch_level / "patches",
                root / "HistGen-test" / feature_key / patch_level / "patches",
            ]
        )
    if "pathtext" in selected:
        candidates.append(root / "PathText" / feature_key / patch_level / "patches")
    if "reg_dataset" in selected:
        candidates.extend(
            [
                root / "REG_dataset" / "REG_train" / feature_key / patch_level / "patches",
                root / "REG_dataset" / "REG_test" / "REG_test1" / feature_key / patch_level / "patches",
                root / "REG_dataset" / "REG_test" / "REG_test2_revised" / feature_key / patch_level / "patches",
            ]
        )

    for p in candidates:
        if p.is_dir():
            out.append(p)

    return sorted(set(out), key=lambda p: str(p))


def _build_h5_index(patch_dirs: Iterable[Path]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """
    Build two indexes:
    - exact stem -> h5 list
    - pathtext prefix (TCGA-XX-XXXX) -> h5 list
    """
    exact: dict[str, list[Path]] = {}
    prefix: dict[str, list[Path]] = {}

    for d in patch_dirs:
        for p in d.glob("*.h5"):
            stem = p.stem.lower()
            exact.setdefault(stem, []).append(p)

            pref = _pathtext_case_prefix(stem)
            if pref:
                prefix.setdefault(pref, []).append(p)

    for k in list(exact.keys()):
        exact[k] = sorted(set(exact[k]), key=lambda x: str(x))
    for k in list(prefix.keys()):
        prefix[k] = sorted(set(prefix[k]), key=lambda x: str(x))

    return exact, prefix


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
    case_ids: list[str],
    *,
    val_size: int | float | str = 500,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    n = len(case_ids)
    val_n = _val_size_to_count(val_size, n)
    if val_n <= 0:
        return list(range(n)), []

    scored: list[tuple[str, int]] = []
    for i, cid in enumerate(case_ids):
        h = hashlib.sha1(f"{seed}:{cid}".encode("utf-8")).hexdigest()
        scored.append((h, i))
    scored.sort(key=lambda t: t[0])

    val_idx = sorted(i for _, i in scored[:val_n])
    train_idx = sorted(i for _, i in scored[val_n:])
    return train_idx, val_idx


def _choose_readable_h5_path(
    *,
    sample: "AlignmentSample",
    feature_key: str,
    probe_cache: dict[str, tuple[bool, str]],
) -> tuple[Path | None, bool, str]:
    """
    Returns:
      (chosen_path, used_alternate, fail_reason)
    """
    first_reason = "unknown"
    for cand_idx, cand in enumerate(sample.h5_paths):
        k = str(cand)
        probe = probe_cache.get(k)
        if probe is None:
            probe = _probe_h5_feature(cand, feature_key)
            probe_cache[k] = probe
        ok, reason = probe
        if ok:
            return cand, bool(cand_idx > 0), "ok"
        if first_reason == "unknown":
            first_reason = str(reason)
    return None, False, first_reason


def _filter_indices_to_readable_h5(
    *,
    base_dataset: "AlignmentCaseDataset",
    indices: list[int],
    split_name: str,
    probe_cache: dict[str, tuple[bool, str]],
) -> tuple[list[int], dict[str, Any], set[str], dict[str, int], int]:
    """
    Validate each sample in `indices` has at least one readable h5 with non-empty features.
    Also rewrites sample.h5_paths to a single chosen readable file to avoid runtime fallback.
    """
    kept: list[int] = []
    dropped = 0
    used_alternate = 0
    reason_counts: dict[str, int] = {}
    invalid_paths: set[str] = set()

    for idx in indices:
        s = base_dataset.samples[idx]
        chosen, used_alt, reason = _choose_readable_h5_path(
            sample=s,
            feature_key=base_dataset.feature_key,
            probe_cache=probe_cache,
        )
        if chosen is None:
            dropped += 1
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for p in s.h5_paths:
                invalid_paths.add(str(p))
            continue

        if used_alt:
            used_alternate += 1

        # Pin a single known-good h5 path to avoid runtime trial-and-error.
        if len(s.h5_paths) != 1 or s.h5_paths[0] != chosen:
            base_dataset.samples[idx] = AlignmentSample(
                case_id=s.case_id,
                report_text=s.report_text,
                h5_paths=(chosen,),
            )

        kept.append(idx)

    stats = {
        "split": split_name,
        "input_rows": int(len(indices)),
        "kept_rows": int(len(kept)),
        "dropped_invalid_h5_rows": int(dropped),
        "used_alternate_h5_rows": int(used_alternate),
        "invalid_h5_reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: kv[0])),
    }
    return kept, stats, invalid_paths, reason_counts, used_alternate


@dataclass(frozen=True)
class AlignmentSample:
    case_id: str
    report_text: str
    h5_paths: tuple[Path, ...]


@dataclass(frozen=True)
class CaseItem:
    vision_embeddings: torch.Tensor
    wsi_patch_counts: torch.Tensor
    report_text: str
    case_id: str
    slide_path: str


class AlignmentCaseDataset(Dataset):
    def __init__(
        self,
        *,
        metadata_json: Path,
        dataset_embeddings_root: Path,
        feature_key: str = "conch_v15",
        patch_level: str = "5x_512",
        datasets: str = "all",
        missing_policy: MissingPolicy = "skip",
        probe_h5_on_init: bool = False,
    ) -> None:
        self.metadata_json = Path(metadata_json)
        self.dataset_embeddings_root = Path(dataset_embeddings_root)
        self.feature_key = str(feature_key)
        self.patch_level = str(patch_level)
        self.datasets = str(datasets)
        self.selected_datasets = parse_dataset_selection(self.datasets)
        self.missing_policy = missing_policy
        self.probe_h5_on_init = bool(probe_h5_on_init)

        data = json.loads(self.metadata_json.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("metadata_json must be a list of records with fields: id, report")

        patch_dirs = _collect_patch_dirs(
            self.dataset_embeddings_root,
            self.feature_key,
            self.patch_level,
            datasets=self.selected_datasets,
        )
        exact_index, prefix_index = _build_h5_index(patch_dirs)

        self.stats: dict[str, Any] = {
            "metadata_rows": len(data),
            "matched_rows": 0,
            "missing_embedding": 0,
            "empty_text": 0,
            "dropped_invalid_h5": 0,
            "used_alternate_h5": 0,
            "patch_dirs": len(patch_dirs),
            "h5_files": sum(len(v) for v in exact_index.values()),
        }
        self.stats["datasets_option"] = self.datasets
        self.stats["selected_datasets"] = sorted(self.selected_datasets)
        self.stats["probe_h5_on_init"] = int(self.probe_h5_on_init)

        self.samples: list[AlignmentSample] = []
        self.invalid_h5_paths: list[str] = []
        self.invalid_h5_reason_counts: dict[str, int] = {}
        self._runtime_invalid_h5_paths: set[str] = set()
        self._runtime_invalid_h5_reason_counts: dict[str, int] = {}

        h5_probe_cache: dict[str, tuple[bool, str]] = {}
        invalid_h5_unique: set[str] = set()
        reason_counts: dict[str, int] = {}

        for row in data:
            if not isinstance(row, dict):
                continue
            case_id = str(row.get("id") or "").strip()
            report = str(row.get("report") or row.get("caption") or "").strip()
            if not case_id:
                continue
            if not report:
                self.stats["empty_text"] += 1
                continue

            key = Path(case_id).stem.lower()
            h5_paths = exact_index.get(key, [])
            if not h5_paths:
                # Fallback for PathText-style prefixes.
                pref = _pathtext_case_prefix(key)
                h5_paths = prefix_index.get(pref, [])

            if not h5_paths:
                self.stats["missing_embedding"] += 1
                if self.missing_policy == "error":
                    raise FileNotFoundError(f"No embedding found for id={case_id}")
                continue

            if self.probe_h5_on_init:
                chosen: Path | None = None
                for p_idx, cand in enumerate(h5_paths):
                    cache_key = str(cand)
                    probe = h5_probe_cache.get(cache_key)
                    if probe is None:
                        probe = _probe_h5_feature(cand, self.feature_key)
                        h5_probe_cache[cache_key] = probe
                        ok_probe, reason_probe = probe
                        if not ok_probe:
                            invalid_h5_unique.add(cache_key)
                            reason_counts[reason_probe] = reason_counts.get(reason_probe, 0) + 1

                    ok, _ = probe
                    if ok:
                        chosen = cand
                        if p_idx > 0:
                            self.stats["used_alternate_h5"] += 1
                        break

                if chosen is None:
                    self.stats["dropped_invalid_h5"] += 1
                    if self.missing_policy == "error":
                        raise RuntimeError(f"No valid h5 with usable features for id={case_id}")
                    continue
                selected_paths = (chosen,)
            else:
                # Fast path: defer h5 content validation to __getitem__.
                selected_paths = tuple(h5_paths)

            self.samples.append(
                AlignmentSample(
                    case_id=case_id,
                    report_text=report,
                    h5_paths=selected_paths,
                )
            )
            self.stats["matched_rows"] += 1

        self.invalid_h5_paths = sorted(invalid_h5_unique)
        self.invalid_h5_reason_counts = dict(sorted(reason_counts.items(), key=lambda kv: kv[0]))
        self.stats["invalid_h5_files_unique"] = len(self.invalid_h5_paths)
        self.stats["h5_probed_unique"] = len(h5_probe_cache)
        for reason, cnt in self.invalid_h5_reason_counts.items():
            self.stats[f"invalid_h5_{reason}"] = int(cnt)

        if not self.samples and self.missing_policy == "error":
            raise RuntimeError("No matched rows after embedding filtering.")

    def __len__(self) -> int:
        return len(self.samples)

    def _record_runtime_invalid_h5(self, path: Path, reason: str) -> None:
        p = str(path)
        if p not in self._runtime_invalid_h5_paths:
            self._runtime_invalid_h5_paths.add(p)
            self.invalid_h5_paths = sorted(set(self.invalid_h5_paths) | self._runtime_invalid_h5_paths)

        self._runtime_invalid_h5_reason_counts[reason] = self._runtime_invalid_h5_reason_counts.get(reason, 0) + 1
        self.invalid_h5_reason_counts[reason] = self.invalid_h5_reason_counts.get(reason, 0) + 1

        self.stats["invalid_h5_files_unique"] = len(self.invalid_h5_paths)
        self.stats[f"invalid_h5_{reason}"] = int(self.invalid_h5_reason_counts.get(reason, 0))
        self.stats["dropped_invalid_h5"] = int(self.stats.get("dropped_invalid_h5", 0)) + 1

    def _try_load_case_item(self, sample: AlignmentSample) -> CaseItem | None:
        for cand_idx, h5_path in enumerate(sample.h5_paths):
            try:
                feats = _read_h5_features(h5_path, self.feature_key)
            except Exception as exc:
                reason = _classify_h5_exception(exc if isinstance(exc, Exception) else Exception(str(exc)))
                self._record_runtime_invalid_h5(h5_path, reason)
                continue

            if cand_idx > 0:
                self.stats["used_alternate_h5"] = int(self.stats.get("used_alternate_h5", 0)) + 1

            vision = torch.from_numpy(feats)
            n = int(feats.shape[0])
            return CaseItem(
                vision_embeddings=vision,
                wsi_patch_counts=torch.tensor([n], dtype=torch.long),
                report_text=sample.report_text,
                case_id=sample.case_id,
                slide_path=str(h5_path),
            )
        return None

    def __getitem__(self, idx: int) -> CaseItem:
        n_samples = len(self.samples)
        if n_samples <= 0:
            raise IndexError("Dataset is empty.")

        # Try requested sample first.
        s = self.samples[idx]
        item = self._try_load_case_item(s)
        if item is not None:
            return item

        # Fallback: search forward for the next readable sample to avoid worker crash.
        for off in range(1, n_samples):
            j = (idx + off) % n_samples
            alt = self._try_load_case_item(self.samples[j])
            if alt is not None:
                return alt

        if self.missing_policy == "error":
            raise RuntimeError("No valid samples with readable h5 features found.")
        raise RuntimeError("All candidate h5 files failed to load.")


class VLMDataCollator:
    def __init__(
        self,
        *,
        tokenizer: Any | None,
        max_text_length: int = 512,
        vision_pad_value: float = 0.0,
        prompt_style: PromptStyle = "single",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = int(max_text_length)
        self.vision_pad_value = float(vision_pad_value)
        self.prompt_style = str(prompt_style).strip().lower()
        self.prompt_text = resolve_prompt_text(self.prompt_style)

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

        raw_reports = [x.report_text for x in batch]
        case_ids = [x.case_id for x in batch]
        slide_paths = [x.slide_path for x in batch]

        if self.tokenizer is None:
            return {
                "vision_embeddings": vision,
                "vision_attention_mask": vision_mask,
                "wsi_patch_counts": wsi_patch_counts,
                "total_patch_counts": lengths,
                "input_text": raw_reports,
                "raw_report_text": raw_reports,
                "case_id": case_ids,
                "slide_paths": slide_paths,
            }

        if not hasattr(self.tokenizer, "apply_chat_template") or self.tokenizer.chat_template is None:
            raise ValueError(
                "Tokenizer has no chat_template/apply_chat_template. "
                "Use an Instruct tokenizer (e.g., Qwen2.5-Instruct) or set tokenizer.chat_template."
            )

        pad_id = int(self.tokenizer.pad_token_id)
        per_input_ids: list[torch.Tensor] = []
        per_labels: list[torch.Tensor] = []

        for item in batch:
            assistant_msg = item.report_text.strip()

            messages_full = [
                {"role": "user", "content": self.prompt_text},
                {"role": "assistant", "content": assistant_msg},
            ]
            messages_prompt = [
                {"role": "user", "content": self.prompt_text},
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

            full_ids = _extract_input_ids_1d(full_ids)
            prompt_ids = _extract_input_ids_1d(prompt_ids)

            ids_1d = full_ids
            labels_1d = ids_1d.clone()

            prompt_len = min(int(prompt_ids.numel()), int(labels_1d.numel()))
            labels_1d[:prompt_len] = -100

            if ids_1d.numel() > self.max_text_length:
                ids_1d = ids_1d[: self.max_text_length]
                labels_1d = labels_1d[: self.max_text_length]

            per_input_ids.append(ids_1d)
            per_labels.append(labels_1d)

        max_t = max(int(x.numel()) for x in per_input_ids)
        bsz = len(per_input_ids)

        input_ids = torch.full((bsz, max_t), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_t), dtype=torch.long)
        labels = torch.full((bsz, max_t), -100, dtype=torch.long)

        for i, (ids_1d, labs_1d) in enumerate(zip(per_input_ids, per_labels)):
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
            "raw_report_text": raw_reports,
            "case_id": case_ids,
            "slide_paths": slide_paths,
        }


def create_train_val_dataloaders(
    *,
    metadata_json: Path,
    dataset_embeddings_root: Path,
    tokenizer: Any,
    feature_key: str = "conch_v15",
    patch_level: str = "5x_512",
    datasets: str = "all",
    prompt_style: PromptStyle = "single",
    batch_size: int = 1,
    num_workers: int = 0,
    max_text_length: int = 512,
    missing_policy: MissingPolicy = "skip",
    probe_h5_on_init: bool = False,
    enforce_readable_h5: bool = True,
    val_size: int | float | str = 500,
    split_seed: int = 42,
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    base = AlignmentCaseDataset(
        metadata_json=Path(metadata_json),
        dataset_embeddings_root=Path(dataset_embeddings_root),
        feature_key=feature_key,
        patch_level=patch_level,
        datasets=datasets,
        missing_policy=missing_policy,
        probe_h5_on_init=bool(probe_h5_on_init),
    )

    case_ids = [x.case_id for x in base.samples]
    train_idx, val_idx = make_train_val_split(case_ids, val_size=val_size, seed=int(split_seed))

    if bool(enforce_readable_h5):
        probe_cache: dict[str, tuple[bool, str]] = {}

        train_idx, train_h5_stats, train_invalid_paths, train_reason_counts, train_used_alt = _filter_indices_to_readable_h5(
            base_dataset=base,
            indices=train_idx,
            split_name="train",
            probe_cache=probe_cache,
        )
        val_idx, val_h5_stats, val_invalid_paths, val_reason_counts, val_used_alt = _filter_indices_to_readable_h5(
            base_dataset=base,
            indices=val_idx,
            split_name="val",
            probe_cache=probe_cache,
        )

        # Merge split-level invalid h5 stats into dataset-level diagnostics.
        invalid_paths_all = set(base.invalid_h5_paths) | train_invalid_paths | val_invalid_paths
        merged_reason_counts = dict(base.invalid_h5_reason_counts)
        for rc in (train_reason_counts, val_reason_counts):
            for k, v in rc.items():
                merged_reason_counts[k] = int(merged_reason_counts.get(k, 0)) + int(v)

        base.invalid_h5_paths = sorted(invalid_paths_all)
        base.invalid_h5_reason_counts = dict(sorted(merged_reason_counts.items(), key=lambda kv: kv[0]))

        base.stats["split_h5_filter_enabled"] = 1
        base.stats["train_input_before_h5_filter"] = int(train_h5_stats["input_rows"])
        base.stats["train_kept_after_h5_filter"] = int(train_h5_stats["kept_rows"])
        base.stats["train_dropped_invalid_h5_rows"] = int(train_h5_stats["dropped_invalid_h5_rows"])
        base.stats["val_input_before_h5_filter"] = int(val_h5_stats["input_rows"])
        base.stats["val_kept_after_h5_filter"] = int(val_h5_stats["kept_rows"])
        base.stats["val_dropped_invalid_h5_rows"] = int(val_h5_stats["dropped_invalid_h5_rows"])

        base.stats["used_alternate_h5"] = int(base.stats.get("used_alternate_h5", 0)) + int(train_used_alt) + int(val_used_alt)
        base.stats["dropped_invalid_h5"] = int(base.stats.get("dropped_invalid_h5", 0)) + int(
            train_h5_stats["dropped_invalid_h5_rows"]
        ) + int(val_h5_stats["dropped_invalid_h5_rows"])
        base.stats["invalid_h5_files_unique"] = int(len(base.invalid_h5_paths))
        for reason, cnt in base.invalid_h5_reason_counts.items():
            base.stats[f"invalid_h5_{reason}"] = int(cnt)

        base.stats["split_h5_filter_stats"] = {
            "train": train_h5_stats,
            "val": val_h5_stats,
        }

    train_ds = Subset(base, train_idx)
    val_ds = Subset(base, val_idx)

    collator = VLMDataCollator(
        tokenizer=tokenizer,
        max_text_length=max_text_length,
        prompt_style=prompt_style,
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
