from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathosynvlm.paths import get_path_defaults


def _parse_case_mapping(case_mapping: str) -> tuple[str, int] | None:
    s = str(case_mapping or "").strip()
    if not s:
        return None
    parts = [p for p in re.split(r"[\\/]+", s) if p]
    if len(parts) < 3:
        return None

    group_raw = parts[-2].lower()
    group = group_raw[len("histai-") :] if group_raw.startswith("histai-") else group_raw

    case_raw = parts[-1].lower()
    m = re.match(r"^case_(\d+)$", case_raw)
    if not m:
        return None

    return group, int(m.group(1))


def _parse_case_from_h5(filename: str) -> tuple[str, int] | None:
    s = str(filename).strip().lower()
    m = re.match(r"^(?P<group>.+?)_case_(?P<num>\d+)(?:_|$)", s)
    if not m:
        return None
    return m.group("group"), int(m.group("num"))


def _canonical_case_mapping(key: tuple[str, int]) -> str:
    group, case_num = key
    return f"histai/HISTAI-{group}/case_{case_num}"


def _load_standardized_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows: list[dict[str, Any]] = []
    for row in data:
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _dedup_standardized_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_case_mapping_rows = 0
    invalid_case_mapping_rows = 0
    duplicate_case_rows = 0

    seen: set[tuple[str, int]] = set()
    dedup_rows: list[dict[str, Any]] = []
    dedup_keys: list[tuple[str, int]] = []

    for row in rows:
        case_mapping = str((row or {}).get("case_mapping") or "").strip()
        if not case_mapping:
            missing_case_mapping_rows += 1
            continue

        key = _parse_case_mapping(case_mapping)
        if key is None:
            invalid_case_mapping_rows += 1
            continue

        if key in seen:
            duplicate_case_rows += 1
            continue

        seen.add(key)
        new_row = dict(row)
        new_row["case_mapping"] = _canonical_case_mapping(key)
        dedup_rows.append(new_row)
        dedup_keys.append(key)

    return {
        "rows": dedup_rows,
        "keys": dedup_keys,
        "stats": {
            "input_rows": int(len(rows)),
            "rows_missing_case_mapping": int(missing_case_mapping_rows),
            "rows_invalid_case_mapping": int(invalid_case_mapping_rows),
            "rows_dropped_as_duplicates": int(duplicate_case_rows),
            "unique_cases_after_dedup": int(len(dedup_rows)),
        },
    }


def _collect_case_paths(
    *,
    dataset_embeddings_root: Path,
    feature_key: str,
    patch_level: str,
) -> dict[tuple[str, int], list[Path]]:
    out: dict[tuple[str, int], list[Path]] = {}
    for histai_dir in sorted(dataset_embeddings_root.glob("HISTAI-*")):
        patch_dir = histai_dir / feature_key / patch_level / "patches"
        if not patch_dir.is_dir():
            continue

        for p in patch_dir.glob("*.h5"):
            key = _parse_case_from_h5(p.name)
            if key is None:
                continue
            out.setdefault(key, []).append(p)

    for key in list(out.keys()):
        out[key] = sorted(set(out[key]), key=lambda x: str(x))

    return out


def _probe_h5_feature(path: Path, feature_key: str) -> tuple[bool, str]:
    try:
        with h5py.File(path, "r") as f:
            features_obj = f.get("features")
            if features_obj is None:
                return False, "missing_features_group"
            if not isinstance(features_obj, h5py.Group):
                return False, "features_not_group"

            ds = features_obj.get(feature_key)
            if ds is None:
                keys = list(features_obj.keys())
                if len(keys) != 1:
                    return False, "missing_feature_key"
                ds = features_obj.get(keys[0])

            if not isinstance(ds, h5py.Dataset):
                return False, "feature_not_dataset"

            shape = getattr(ds, "shape", None)
            if shape is None or len(shape) != 2:
                return False, "invalid_shape"
            if int(shape[0]) <= 0 or int(shape[1]) <= 0:
                return False, "empty_shape"
            return True, "ok"
    except OSError:
        return False, "open_error"
    except Exception:
        return False, "other_error"


def _evaluate_valid_cases(
    *,
    case_paths: dict[tuple[str, int], list[Path]],
    feature_key: str,
    probe_valid_embeddings: bool,
) -> dict[str, Any]:
    all_cases = set(case_paths.keys())

    if not probe_valid_embeddings:
        return {
            "all_cases": all_cases,
            "valid_cases": set(all_cases),
            "h5_total": int(sum(len(v) for v in case_paths.values())),
            "h5_probed": 0,
            "invalid_h5_reason_counts": {},
            "cases_without_any_valid_h5": 0,
        }

    valid_cases: set[tuple[str, int]] = set()
    invalid_reasons: dict[str, int] = {}
    h5_probed = 0
    cases_without_any_valid_h5 = 0
    probe_cache: dict[str, tuple[bool, str]] = {}

    for case_key, paths in case_paths.items():
        case_ok = False
        for p in paths:
            p_str = str(p)
            result = probe_cache.get(p_str)
            if result is None:
                result = _probe_h5_feature(p, feature_key)
                probe_cache[p_str] = result
            ok, reason = result
            h5_probed += 1
            if ok:
                case_ok = True
                break
            invalid_reasons[reason] = int(invalid_reasons.get(reason, 0)) + 1

        if case_ok:
            valid_cases.add(case_key)
        else:
            cases_without_any_valid_h5 += 1

    return {
        "all_cases": all_cases,
        "valid_cases": valid_cases,
        "h5_total": int(sum(len(v) for v in case_paths.values())),
        "h5_probed": int(h5_probed),
        "invalid_h5_reason_counts": dict(sorted(invalid_reasons.items())),
        "cases_without_any_valid_h5": int(cases_without_any_valid_h5),
    }


def _write_stats_md(
    *,
    output_md: Path,
    metadata_path: Path,
    embeddings_root: Path,
    feature_key: str,
    patch_levels: list[str],
    probe_valid_embeddings: bool,
    dedup_stats: dict[str, Any],
    per_patch: dict[str, dict[str, Any]],
) -> None:
    md: list[str] = []
    md.append("# Filtered Standardized Metadata Stats")
    md.append("")
    md.append("## Inputs")
    md.append("")
    md.append(f"- Metadata: `{metadata_path}`")
    md.append(f"- Embeddings root: `{embeddings_root}`")
    md.append(f"- Feature key: `{feature_key}`")
    md.append(f"- Patch levels: `{', '.join(patch_levels)}`")
    md.append(
        "- Validity mode: "
        + (
            "`h5_readability_probed`"
            if probe_valid_embeddings
            else "`case_level_embedding_presence` (no per-file h5 probe)"
        )
    )
    md.append("")
    md.append("## Standardized Dedup Stats")
    md.append("")
    md.append(f"- Input rows: `{dedup_stats['input_rows']}`")
    md.append(f"- Rows missing `case_mapping`: `{dedup_stats['rows_missing_case_mapping']}`")
    md.append(f"- Rows invalid `case_mapping`: `{dedup_stats['rows_invalid_case_mapping']}`")
    md.append(f"- Duplicate case rows dropped: `{dedup_stats['rows_dropped_as_duplicates']}`")
    md.append(f"- Unique cases after dedup: `{dedup_stats['unique_cases_after_dedup']}`")
    md.append("")
    md.append("## Per Patch")
    md.append("")
    md.append("| Patch | Embedding Cases (All) | Embedding Cases (Valid) | Filtered Rows | Filtered Unique Cases | Std Unique Cases Without Embedding |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for patch_level in patch_levels:
        row = per_patch[patch_level]
        md.append(
            f"| {patch_level} | {row['embedding_cases_all']} | {row['embedding_cases_valid']} | {row['filtered_rows']} | {row['filtered_unique_cases']} | {row['standardized_unique_without_embedding']} |"
        )
    md.append("")

    output_md.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = REPO_ROOT
    paths = get_path_defaults(repo_root)
    p = argparse.ArgumentParser(
        description="Filter standardized HistAI metadata to cases that have embeddings for each patch level."
    )
    p.add_argument(
        "--metadata-standardized-json",
        type=Path,
        default=paths.raw_data_root / "histai" / "standardized_metadata_fixed.json",
    )
    p.add_argument(
        "--dataset-embeddings-root",
        type=Path,
        default=paths.embeddings_root,
        help="Root containing dataset embedding folders. Defaults to PATHOSYNVLM_EMBEDDINGS_ROOT or data/embeddings.",
    )
    p.add_argument("--feature-key", type=str, default="conch_v15")
    p.add_argument("--patch-levels", type=str, default="1x_512,5x_512")
    p.add_argument("--probe-valid-embeddings", action="store_true", default=False)
    p.add_argument("--no-probe-valid-embeddings", dest="probe_valid_embeddings", action="store_false")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=paths.histai_metadata_dir,
    )
    p.add_argument("--output-prefix", type=str, default="standardized_metadata_fixed_filtered")
    args = p.parse_args()

    patch_levels = [x.strip() for x in str(args.patch_levels).split(",") if x.strip()]
    if not patch_levels:
        raise ValueError("No patch levels provided")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    std_rows = _load_standardized_rows(args.metadata_standardized_json)
    dedup = _dedup_standardized_rows(std_rows)
    dedup_rows = dedup["rows"]
    dedup_keys = dedup["keys"]
    dedup_stats = dedup["stats"]

    key_to_row: dict[tuple[str, int], dict[str, Any]] = {}
    for key, row in zip(dedup_keys, dedup_rows):
        key_to_row[key] = row

    per_patch_summary: dict[str, dict[str, Any]] = {}
    report_payload: dict[str, Any] = {
        "inputs": {
            "metadata_standardized_json": str(args.metadata_standardized_json),
            "dataset_embeddings_root": str(args.dataset_embeddings_root),
            "feature_key": str(args.feature_key),
            "patch_levels": patch_levels,
            "probe_valid_embeddings": bool(args.probe_valid_embeddings),
            "validity_mode": (
                "h5_readability_probed"
                if bool(args.probe_valid_embeddings)
                else "case_level_embedding_presence"
            ),
        },
        "standardized_dedup_stats": dedup_stats,
        "outputs": {},
        "per_patch": {},
    }

    standardized_key_set = set(dedup_keys)

    for patch_level in patch_levels:
        case_paths = _collect_case_paths(
            dataset_embeddings_root=args.dataset_embeddings_root,
            feature_key=args.feature_key,
            patch_level=patch_level,
        )
        eval_info = _evaluate_valid_cases(
            case_paths=case_paths,
            feature_key=args.feature_key,
            probe_valid_embeddings=bool(args.probe_valid_embeddings),
        )

        valid_cases = set(eval_info["valid_cases"])
        keep_keys = standardized_key_set & valid_cases

        filtered_rows: list[dict[str, Any]] = []
        for key in dedup_keys:
            if key in keep_keys:
                filtered_rows.append(key_to_row[key])

        dropped_keys = sorted(standardized_key_set - valid_cases)

        output_json = output_dir / f"{args.output_prefix}_{patch_level}.json"
        output_json.write_text(json.dumps(filtered_rows, indent=2), encoding="utf-8")

        dropped_output = output_dir / f"{args.output_prefix}_{patch_level}_dropped_cases.txt"
        dropped_output.write_text(
            "".join(f"{_canonical_case_mapping(k)}\n" for k in dropped_keys),
            encoding="utf-8",
        )

        group_keep = Counter(k[0] for k in keep_keys)
        group_drop = Counter(k[0] for k in dropped_keys)
        group_embedding = Counter(k[0] for k in valid_cases)

        per_patch_summary[patch_level] = {
            "embedding_cases_all": int(len(eval_info["all_cases"])),
            "embedding_cases_valid": int(len(valid_cases)),
            "h5_files_total": int(eval_info["h5_total"]),
            "h5_files_probed": int(eval_info["h5_probed"]),
            "invalid_h5_reason_counts": eval_info["invalid_h5_reason_counts"],
            "cases_without_any_valid_h5": int(eval_info["cases_without_any_valid_h5"]),
            "filtered_rows": int(len(filtered_rows)),
            "filtered_unique_cases": int(len(keep_keys)),
            "standardized_unique_without_embedding": int(len(dropped_keys)),
            "standardized_unique_without_embedding_examples": [
                _canonical_case_mapping(k) for k in dropped_keys[:100]
            ],
            "group_counts_filtered_unique_cases": dict(sorted(group_keep.items())),
            "group_counts_standardized_missing_embedding": dict(sorted(group_drop.items())),
            "group_counts_valid_embedding_cases": dict(sorted(group_embedding.items())),
        }

        report_payload["outputs"][patch_level] = {
            "filtered_json": str(output_json),
            "dropped_cases_txt": str(dropped_output),
        }
        report_payload["per_patch"][patch_level] = per_patch_summary[patch_level]

        print(
            f"[{patch_level}] filtered_rows={len(filtered_rows)} "
            f"filtered_unique_cases={len(keep_keys)} "
            f"standardized_unique_without_embedding={len(dropped_keys)}"
        )

    stats_json = output_dir / "standardized_metadata_filter_stats.json"
    stats_md = output_dir / "standardized_metadata_filter_stats.md"
    stats_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    _write_stats_md(
        output_md=stats_md,
        metadata_path=args.metadata_standardized_json,
        embeddings_root=args.dataset_embeddings_root,
        feature_key=args.feature_key,
        patch_levels=patch_levels,
        probe_valid_embeddings=bool(args.probe_valid_embeddings),
        dedup_stats=dedup_stats,
        per_patch=per_patch_summary,
    )

    print(f"Wrote stats JSON: {stats_json}")
    print(f"Wrote stats Markdown: {stats_md}")


if __name__ == "__main__":
    main()
