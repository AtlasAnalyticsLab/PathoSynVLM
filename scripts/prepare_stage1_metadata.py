from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StandardizedRecord:
    source_dataset: str
    id_filename: str
    report: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stem(name: str) -> str:
    return Path(str(name).strip()).stem.lower()


def _histgen_case_prefix(text: str) -> str:
    s = Path(str(text).strip()).stem
    if not s:
        return ""
    return s.split(".", 1)[0].lower()


def _pathtext_case_prefix(text: str) -> str:
    s = Path(str(text).strip()).stem
    if not s:
        return ""
    m = re.match(r"^(TCGA-[^-]+-[^-]+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return s.lower()


def _collect_conch_h5_paths(dataset_embeddings_root: Path, patch_level: str) -> dict[str, list[Path]]:
    root = Path(dataset_embeddings_root)
    out: dict[str, list[Path]] = {
        "histgen": [],
        "pathtext": [],
        "reg_dataset": [],
    }

    histgen_roots = [
        root / "HistGen-train" / "conch_v15" / patch_level / "patches",
        root / "HistGen-val" / "conch_v15" / patch_level / "patches",
        root / "HistGen-test" / "conch_v15" / patch_level / "patches",
    ]
    pathtext_roots = [root / "PathText" / "conch_v15" / patch_level / "patches"]
    reg_roots = [
        root / "REG_dataset" / "REG_train" / "conch_v15" / patch_level / "patches",
        root / "REG_dataset" / "REG_test" / "REG_test1" / "conch_v15" / patch_level / "patches",
        root / "REG_dataset" / "REG_test" / "REG_test2_revised" / "conch_v15" / patch_level / "patches",
    ]

    for p in histgen_roots:
        if p.is_dir():
            out["histgen"].extend(sorted(p.glob("*.h5")))
    for p in pathtext_roots:
        if p.is_dir():
            out["pathtext"].extend(sorted(p.glob("*.h5")))
    for p in reg_roots:
        if p.is_dir():
            out["reg_dataset"].extend(sorted(p.glob("*.h5")))

    return out


def _build_embedding_indexes(embedding_paths: dict[str, list[Path]]) -> dict[str, Any]:
    histgen_exact: set[str] = set()
    histgen_prefix: set[str] = set()
    pathtext_prefix_to_stems: dict[str, list[str]] = defaultdict(list)
    reg_exact: set[str] = set()

    for p in embedding_paths.get("histgen", []):
        stem = p.stem.lower()
        histgen_exact.add(stem)
        histgen_prefix.add(stem.split(".", 1)[0])

    for p in embedding_paths.get("pathtext", []):
        stem = p.stem.lower()
        pref = _pathtext_case_prefix(stem)
        pathtext_prefix_to_stems[pref].append(stem)

    for pref in list(pathtext_prefix_to_stems.keys()):
        pathtext_prefix_to_stems[pref] = sorted(set(pathtext_prefix_to_stems[pref]))

    for p in embedding_paths.get("reg_dataset", []):
        reg_exact.add(p.stem.lower())

    return {
        "histgen_exact": histgen_exact,
        "histgen_prefix": histgen_prefix,
        "pathtext_prefix_to_stems": pathtext_prefix_to_stems,
        "pathtext_prefix_set": set(pathtext_prefix_to_stems.keys()),
        "reg_exact": reg_exact,
    }


def build_standardized_records(
    *,
    histgen_json: Path,
    pathtext_json: Path | None,
    reg_json: Path,
    embedding_indexes: dict[str, Any],
) -> tuple[list[StandardizedRecord], dict[str, Any]]:
    stats: dict[str, Any] = {
        "histgen": {
            "source_rows": 0,
            "empty_text": 0,
            "standardized_rows": 0,
        },
        "pathtext": {
            "source_rows": 0,
            "empty_text": 0,
            "expanded_rows": 0,
            "prefix_without_embedding": 0,
        },
        "reg_dataset": {
            "source_rows": 0,
            "empty_text": 0,
            "standardized_rows": 0,
        },
    }

    standardized: list[StandardizedRecord] = []

    # HistGen
    histgen_data = _load_json(Path(histgen_json))
    histgen_rows: list[dict[str, Any]] = []
    if isinstance(histgen_data, dict):
        for split_key in ("train", "val", "test"):
            arr = histgen_data.get(split_key, [])
            if isinstance(arr, list):
                histgen_rows.extend(x for x in arr if isinstance(x, dict))
    elif isinstance(histgen_data, list):
        histgen_rows = [x for x in histgen_data if isinstance(x, dict)]
    else:
        raise ValueError("HistGen annotation JSON must be dict or list")

    stats["histgen"]["source_rows"] = len(histgen_rows)
    for row in histgen_rows:
        raw_id = str(row.get("id") or "").strip()
        report = str(row.get("report") or "").strip()
        if not raw_id:
            continue
        if not report:
            stats["histgen"]["empty_text"] += 1
            continue

        filename = raw_id if Path(raw_id).suffix else f"{raw_id}.svs"
        standardized.append(StandardizedRecord(source_dataset="histgen", id_filename=filename, report=report))
        stats["histgen"]["standardized_rows"] += 1

    # PathText is retained as an optional compatibility path for non-default ablations.
    # The paper-default path uses HistGen + REG2025.
    if pathtext_json is not None:
        pathtext_data = _load_json(Path(pathtext_json))
        if not isinstance(pathtext_data, list):
            raise ValueError("PathText JSON must be a list")

        prefix_to_stems: dict[str, list[str]] = embedding_indexes["pathtext_prefix_to_stems"]

        stats["pathtext"]["source_rows"] = len(pathtext_data)
        for row in pathtext_data:
            if not isinstance(row, dict):
                continue
            raw_id = str(row.get("id") or "").strip()
            caption = str(row.get("caption") or "").strip()
            if not raw_id:
                continue
            if not caption:
                stats["pathtext"]["empty_text"] += 1
                continue

            pref = _pathtext_case_prefix(raw_id)
            stems = prefix_to_stems.get(pref, [])
            if not stems:
                stats["pathtext"]["prefix_without_embedding"] += 1
                stems = [raw_id.lower()]

            for st in stems:
                filename = f"{st}.svs"
                standardized.append(StandardizedRecord(source_dataset="pathtext", id_filename=filename, report=caption))
                stats["pathtext"]["expanded_rows"] += 1

    # REG
    reg_data = _load_json(Path(reg_json))
    if not isinstance(reg_data, list):
        raise ValueError("REG JSON must be a list")

    stats["reg_dataset"]["source_rows"] = len(reg_data)
    for row in reg_data:
        if not isinstance(row, dict):
            continue
        raw_id = str(row.get("id") or "").strip()
        report = str(row.get("report") or "").strip()
        if not raw_id:
            continue
        if not report:
            stats["reg_dataset"]["empty_text"] += 1
            continue

        standardized.append(StandardizedRecord(source_dataset="reg_dataset", id_filename=raw_id, report=report))
        stats["reg_dataset"]["standardized_rows"] += 1

    return standardized, stats


def filter_records_by_embeddings(
    records: list[StandardizedRecord],
    *,
    embedding_paths: dict[str, list[Path]],
    embedding_indexes: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "histgen": {"embedding_files": 0, "input_rows": 0, "kept": 0, "dropped_no_embedding": 0},
        "pathtext": {"embedding_files": 0, "input_rows": 0, "kept": 0, "dropped_no_embedding": 0},
        "reg_dataset": {"embedding_files": 0, "input_rows": 0, "kept": 0, "dropped_no_embedding": 0},
    }

    histgen_exact: set[str] = embedding_indexes["histgen_exact"]
    histgen_prefix: set[str] = embedding_indexes["histgen_prefix"]
    pathtext_prefix_set: set[str] = embedding_indexes["pathtext_prefix_set"]
    reg_exact: set[str] = embedding_indexes["reg_exact"]

    stats["histgen"]["embedding_files"] = len(embedding_paths.get("histgen", []))
    stats["pathtext"]["embedding_files"] = len(embedding_paths.get("pathtext", []))
    stats["reg_dataset"]["embedding_files"] = len(embedding_paths.get("reg_dataset", []))

    filtered_records: list[StandardizedRecord] = []
    for rec in records:
        src = rec.source_dataset
        stats[src]["input_rows"] += 1
        rec_stem = _stem(rec.id_filename)

        keep = False
        if src == "histgen":
            keep = rec_stem in histgen_exact or _histgen_case_prefix(rec_stem) in histgen_prefix
        elif src == "pathtext":
            keep = _pathtext_case_prefix(rec_stem) in pathtext_prefix_set
        elif src == "reg_dataset":
            keep = rec_stem in reg_exact

        if keep:
            filtered_records.append(rec)
            stats[src]["kept"] += 1
        else:
            stats[src]["dropped_no_embedding"] += 1

    # Final output only two fields: id + report
    seen: set[tuple[str, str]] = set()
    filtered: list[dict[str, str]] = []
    for rec in filtered_records:
        pair = (rec.id_filename, rec.report)
        if pair in seen:
            continue
        seen.add(pair)
        filtered.append({"id": rec.id_filename, "report": rec.report})

    stats["total_input_rows"] = len(records)
    stats["total_filtered_rows_before_dedup"] = len(filtered_records)
    stats["total_filtered_rows"] = len(filtered)
    stats["dedup_removed"] = len(filtered_records) - len(filtered)

    return filtered, stats


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_dir = repo_root / "data" / "stage1"

    parser = argparse.ArgumentParser(description="Merge and filter HistGen/REG metadata for Stage 1 alignment training.")
    parser.add_argument("--histgen-json", type=Path, default=repo_root / "data" / "raw" / "histgen" / "annotation_update.json")
    parser.add_argument("--pathtext-json", type=Path, default=repo_root / "data" / "raw" / "pathtext" / "PathText.json")
    parser.add_argument("--include-pathtext", action="store_true", help="Include PathText compatibility data (not used in the paper default).")
    parser.add_argument("--reg-json", type=Path, default=repo_root / "data" / "raw" / "reg2025" / "train.json")

    parser.add_argument(
        "--dataset-embeddings-root",
        type=Path,
        default=repo_root / "data" / "embeddings",
    )
    parser.add_argument("--patch-level", default="5x_512")

    parser.add_argument(
        "--raw-output-json",
        type=Path,
        default=default_dir / "merged_metadata_3datasets_raw.json",
    )
    parser.add_argument(
        "--filtered-output-json",
        type=Path,
        default=default_dir / "merged_metadata_3datasets_filtered_conch_v15.json",
    )
    parser.add_argument(
        "--stats-output-json",
        type=Path,
        default=default_dir / "merged_metadata_3datasets_filtered_conch_v15_stats.json",
    )

    args = parser.parse_args()

    embedding_paths = _collect_conch_h5_paths(
        dataset_embeddings_root=args.dataset_embeddings_root,
        patch_level=str(args.patch_level),
    )
    embedding_indexes = _build_embedding_indexes(embedding_paths)

    standardized_records, standardize_stats = build_standardized_records(
        histgen_json=args.histgen_json,
        pathtext_json=(args.pathtext_json if bool(args.include_pathtext) else None),
        reg_json=args.reg_json,
        embedding_indexes=embedding_indexes,
    )

    raw_rows = [{"id": r.id_filename, "report": r.report} for r in standardized_records]

    filtered_rows, filter_stats = filter_records_by_embeddings(
        standardized_records,
        embedding_paths=embedding_paths,
        embedding_indexes=embedding_indexes,
    )

    args.raw_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.filtered_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output_json.parent.mkdir(parents=True, exist_ok=True)

    args.raw_output_json.write_text(json.dumps(raw_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    args.filtered_output_json.write_text(json.dumps(filtered_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    all_stats = {
        "inputs": {
            "histgen_json": str(args.histgen_json),
            "pathtext_json": (str(args.pathtext_json) if bool(args.include_pathtext) else None),
            "include_pathtext": bool(args.include_pathtext),
            "reg_json": str(args.reg_json),
            "dataset_embeddings_root": str(args.dataset_embeddings_root),
            "feature_key": "conch_v15",
            "patch_level": str(args.patch_level),
        },
        "raw_rows": len(raw_rows),
        "filtered_rows": len(filtered_rows),
        "standardize_stats": standardize_stats,
        "filter_stats": filter_stats,
        "outputs": {
            "raw_output_json": str(args.raw_output_json),
            "filtered_output_json": str(args.filtered_output_json),
            "stats_output_json": str(args.stats_output_json),
        },
    }
    args.stats_output_json.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Wrote raw metadata:", args.raw_output_json)
    print("Wrote filtered metadata:", args.filtered_output_json)
    print("Wrote stats:", args.stats_output_json)
    print("raw_rows=", len(raw_rows), "filtered_rows=", len(filtered_rows))


if __name__ == "__main__":
    main()
