from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ENV_DATA_ROOT = "PATHOSYNVLM_DATA_ROOT"
ENV_RAW_DATA_ROOT = "PATHOSYNVLM_RAW_DATA_ROOT"
ENV_EMBEDDINGS_ROOT = "PATHOSYNVLM_EMBEDDINGS_ROOT"
ENV_STAGE1_METADATA_DIR = "PATHOSYNVLM_STAGE1_METADATA_DIR"
ENV_HISTAI_METADATA_DIR = "PATHOSYNVLM_HISTAI_METADATA_DIR"
ENV_RUNS_ROOT = "PATHOSYNVLM_RUNS_ROOT"
ENV_WEIGHTS_ROOT = "PATHOSYNVLM_WEIGHTS_ROOT"


@dataclass(frozen=True)
class PathDefaults:
    data_root: Path
    raw_data_root: Path
    embeddings_root: Path
    stage1_metadata_dir: Path
    histai_metadata_dir: Path
    runs_root: Path
    weights_root: Path


def _expand_path(value: str | os.PathLike[str] | Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw:
        return _expand_path(raw)
    return default


def get_path_defaults(repo_root: Path) -> PathDefaults:
    """Resolve repository path defaults, allowing users to override local storage roots."""

    repo_root = Path(repo_root).resolve()
    data_root = _env_path(ENV_DATA_ROOT, repo_root / "data")
    raw_data_root = _env_path(ENV_RAW_DATA_ROOT, data_root / "raw")
    embeddings_root = _env_path(ENV_EMBEDDINGS_ROOT, data_root / "embeddings")
    stage1_metadata_dir = _env_path(ENV_STAGE1_METADATA_DIR, data_root / "stage1")
    histai_metadata_dir = _env_path(ENV_HISTAI_METADATA_DIR, data_root / "histai")
    runs_root = _env_path(ENV_RUNS_ROOT, repo_root / "runs")
    weights_root = _env_path(ENV_WEIGHTS_ROOT, repo_root / "weights")

    return PathDefaults(
        data_root=data_root,
        raw_data_root=raw_data_root,
        embeddings_root=embeddings_root,
        stage1_metadata_dir=stage1_metadata_dir,
        histai_metadata_dir=histai_metadata_dir,
        runs_root=runs_root,
        weights_root=weights_root,
    )


def resolve_existing_or_rooted(path: Path, root: Path) -> Path:
    """Resolve a path that may be absolute, current-directory relative, or root relative."""

    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return Path(root).expanduser() / p
