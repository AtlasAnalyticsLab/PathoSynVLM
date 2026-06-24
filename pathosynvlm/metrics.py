from __future__ import annotations

"""
Helpers for evaluating VLM outputs in this repo.

- `eval_vlm_output(text: str) -> tuple[int, dict[str, str], list[str]]`
  - Scores formatting from 0..3 (max=3):
    +1 for each required field line that appears anywhere in the text as `"<Prefix>: <non-empty>"`.
  - Returns `(format_score, fields, missing_fields)` where `fields` is best-effort extracted.

- `eval_diagnosis_match(standardized_diagnosis: str, predicted_diagnosis: str, threshold: float = 0.85)
      -> tuple[bool, list[str], list[str], list[str]]`
  - Relaxed matching on the *disease core* (often ignores body-location details).

- `eval_field_accuracy(predicted_text: str, reference_text: str, diagnosis_fuzzy_threshold: float = 0.85, certainty_percent_threshold: float = 50.0)
      -> dict[str, Any]`
  - Per-sample metrics: format OK, per-field exact matches, normalized certainty match,
    and relaxed diagnosis match (via `eval_diagnosis_match`). Conclusion similarity uses Rouge/METEOR/BLEU when available.

- `summarize_field_accuracy(predicted_texts: list[str], reference_texts: list[str], diagnosis_fuzzy_threshold: float = 0.85, certainty_percent_threshold: float = 50.0)
      -> dict[str, Any]`
  - Aggregates `eval_field_accuracy` across a batch (rates and counts).
"""

import re
from difflib import SequenceMatcher
from typing import Any

__all__ = [
    "eval_vlm_output",
    "eval_diagnosis_match",
    "normalize_certainty",
    "normalize_certainty_from_percent",
    "eval_field_accuracy",
    "summarize_field_accuracy",
    # Backwards-compatible aliases (do not add new codepaths):
    "extract_fields_best_effort",
    "check_output_format_strict",
]

_BASE_PREFIXES: list[tuple[str, str]] = [
    ("diagnosis", "Diagnosis:"),
    ("certainty", "Certainty:"),
]

_THIRD_FIELD_LABELS: dict[str, str] = {
    "conclusion": "Conclusion",
    "micro_protocol": "Micro protocol",
}


def normalize_target_field_name(third_field_name: str) -> str:
    name = str(third_field_name or "conclusion").strip().lower()
    if not name:
        return "conclusion"
    return name


def resolve_target_field_label(third_field_name: str, third_field_label: str | None = None) -> str:
    if third_field_label is not None and str(third_field_label).strip():
        return str(third_field_label).strip()
    name = normalize_target_field_name(third_field_name)
    return _THIRD_FIELD_LABELS.get(name, name.replace("_", " ").strip().title())


def _resolve_prefixes(
    third_field_name: str = "conclusion",
    third_field_label: str | None = None,
) -> list[tuple[str, str]]:
    third_key = normalize_target_field_name(third_field_name)
    third_label = resolve_target_field_label(third_key, third_field_label)
    return [*_BASE_PREFIXES, (third_key, f"{third_label}:")]


_METRICS_CACHE: dict[str, Any] = {}


def _load_default_text_metrics():
    rouge = _METRICS_CACHE.get("rouge")
    meteor = _METRICS_CACHE.get("meteor")
    bleu = _METRICS_CACHE.get("bleu")

    if rouge is None or meteor is None or bleu is None:
        import evaluate  # type: ignore

        rouge = rouge or evaluate.load("rouge")
        meteor = meteor or evaluate.load("meteor")
        bleu = bleu or evaluate.load("sacrebleu")
        _METRICS_CACHE["rouge"] = rouge
        _METRICS_CACHE["meteor"] = meteor
        _METRICS_CACHE["bleu"] = bleu

    return rouge, meteor, bleu


def eval_vlm_output(
    text: str,
    *,
    third_field_name: str = "conclusion",
    third_field_label: str | None = None,
) -> tuple[int, dict[str, str], list[str]]:
    """
    Score model output format (0..3):

      Diagnosis: <content>
      Certainty: <content>
      Conclusion: <content>

    Returns:
      - (format_score, fields, missing_fields)

    Notes:
      - `format_score` is +1 for each required field found anywhere as a line starting with the
        exact prefix (`Diagnosis:`, `Certainty:`, `Conclusion:`) and with non-empty content.
      - If `format_score < 3`, `fields` still contains any non-empty fields we can extract anywhere
        in the text, and `missing_fields` lists which of the 3 fields were missing/empty.
    """
    s = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = s.split("\n") if s else []
    prefixes = _resolve_prefixes(third_field_name=third_field_name, third_field_label=third_field_label)

    fields: dict[str, str] = {}
    for i, line in enumerate(lines):
        line = line.lstrip()

        # 1) Best-effort extraction (any order, any extra lines).
        for key, prefix in prefixes:
            if key in fields:
                continue
            if not line.startswith(prefix):
                continue
            value = line[len(prefix) :].strip()
            if value:
                fields[key] = value

    missing_fields = [k for k, _ in prefixes if k not in fields]
    format_score = sum(1 for k, _ in prefixes if k in fields)
    return int(format_score), fields, missing_fields


_NUM_ITEM_RE = re.compile(r"(?:^|[\s;])(\d{1,2})\s*[\)\.\-]\s*(?=[A-Za-z])")
_HASH_ITEM_RE = re.compile(r"(?:^|\s)#\s*(\d{1,2})\s*[-–—]\s*(?=[A-Za-z])")
_DATE_RE = re.compile(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})\b", re.I)
_TNM_RE = re.compile(r"\bT\d+[a-z]?\b|\bN\d+[a-z]?\b|\bM\d+[a-z]?\b", re.I)
_STAGE_RE = re.compile(r"\bstage\b[^,;]*", re.I)
_BIOPSY_RE = re.compile(r"\b(post[-\s]?(incisional|excisional)?\s*biopsy)\b[^,;]*", re.I)


def eval_diagnosis_match(
    standardized_diagnosis: str,
    predicted_diagnosis: str,
    *,
    threshold: float = 0.85,
) -> tuple[bool, list[str], list[str], list[str]]:
    """
    Evaluate whether the model's predicted diagnosis "matches" the standardized diagnosis.

    Matching is intentionally relaxed: we focus on the disease label, not detailed location.
    Example: "Basal cell carcinoma" matches "Basal cell carcinoma of the right infraorbital region".

    Returns:
      - (is_match, required_labels, predicted_labels, missing_labels)

    Notes:
      - Multi-diagnosis strings (e.g. "1. A; 2. B") are split into items.
      - Each item is reduced to a "disease core" label via simple heuristics (drops some noise
        like dates/staging and often drops trailing "of the <location>" segments).
      - A required label is considered matched if it is a substring of any predicted label (or vice versa),
        or their fuzzy similarity meets `threshold`.
    """

    def split_into_items(text: str) -> list[str]:
        if not isinstance(text, str):
            return []
        t = " ".join(text.strip().split())
        if not t:
            return []

        if _HASH_ITEM_RE.search(t):
            matches = list(_HASH_ITEM_RE.finditer(t))
            if len(matches) >= 2:
                spans: list[tuple[int, int]] = []
                for i, m in enumerate(matches):
                    start = m.end()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
                    spans.append((start, end))
                items = [t[a:b].strip(" ;") for a, b in spans]
                items = [x for x in items if x]
                return items if items else [t]

        matches = list(_NUM_ITEM_RE.finditer(t))
        if len(matches) >= 2:
            spans = []
            for i, m in enumerate(matches):
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
                spans.append((start, end))
            items = [t[a:b].strip(" ;") for a, b in spans]
            items = [x for x in items if x]
            return items if items else [t]

        return [t]

    def clean_noise(s: str) -> str:
        s = s.strip()
        s = re.sub(r"\([^)]*\)", " ", s)
        s = _DATE_RE.sub(" ", s)
        s = _TNM_RE.sub(" ", s)
        s = _STAGE_RE.sub(" ", s)
        s = _BIOPSY_RE.sub(" ", s)
        s = s.replace("’", "'")
        s = re.sub(r"[,:;]+", " ", s)
        return " ".join(s.split()).strip()

    def extract_disease_label(item: str) -> str:
        s = clean_noise(item).lower()
        if not s:
            return ""

        m = re.search(r"\b(of the skin)\b", s)
        if m:
            return s[: m.end()].strip()

        if " of the " in s:
            return s.split(" of the ", 1)[0].strip()

        return s

    def extract_disease_set(text: str) -> set[str]:
        items = split_into_items(text)
        labels = [extract_disease_label(x) for x in items]
        labels = [x for x in labels if x]
        return set(labels)

    def norm_for_match(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9\s'-]+", " ", s)
        return " ".join(s.split())

    def is_label_matched(required: str, predicted_labels: set[str]) -> bool:
        r = norm_for_match(required)
        if not r:
            return True

        for p in predicted_labels:
            pn = norm_for_match(p)
            if not pn:
                continue

            if r in pn or pn in r:
                return True

            if SequenceMatcher(None, r, pn).ratio() >= float(threshold):
                return True

        return False

    required = extract_disease_set(standardized_diagnosis)
    predicted = extract_disease_set(predicted_diagnosis)

    missing: list[str] = []
    for req in sorted(required):
        if not is_label_matched(req, predicted):
            missing.append(req)

    return (len(missing) == 0), sorted(required), sorted(predicted), missing


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_space(text: str) -> str:
    return " ".join(text.strip().split())


_FIRST_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def normalize_certainty_from_percent(value: str, *, percent_threshold: float = 50.0) -> str:
    """
    Interprets a numeric certainty as a percentage and thresholds into:
      - "confirmed"  if >= percent_threshold
      - "suspected"  if < percent_threshold

    Accepts:
      - "51%", "51", "0.51" (treated as 51%)

    Returns "" if no numeric value can be parsed.
    """
    v = _normalize_space(value)
    if not v:
        return ""

    m = _FIRST_NUMBER_RE.search(v)
    if not m:
        return ""

    try:
        num = float(m.group(0))
    except Exception:
        return ""

    is_percent_literal = "%" in v
    pct = num * 100.0 if (not is_percent_literal and 0.0 <= num <= 1.0) else num
    return "confirmed" if pct >= float(percent_threshold) else "suspected"


def normalize_certainty(value: str, *, percent_threshold: float = 50.0) -> str:
    """
    Normalize certainty strings into the expected labels:
      - "confirmed"
      - "suspected"

    If input is unknown/empty, returns "".
    """
    v = _normalize_space(value).lower()
    if not v:
        return ""

    # Most common labels in this repo.
    if v in {"confirmed", "confirm"}:
        return "confirmed"
    if v in {"suspected", "suspect"}:
        return "suspected"

    # Numeric percent values (common after feedback like "51%").
    numeric_norm = normalize_certainty_from_percent(v, percent_threshold=percent_threshold)
    if numeric_norm in {"confirmed", "suspected"}:
        return numeric_norm

    # Common synonyms that model outputs might produce.
    if any(x in v for x in ["uncertain", "possible", "probable", "likely", "maybe", "suspicious"]):
        return "suspected"
    if any(x in v for x in ["certain", "definite"]):
        return "confirmed"

    return v


def extract_fields_best_effort(
    text: str,
    *,
    third_field_name: str = "conclusion",
    third_field_label: str | None = None,
) -> dict[str, str]:
    # Backwards-compatible alias for older code paths.
    return eval_vlm_output(
        text,
        third_field_name=third_field_name,
        third_field_label=third_field_label,
    )[1]


def check_output_format_strict(
    text: str,
    *,
    third_field_name: str = "conclusion",
    third_field_label: str | None = None,
) -> tuple[bool, dict[str, str], list[str]]:
    # Backwards-compatible alias for older code paths.
    score, fields, missing = eval_vlm_output(
        text,
        third_field_name=third_field_name,
        third_field_label=third_field_label,
    )
    return (score == 3), fields, missing


def _normalized_text_for_exact_match(s: str) -> str:
    return _normalize_space(_normalize_newlines(s)).lower()


def eval_field_accuracy(
    *,
    predicted_text: str,
    reference_text: str,
    diagnosis_fuzzy_threshold: float = 0.85,
    certainty_percent_threshold: float = 50.0,
    third_field_name: str = "conclusion",
    third_field_label: str | None = None,
    rouge_metric=None,
    meteor_metric=None,
    bleu_metric=None,
) -> dict[str, Any]:
    """
    Compute structured evaluation for one prediction/reference pair.

    Returns a dict with:
      - strict format checks for prediction and reference
      - extracted fields (best-effort) for both
      - per-field exact matches (normalized string equality)
      - certainty match (normalized to confirmed/suspected when possible)
      - diagnosis relaxed match (via `eval_diagnosis_match`) when both diagnoses exist
    """
    third_key = normalize_target_field_name(third_field_name)
    prefixes = _resolve_prefixes(third_field_name=third_key, third_field_label=third_field_label)
    pred_format_score, pred_fields, pred_missing = eval_vlm_output(
        predicted_text,
        third_field_name=third_key,
        third_field_label=third_field_label,
    )
    ref_format_score, ref_fields, ref_missing = eval_vlm_output(
        reference_text,
        third_field_name=third_key,
        third_field_label=third_field_label,
    )

    out: dict[str, Any] = {
        "pred_format_score": int(pred_format_score),
        "pred_format_ok": bool(pred_format_score == 3),
        "pred_missing_fields": pred_missing,
        "ref_format_score": int(ref_format_score),
        "ref_format_ok": bool(ref_format_score == 3),
        "ref_missing_fields": ref_missing,
        "pred_fields": pred_fields,
        "ref_fields": ref_fields,
    }

    # Exact (normalized) field matches.
    for k, _ in prefixes:
        pv = pred_fields.get(k, "")
        rv = ref_fields.get(k, "")
        if pv and rv:
            out[f"{k}_exact_match"] = _normalized_text_for_exact_match(pv) == _normalized_text_for_exact_match(rv)
        else:
            out[f"{k}_exact_match"] = None

    # Certainty match with normalization.
    pred_cert = normalize_certainty(
        pred_fields.get("certainty", ""),
        percent_threshold=float(certainty_percent_threshold),
    )
    ref_cert = normalize_certainty(
        ref_fields.get("certainty", ""),
        percent_threshold=float(certainty_percent_threshold),
    )
    out["certainty_normalized_pred"] = pred_cert
    out["certainty_normalized_ref"] = ref_cert
    out["certainty_match"] = (pred_cert == ref_cert) if (pred_cert and ref_cert) else None

    # Diagnosis relaxed match (location-insensitive) + missing list.
    pred_diag = pred_fields.get("diagnosis", "")
    ref_diag = ref_fields.get("diagnosis", "")
    if pred_diag and ref_diag:
        diag_ok, required, predicted, missing = eval_diagnosis_match(
            ref_diag,
            pred_diag,
            threshold=float(diagnosis_fuzzy_threshold),
        )
        out["diagnosis_relaxed_match"] = diag_ok
        out["diagnosis_required_labels"] = required
        out["diagnosis_predicted_labels"] = predicted
        out["diagnosis_missing_labels"] = missing
    else:
        out["diagnosis_relaxed_match"] = None
        out["diagnosis_required_labels"] = []
        out["diagnosis_predicted_labels"] = []
        out["diagnosis_missing_labels"] = []

    # Conclusion similarity metrics (optional).
    pred_conc = pred_fields.get(third_key, "")
    ref_conc = ref_fields.get(third_key, "")
    out[f"{third_key}_rougeL"] = None
    out[f"{third_key}_meteor"] = None
    out[f"{third_key}_bleu4"] = None
    if pred_conc and ref_conc:
        try:
            if rouge_metric is None or meteor_metric is None or bleu_metric is None:
                rouge_metric, meteor_metric, bleu_metric = _load_default_text_metrics()

            rouge = rouge_metric.compute(
                predictions=[pred_conc],
                references=[ref_conc],
                rouge_types=["rougeL"],
                use_stemmer=True,
            )
            meteor = meteor_metric.compute(predictions=[pred_conc], references=[ref_conc])
            bleu = bleu_metric.compute(predictions=[pred_conc], references=[[ref_conc]])

            out[f"{third_key}_rougeL"] = float(rouge.get("rougeL", 0.0))
            out[f"{third_key}_meteor"] = float(meteor.get("meteor", 0.0))
            out[f"{third_key}_bleu4"] = float(bleu.get("score", 0.0))
        except Exception:
            pass

    return out


def summarize_field_accuracy(
    *,
    predicted_texts: list[str],
    reference_texts: list[str],
    diagnosis_fuzzy_threshold: float = 0.85,
    certainty_percent_threshold: float = 50.0,
    third_field_name: str = "conclusion",
    third_field_label: str | None = None,
    rouge_metric=None,
    meteor_metric=None,
    bleu_metric=None,
) -> dict[str, Any]:
    """
    Aggregate `eval_field_accuracy` over many samples.

    Returns rates (0..1) and counts. Intended to be logged from `main.py` validation.
    """
    if len(predicted_texts) != len(reference_texts):
        raise ValueError("predicted_texts and reference_texts must have the same length.")

    n = len(predicted_texts)
    third_key = normalize_target_field_name(third_field_name)
    prefixes = _resolve_prefixes(third_field_name=third_key, third_field_label=third_field_label)
    rows = [
        eval_field_accuracy(
            predicted_text=p,
            reference_text=r,
            diagnosis_fuzzy_threshold=diagnosis_fuzzy_threshold,
            certainty_percent_threshold=certainty_percent_threshold,
            third_field_name=third_key,
            third_field_label=third_field_label,
            rouge_metric=rouge_metric,
            meteor_metric=meteor_metric,
            bleu_metric=bleu_metric,
        )
        for p, r in zip(predicted_texts, reference_texts)
    ]

    def rate(key: str) -> float:
        vals = [x.get(key) for x in rows]
        vals = [v for v in vals if isinstance(v, bool)]
        return float(sum(1 for v in vals if v) / max(1, len(vals)))

    def count_true(key: str) -> int:
        return sum(1 for x in rows if x.get(key) is True)

    def score_counts(key: str) -> dict[str, int]:
        vals = [x.get(key) for x in rows]
        vals = [v for v in vals if isinstance(v, int)]
        out = {"0": 0, "1": 0, "2": 0, "3": 0}
        for v in vals:
            if v in (0, 1, 2, 3):
                out[str(v)] += 1
        return out

    summary: dict[str, Any] = {
        "n": n,
        "pred_format_score_counts": score_counts("pred_format_score"),
        "ref_format_score_counts": score_counts("ref_format_score"),
        "diagnosis_relaxed_match_rate": rate("diagnosis_relaxed_match"),
        "certainty_match_rate": rate("certainty_match"),
        "diagnosis_exact_match_rate": rate("diagnosis_exact_match"),
        "certainty_exact_match_rate": rate("certainty_exact_match"),
        f"{third_key}_exact_match_rate": rate(f"{third_key}_exact_match"),
    }

    # Field presence rates (best-effort parsing).
    for field, _ in prefixes:
        present = 0
        for x in rows:
            if x.get("pred_fields", {}).get(field, ""):
                present += 1
        summary[f"pred_{field}_present_rate"] = float(present / max(1, n))

    # Conclusion similarity metrics on extracted conclusions (preferred over exact match).
    pred_concs = []
    ref_concs = []
    for x in rows:
        pc = str(x.get("pred_fields", {}).get(third_key, "") or "").strip()
        rc = str(x.get("ref_fields", {}).get(third_key, "") or "").strip()
        if pc and rc:
            pred_concs.append(pc)
            ref_concs.append(rc)

    summary[f"{third_key}_metrics_n"] = int(min(len(pred_concs), len(ref_concs)))
    summary[f"{third_key}_rougeL"] = None
    summary[f"{third_key}_meteor"] = None
    summary[f"{third_key}_bleu4"] = None
    if pred_concs and ref_concs:
        try:
            if rouge_metric is None or meteor_metric is None or bleu_metric is None:
                rouge_metric, meteor_metric, bleu_metric = _load_default_text_metrics()

            rouge = rouge_metric.compute(
                predictions=pred_concs,
                references=ref_concs,
                rouge_types=["rougeL"],
                use_stemmer=True,
            )
            meteor = meteor_metric.compute(predictions=pred_concs, references=ref_concs)
            bleu = bleu_metric.compute(predictions=pred_concs, references=[[r] for r in ref_concs])
            summary[f"{third_key}_rougeL"] = float(rouge.get("rougeL", 0.0))
            summary[f"{third_key}_meteor"] = float(meteor.get("meteor", 0.0))
            summary[f"{third_key}_bleu4"] = float(bleu.get("score", 0.0))
        except Exception:
            pass

    return summary
