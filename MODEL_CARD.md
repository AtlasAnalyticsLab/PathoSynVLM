# PathoSynVLM Model Card

[Paper](https://arxiv.org/abs/2605.30716) | [Code](https://github.com/AtlasAnalyticsLab/PathoSynVLM) | [Model weights](https://huggingface.co/AtlasAnalyticsLab/PathoSynVLM) | [Training data](https://huggingface.co/datasets/histai/HISTAI-metadata) | [Project page](https://atlasanalyticslab.github.io/PathoSynVLM/)

## Model

PathoSynVLM is a token-efficient vision-language model for case-level pathology synoptic report generation.

Architecture:

- Frozen WSI patch encoder: CONCHv1.5
- Vision-language bridge: two-layer MLP aligner
- Language decoder: Qwen2.5-3B-Instruct
- Case structure: one or more WSI embedding files with optional WSI marker tokens

## Intended Use

The model is intended for research on pathology report generation from precomputed WSI patch embeddings. Given one or more slide embedding files for a case, it generates:

```text
Diagnosis: ...
Certainty: ...
Conclusion: ...
```

This model is not a clinical diagnostic device and should not be used for patient care without appropriate validation, regulatory review, and expert oversight.

## Training Data

The paper-relevant pipeline uses:

- Stage 1: HistGen + REG2025
- Stage 2: [HISTAI case-report pairs](https://huggingface.co/datasets/histai/HISTAI-metadata)

The official HISTAI metadata repository links to its organ-specific WSI repositories. See the [HISTAI source documentation](https://github.com/HistAI/HISTAI) for dataset structure, access instructions, licensing, and citation. Users are responsible for following each dataset's access terms and redistribution rules.

## Released Weights

The official release is hosted at [AtlasAnalyticsLab/PathoSynVLM](https://huggingface.co/AtlasAnalyticsLab/PathoSynVLM). Download it into:

```text
$PATHOSYNVLM_WEIGHTS_ROOT/pathosynvlm-stage2-main/
```

```bash
source configs/paths.example.env
hf download AtlasAnalyticsLab/PathoSynVLM \
  --local-dir "$PATHOSYNVLM_WEIGHTS_ROOT/pathosynvlm-stage2-main"
```

See [docs/weights.md](docs/weights.md) for the required package layout. The Stage 2 paper run used `unfreeze_llm_base=true`, so the release package includes the merged/full LLM weights, not only a LoRA adapter.

## Reported Metrics

The main reported Stage 2 HISTAI result is:

| ROUGE-L | METEOR | BLEU-4 | BERTScore F1 | Diagnosis Exact | Diagnosis Relaxed | Certainty |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2495 | 0.1988 | 0.0525 | 0.3018 | 0.1667 | 0.3333 | 0.9000 |

See [configs/reported_results.json](configs/reported_results.json) for the machine-readable values.

## Limitations

- Inputs are precomputed patch embeddings, not raw WSIs.
- Result matching depends on using the same CONCHv1.5 feature format and metadata filtering.
- The model may generate incorrect or incomplete pathology statements.
- Dataset distributions and reporting styles may not generalize across institutions.

## Citation

Yang, Z.; Cheng, J.; Trinh, V. Q.-H. and Hosseini, M. S. (2026). **Simple Token-Efficient Vision-Language Model for Case-Level Pathology Synoptic Report Generation.** In *Proceedings of the 7th International Conference on Deep Learning Theory and Applications*, ISSN 2184-9277, pages 514–537.

```bibtex
@inproceedings{yang2026simpletokenvlm,
  title     = {Simple Token-Efficient Vision-Language Model for Case-Level Pathology Synoptic Report Generation},
  author    = {Yang, Zhiyuan and Cheng, Jiahao and Trinh, Vincent Quoc-Huy and Hosseini, Mahdi S.},
  booktitle = {Proceedings of the 7th International Conference on Deep Learning Theory and Applications},
  pages     = {514--537},
  year      = {2026},
  issn      = {2184-9277}
}
```
