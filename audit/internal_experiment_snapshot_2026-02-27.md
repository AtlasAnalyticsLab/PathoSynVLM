# HistAI Finetune Experiment Snapshot (2026-02-27)

Generated at: `2026-02-27T08:50:44`

## Running Experiments
- `histai_ft_5x512_from_5x512_all_baseline_double_max512_dropout02_ema_2`
- `histai_ft_5x512_from_5x512_all_baseline_double_min96_max512_dropout02_ema`
- `histai_ft_5x512_from_5x512_all_baseline_use_markers_no_index`

## Best Metrics by Run (sorted by best val loss)

| Run | Status | Best Val Loss | Best Step | Best Epoch | BERT F1 | Diag Relaxed | Certainty Match | METEOR | BLEU4 | ROUGE-L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `histai_ft_5x512_from_5x512_all_baseline_prompt_double` | completed_or_stopped | 1.0109 | 30400 | 7 | 0.3018 | 0.3333 | 0.9000 | 0.1988 | 5.2512 | 0.2495 |
| `histai_ft_5x512_from_5x512_all_baseline_double_min96_max512_dropout02_ema` | in_progress | 1.0161 | 26200 | 7 | 0.2093 | 0.3333 | 0.8667 | 0.2750 | 9.6432 | 0.2511 |
| `histai_ft_5x512_from_5x512_all_baseline_dropout_02` | completed_or_stopped | 1.0281 | 39200 | 9 | 0.3036 | 0.2667 | 0.9000 | 0.1910 | 3.7313 | 0.2275 |
| `histai_ft_5x512_from_5x512_all_baseline_fp32` | completed_or_stopped | 1.0281 | 29000 | 7 | 0.2788 | 0.1667 | 0.8667 | 0.1971 | 3.4638 | 0.2468 |
| `histai_ft_5x512_from_5x512_all_baseline_use_markers_no_index` | in_progress | 1.0293 | 34600 | 8 | 0.3081 | 0.3000 | 0.9333 | 0.2232 | 6.7554 | 0.2625 |
| `histai_ft_5x512_baseline_scratch` | completed_or_stopped | 1.0301 | 29000 | 7 | 0.2862 | 0.2333 | 0.8667 | 0.1846 | 3.0922 | 0.2101 |
| `histai_ft_5x512_from_5x512_all_baseline_512_gen_token` | completed_or_stopped | 1.0321 | 29200 | 7 | 0.2902 | 0.3000 | 0.8667 | 0.1913 | 1.8953 | 0.2302 |
| `histai_ft_5x512_from_5x512_all_baseline_96_min_token` | completed_or_stopped | 1.0369 | 29000 | 7 | 0.2232 | 0.2000 | 0.8667 | 0.2628 | 8.6137 | 0.2454 |
| `histai_ft_5x512_from_5x512_all_baseline` | completed | 1.0392 | 30534 | 7 | 0.2711 | 0.3000 | 0.9000 | 0.1772 | 3.8289 | 0.2168 |
| `histai_ft_5x512_from_5x512_all_baseline_wsi_markers` | completed_or_stopped | 1.0406 | 30534 | 7 | 0.2934 | 0.3000 | 0.9000 | 0.1912 | 3.1213 | 0.2292 |
| `histai_ft_5x512_from_5x512_all_baseline_freeze_aligner` | completed_or_stopped | 1.0439 | 34600 | 8 | 0.3138 | 0.1667 | 0.9000 | 0.1942 | 2.7475 | 0.2498 |
| `histai_ft_5x512_from_5x512_all_baseline_double_max512_dropout02_ema_2` | in_progress | 1.0722 | 47400 | 11 | 0.2830 | 0.2667 | 0.8000 | 0.1768 | 3.5313 | 0.2151 |
| `histai_ft_5x512_from_5x512_all_baseline_fp16` | completed_or_stopped | 2.2238 | 200 | 1 | n/a | 0.0667 | 0.0000 | 0.0339 | 0.0136 | 0.0539 |
