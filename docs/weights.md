# Weights

PathoSynVLM has code in Git and model artifacts outside Git.

The intended user experience is:

1. The authors export the trained Stage 2 model once.
2. The authors upload that exported directory to Hugging Face, GitHub Releases, institutional storage, or another artifact host.
3. Users download the exported directory into `weights/pathosynvlm-stage2-main/`.
4. Users run inference or evaluation directly from those weights.

Users should **not** need to train the model just to generate a report, as long as the author-uploaded weights are available.

## What Users Download

The weight package should have this layout:

```text
weights/pathosynvlm-stage2-main/
  config.json
  vlm_state.pt
  tokenizer/
  llm/
  best_checkpoint_summary.json
```

`llm/` should be a merged Hugging Face model directory whenever the LLM base was trainable. This matters because the paper run used `unfreeze_llm_base=true`; a LoRA adapter alone is not sufficient for exact reruns unless the base updates are also included.

After the final artifact URL is known, the README command should be updated, for example:

```bash
huggingface-cli download <ORG_OR_USER>/pathosynvlm-stage2-main \
  --local-dir weights/pathosynvlm-stage2-main
```

If the release is hosted somewhere other than Hugging Face, download and unpack it so that `weights/pathosynvlm-stage2-main/config.json` exists.

## What `export_release_weights.py` Does

`scripts/export_release_weights.py` does **not** download weights from the internet and does **not** recreate the paper weights by itself. It converts a completed local training run into a clean inference package.

Use it when:

- You are the author preparing the official model release.
- You ran the training pipeline and want to package your own checkpoint.

Do not use it when:

- You only want to run the pretrained model. In that case, download the uploaded package instead.

Export command for the authors:

```bash
python scripts/export_release_weights.py \
  --run_dir runs/stage2_main \
  --output_dir weights/pathosynvlm-stage2-main \
  --overwrite
```

This script:

1. Reads `train_args.json` and the selected trainer checkpoint.
2. Loads the base Qwen2.5-3B-Instruct model.
3. Applies LoRA if the run used LoRA.
4. Loads full LLM state if present in `trainer_state_step_<N>.pt`.
5. Merges LoRA into the LLM unless `--no_merge_lora` is set.
6. Saves `llm/`, `tokenizer/`, `vlm_state.pt`, and `config.json`.

The raw trainer state can be large because it may include optimizer and full LLM state. Run export on a compute node with enough CPU RAM/GPU memory. The resulting `config.json` is sanitized to avoid embedding local absolute training paths.

## Inference

```bash
python scripts/generate_case_report.py \
  --weights weights/pathosynvlm-stage2-main \
  --embeddings data/embeddings/HISTAI-skin-b2/conch_v15/5x_512/patches/example_1.h5 \
               data/embeddings/HISTAI-skin-b2/conch_v15/5x_512/patches/example_2.h5 \
  --output_json report.json
```

The output is expected to follow:

```text
Diagnosis: ...
Certainty: ...
Conclusion: ...
```

## Usage Modes

| Goal | Need released weights? | Need to train? |
|---|---:|---:|
| Generate a report for a case | yes | no |
| Evaluate the official model on prepared HISTAI embeddings | yes | no |
| Run the full training pipeline | no | yes |
| Create your own redistributable model package | no | yes, then export |
