# face-decision-lab

A research lab for one question: can a small language model with a decision head look at a face
crop and tell a live face from a presentation attack or a deepfake, on datasets it never saw?

The model is `Qwen3.5-4B` fine-tuned with Unsloth's `FastDecisionModel`. It does not generate text.
For each crop it answers two questions with calibrated probabilities:

- `is_bona_fide`: is this a live, unaltered face?
- `attack_type`: bona fide, print, replay, mask, face swap, face reenactment or fully synthetic.

Identity matching is not part of this model. It stays with a face recognition model such as
DeepFace's Facenet512.

## Status

Research only. The plan and its pass criteria are in
[docs/experiment-plan.md](docs/experiment-plan.md); measured results are in
[docs/results.md](docs/results.md).

## Layout

| Path | What it does |
|---|---|
| `src/face_decision_lab/labels.py` | Attack taxonomy and the two decision questions |
| `src/face_decision_lab/manifest.py` | Reads and writes the CSV manifest of face crops |
| `src/face_decision_lab/splits.py` | Subject-level and cross-dataset splits with no leakage |
| `src/face_decision_lab/rows.py` | Turns samples into training rows and answers into attack scores |
| `src/face_decision_lab/metrics.py` | APCER, BPCER, ACER, AUC and EER |
| `src/face_decision_lab/faces.py` | RetinaFace detection and the standard 256x256 face crop |
| `src/face_decision_lab/sources/` | Adapters mapping each source dataset to labels and subjects |
| `src/face_decision_lab/build.py` | Resumable dataset build: crops, manifest and detection report |
| `src/face_decision_lab/hub.py` | Uploads the build as a private Hugging Face dataset and loads it back |
| `src/face_decision_lab/protocol.py` | The split, test subset and metrics every model is scored with |
| `src/face_decision_lab/baselines.py` | Linear probe baseline |
| `src/face_decision_lab/pairs.py` | Batches that keep a real photo with the fakes made from it |
| `notebooks/build_dataset.ipynb` | Builds the dataset on a Colab GPU and uploads it |
| `notebooks/build_celebdf_test.ipynb` | Crops the official Celeb-DF v2 test videos into a separate private dataset |
| `notebooks/build_faceforensics.ipynb` | Downloads FaceForensics++ c23 and crops it into a separate private dataset |
| `notebooks/clip_baseline.ipynb` | Scores the CLIP linear probe under the same protocol |
| `notebooks/clip_layernorm_detector.ipynb` | Trains and scores a LayerNorm-tuned CLIP detector (GenD recipe) per protocol |
| `notebooks/score_pretrained_detectors.ipynb` | Scores the official GenD checkpoint and earlier lab detectors under every protocol, `cross_celebdf` included |
| `notebooks/train_decision_model.ipynb` | Trains, calibrates and evaluates on Colab |

## Running

Local checks:

```bash
uv sync
uv run pytest
uv run ruff check .
```

Both notebooks run on Colab Pro (L4 or A100) and install this package from GitHub.

1. `notebooks/train_decision_model.ipynb` with `SMOKE = True` (the default) runs on generated noise
   images and needs no data. It checks the setup on the GPU.
2. `notebooks/build_dataset.ipynb` crops the source datasets and uploads them as a private Hugging
   Face dataset. See [docs/data.md](docs/data.md) for sources, labels and the crop.
3. `notebooks/train_decision_model.ipynb` with `SMOKE = False` trains on that dataset.

Both read a Colab secret `HF_TOKEN` to reach the private dataset.

## Licenses

The code in this repository is Apache-2.0. What you train with it is not free of restrictions:

- The public face datasets listed in `docs/data.md` are research-only.
- Unsloth's decision model code (`unsloth/models/decision*.py`) is AGPL-3.0.
- `Qwen3.5-4B` is Apache-2.0.

Face images are biometric data. Store them only where the dataset license allows, and never commit
them: `data/` is ignored by git.
