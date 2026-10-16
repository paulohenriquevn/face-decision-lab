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

Research only. Nothing here has been trained or measured yet. The plan and its pass criteria are in
[docs/experiment-plan.md](docs/experiment-plan.md).

## Layout

| Path | What it does |
|---|---|
| `src/face_decision_lab/labels.py` | Attack taxonomy and the two decision questions |
| `src/face_decision_lab/manifest.py` | Reads and writes the CSV manifest of face crops |
| `src/face_decision_lab/splits.py` | Subject-level and cross-dataset splits with no leakage |
| `src/face_decision_lab/rows.py` | Turns samples into training rows and answers into attack scores |
| `src/face_decision_lab/metrics.py` | APCER, BPCER, ACER, AUC and EER |
| `src/face_decision_lab/extract.py` | Samples video frames and saves face crops with DeepFace |
| `notebooks/train_decision_model.ipynb` | Trains, calibrates and evaluates on Colab |

## Running

Local checks:

```bash
uv sync
uv run pytest
uv run ruff check .
```

Training runs on Colab Pro (L4 or A100). Open `notebooks/train_decision_model.ipynb` in Colab; it
installs this package from GitHub. With `SMOKE = True` (the default) it runs on generated noise
images and needs no data, which checks the setup on the GPU. For a real run, prepare the data as
described in [docs/data.md](docs/data.md), put it in Google Drive and set `SMOKE = False`.

## Licenses

The code in this repository is Apache-2.0. What you train with it is not free of restrictions:

- The public face datasets listed in `docs/data.md` are research-only.
- Unsloth's decision model code (`unsloth/models/decision*.py`) is AGPL-3.0.
- `Qwen3.5-4B` is Apache-2.0.

Face images are biometric data. Store them only where the dataset license allows, and never commit
them: `data/` is ignored by git.
