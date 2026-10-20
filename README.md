# face-decision-lab

A research lab for one question: can a detector tell a real face photo or video frame from a
deepfake (face swap, reenactment, diffusion-generated or diffusion-edited face) or a presentation
attack, on datasets it never saw?

The detector is CLIP ViT-L/14 with only its LayerNorm layers tuned and a linear head on the
normalized image features, the recipe of GenD (Yermakov et al., WACV 2026), trained on the lab's
own mix of data. For video it is fused, by the larger score, with the official GenD CLIP and
Perception Encoder checkpoints. Each crop gets one score, the probability that it is an attack.
`docs/results.md` holds every measured result, against published ones where a benchmark allows.

The lab started from `Qwen3.5-4B` fine-tuned with Unsloth's `FastDecisionModel`, which answers
`is_bona_fide` and `attack_type` for each crop. The LayerNorm-tuned CLIP detector beat it on both
protocols the two shared, by the widest margin on an unseen generator, and trains in a fraction of
the time, so the decision model is closed as a pixel-level detector. The reasons are in
[docs/experiment-plan.md](docs/experiment-plan.md#decision-model-closed).

Identity matching is not part of this lab. It stays with a face recognition model such as
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
| `notebooks/build_diff_test.ipynb` | Crops the official DiFF test split (diffusion faces) into a separate private dataset |
| `notebooks/build_diffusionface_test.ipynb` | Crops a fixed sample of DiffusionFace (diffusion faces with their real photos) into a separate private dataset |
| `notebooks/build_faceforensics.ipynb` | Downloads FaceForensics++ c23 and crops it into a separate private dataset |
| `notebooks/clip_baseline.ipynb` | Scores the CLIP linear probe under the same protocol |
| `notebooks/clip_layernorm_detector.ipynb` | Trains and scores a LayerNorm-tuned CLIP detector (GenD recipe) per protocol |
| `notebooks/score_pretrained_detectors.ipynb` | Scores the official GenD checkpoint and earlier lab detectors under every protocol, `cross_celebdf` included |
| `notebooks/score_gend_variants.ipynb` | Scores the GenD CLIP and Perception Encoder checkpoints and their ensemble on square and aligned Celeb-DF crops |
| `notebooks/score_gated_fusion.ipynb` | Gates between the lab detector and the GenD ensemble by predicted domain |
| `notebooks/score_diffusion_benchmarks.ipynb` | Scores the photo detectors, GenD and their fusions on DiFF, DiffusionFace or DF40's unknown methods, with the AUC of each subset or method |
| `notebooks/analyze_diffusion_benchmarks.ipynb` | Bootstrap intervals, the paired v2 vs GenD PE difference and a swap of the real photos between DiFF and DiffusionFace, from saved per-crop scores (no GPU) |
| `notebooks/diagnose_bona_fide_scores.ipynb` | Scores real crops from every source at native and lowered resolution, to tell whether low resolution or CelebA-HQ itself makes the detectors flag DiffusionFace reals |
| `notebooks/build_superresolved.ipynb` | Builds super-resolved copies of the training-side real crops (Real-ESRGAN, Swin2SR) for `lab_ffpp_real_sr` |
| `notebooks/build_df40.ipynb` | Crops DF40's known methods (FF++ domain) and unknown methods into a private dataset |
| `notebooks/train_decision_model.ipynb` | Trains, calibrates and evaluates the closed Qwen3.5-4B decision model on Colab |

## Running

Local checks:

```bash
uv sync
uv run pytest
uv run ruff check .
```

The notebooks run on Colab Pro (L4 or A100) and install this package from GitHub.

1. `notebooks/build_dataset.ipynb` crops the source datasets and uploads them as a private Hugging
   Face dataset. See [docs/data.md](docs/data.md) for sources, labels and the crop. The `build_*`
   notebooks do the same for each external test set.
2. `notebooks/clip_layernorm_detector.ipynb` trains the detector for each protocol in
   `PROTOCOL_NAMES` and uploads its checkpoint and reports.
3. `notebooks/score_diffusion_benchmarks.ipynb` scores the detectors and their fusions on DiFF,
   DiffusionFace or DF40 and saves every crop's score.

They read a Colab secret `HF_TOKEN` to reach the private datasets.

## Licenses

The code in this repository is Apache-2.0. What you train with it is not free of restrictions:

- The public face datasets listed in `docs/data.md` are research-only.
- OpenAI's CLIP repository is MIT, and so are the GenD checkpoints (`yermandy/GenD_CLIP_L_14`,
  `yermandy/GenD_PE_L`).
- Unsloth's decision model code (`unsloth/models/decision*.py`), used only by the closed decision
  model notebook, is AGPL-3.0. `Qwen3.5-4B` is Apache-2.0.

Face images are biometric data. Store them only where the dataset license allows, and never commit
them: `data/` is ignored by git.
