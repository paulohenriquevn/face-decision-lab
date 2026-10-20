# Experiment plan

The question: can a detector tell real faces from deepfakes and presentation attacks on datasets
it never trained on, at the level of the best published detectors on public benchmarks? The lab
started with an LLM with a decision head as the detector and closed it (see
[Decision model, closed](#decision-model-closed)); the detector is now the LayerNorm-tuned CLIP
ViT-L/14, alone or fused with the official GenD checkpoints.

Experiments use the splits in `face_decision_lab.splits` and the metrics in
`face_decision_lab.metrics`. Results are reported as measured on the test side, with the number of
test crops and APCER broken down by attack type and by generator.

The target protocol is cross-dataset: train on some datasets, test on others. The training data
(DeepFakeFace plus the AxonData sample, see `docs/data.md`) supports a subject holdout (new people,
same generators) and the unseen-generator protocols (one DeepFakeFace method left out of training).
`cross_celebdf` trains on the subject-holdout training side and tests on every crop of the official
Celeb-DF v2 test videos: another dataset, another face-swap pipeline, video frames instead of photos.
Its report adds `video_auc`, the AUC of the mean frame score per video. With FaceForensics++,
`ffpp` tests in-domain on its official test split, `ffpp_to_celebdf` is the cross-dataset protocol
of the deepfake literature (train on FaceForensics++, test on Celeb-DF), and `lab_ffpp_to_celebdf`
trains on both the lab data and FaceForensics++.

## Baselines

A detector only counts as an improvement if it beats these on the same split.

| Baseline | Attack family | Status |
|---|---|---|
| Linear probe on frozen CLIP image features | deepfake | built, see `docs/results.md` |
| DeepFace `anti_spoofing=True` (MiniFASNet) | print, replay | not built |

## Experiments

| # | Experiment | Passes when |
|---|---|---|
| 1 | Smoke run: notebook trains on generated noise images and scores the test split | the evaluation cell prints AUC, EER, BPCER at the APCER limit and APCER per attack type |
| 1b | Subject holdout on the current dataset: decision model vs CLIP probe | decision model AUC is higher than the CLIP probe AUC on the same held-out subjects |
| 2a | CLIP probe, GenD CLIP detector and decision model under `cross_celebdf` | frame AUC and video AUC of all three are reported on the same Celeb-DF crops |
| 2b | Detector trained on lab data and FF++ (`lab_ffpp_to_celebdf`) vs the official GenD checkpoint | Celeb-DF video AUC at or above GenD's 0.951 on the same crops, with subject_holdout AUC at or above 0.99 |
| 2c | Photo expert retrained with FF++ originals as bona fide (`lab_ffpp_real`), then gated with the GenD ensemble | (a) its mean score on real Celeb-DF frames is below its mean on fake ones; (b) the gated system reaches Celeb-DF video AUC ≥ 0.951 with subject_holdout AUC ≥ 0.99 |
| 2d | v2 retrained with super-resolved copies of its real crops labeled bona fide (`lab_ffpp_real_sr`) | (a) DiffusionFace BPCER at APCER ≤ 5% at or below 39% (v2: 78.7%); (b) subject_holdout AUC at or above 0.9937 (v2: 0.9987); (c) Celeb-DF video AUC at or above v2's 0.649 |
| 2e | v2's recipe trained with DF40's 31 known methods added (`lab_df40`) | (a) mean AUC over DF40's 9 unknown methods, each against its own reals, above both v2's and the GenD ensemble's on the same crops; (b) DiffusionFace AUC above 0.759, the best so far; (c) subject_holdout AUC at or above 0.9937; (d) Celeb-DF video AUC at or above v2's 0.649 |
| 2 | Decision model vs CLIP probe, train FF++, test Celeb-DF v2 | decision model AUC is higher than the CLIP probe AUC on the same test crops |
| 3 | Decision model vs MiniFASNet, train CelebA-Spoof, test OULU-NPU | decision model BPCER at APCER ≤ 5% is lower than MiniFASNet's |
| 4 | Frozen vision tower as a limit | if experiments 2 and 3 fail, a run with vision layers unfrozen changes the result; otherwise the frozen encoder is ruled out as the cause |
| 5 | Calibration | ECE reported by `FastDecisionModel.calibrate` on the calibration split, and the same measure on the test split |
| 6 | Latency | p95 time of one `predict` call on the chosen GPU, measured over at least 200 crops |

Experiments 2 to 6 were written for the decision model and are closed with it, unrun.

## Decision model, closed

On 2026-10-20 the lab stopped work on `Qwen3.5-4B` with Unsloth's `FastDecisionModel` as a
pixel-level detector. It met the gate of experiment 1b and then lost to the LayerNorm-tuned CLIP
detector on both protocols the two were scored on, on the same test crops (`docs/results.md`,
2026-10-16 and 2026-10-17):

| Protocol | Metric | Decision model, vision LoRA, 600 steps | LayerNorm-tuned CLIP |
|---|---|---|---|
| subject_holdout | AUC | 0.996 | 0.998 |
| subject_holdout | BPCER at APCER ≤ 5% | 2.3% | 1.2% |
| unseen_text2img | AUC | 0.972 | 0.989 |
| unseen_text2img | BPCER at APCER ≤ 5% | 13.1% | 4.5% |

Its 600-step run needed three resumes after Colab session losses, and its `predict` took about
320 ms per crop at p95 on an L4. The deepfake detectors the lab compares with (GenD, FSFM, FS-VFM)
are vision encoders, so a CLIP detector is also the one with published numbers to stand against.
The decision model was never scored on Celeb-DF or DiFF. Using it over detector scores and session
metadata, the fallback this plan named, was not tried.

## Out of scope

Injection through the capture channel. See `docs/data.md`.
