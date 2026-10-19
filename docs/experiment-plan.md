# Experiment plan

The question: does an LLM with a decision head, reading one face crop, detect presentation attacks
and deepfakes on datasets it never trained on better than simpler detectors do?

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

The decision model only counts as an improvement if it beats these on the same split.

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
| 2 | Decision model vs CLIP probe, train FF++, test Celeb-DF v2 | decision model AUC is higher than the CLIP probe AUC on the same test crops |
| 3 | Decision model vs MiniFASNet, train CelebA-Spoof, test OULU-NPU | decision model BPCER at APCER ≤ 5% is lower than MiniFASNet's |
| 4 | Frozen vision tower as a limit | if experiments 2 and 3 fail, a run with vision layers unfrozen changes the result; otherwise the frozen encoder is ruled out as the cause |
| 5 | Calibration | ECE reported by `FastDecisionModel.calibrate` on the calibration split, and the same measure on the test split |
| 6 | Latency | p95 time of one `predict` call on the chosen GPU, measured over at least 200 crops |

If experiments 2 and 3 both fail after experiment 4, the decision model does not add value as a
pixel-level detector and the lab moves on to using it over detector scores and session metadata.

## Out of scope

Injection through the capture channel. See `docs/data.md`.
