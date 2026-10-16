# Experiment plan

The question: does an LLM with a decision head, reading one face crop, detect presentation attacks
and deepfakes on datasets it never trained on better than simpler detectors do?

Every experiment uses the cross-dataset protocol in `face_decision_lab.splits` and the metrics in
`face_decision_lab.metrics`. Results are reported as measured on the test datasets, with the number
of test crops.

## Baselines

The decision model only counts as an improvement if it beats these on the same split.

| Baseline | Attack family | Status |
|---|---|---|
| Linear probe on frozen CLIP image features | deepfake | not built |
| DeepFace `anti_spoofing=True` (MiniFASNet) | print, replay | not built |

## Experiments

| # | Experiment | Passes when |
|---|---|---|
| 1 | Smoke run: notebook trains 60 steps and scores the test split | the evaluation cell prints AUC, EER, BPCER at the APCER limit and APCER per attack type |
| 2 | Decision model vs CLIP probe, train FF++, test Celeb-DF v2 | decision model AUC is higher than the CLIP probe AUC on the same test crops |
| 3 | Decision model vs MiniFASNet, train CelebA-Spoof, test OULU-NPU | decision model BPCER at APCER ≤ 5% is lower than MiniFASNet's |
| 4 | Frozen vision tower as a limit | if experiments 2 and 3 fail, a run with vision layers unfrozen changes the result; otherwise the frozen encoder is ruled out as the cause |
| 5 | Calibration | ECE reported by `FastDecisionModel.calibrate` on the calibration split, and the same measure on the test split |
| 6 | Latency | p95 time of one `predict` call on the chosen GPU, measured over at least 200 crops |

If experiments 2 and 3 both fail after experiment 4, the decision model does not add value as a
pixel-level detector and the lab moves on to using it over detector scores and session metadata.

## Out of scope

Injection through the capture channel. See `docs/data.md`.
