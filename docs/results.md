# Results

Each run below was scored under `face_decision_lab.protocol`: subject holdout on
`paulohenriquevn/face-decision-data` (seed 3407, 20% of subjects for test), the same 4,000 test
crops, operating point at APCER ≤ 5%. Hardware: Colab L4. The split holds out people, not
generators or datasets, so these numbers say nothing about unseen manipulation methods.

## 2026-10-16: first comparison

| Metric | CLIP ViT-L/14 linear probe | Qwen3.5-4B decision model, 60 steps |
|---|---|---|
| Training rows | 79,087 (whole train side) | 1,920 (60 steps x 32) |
| AUC | 0.994 | 0.751 |
| EER | 3.5% | 31.6% |
| BPCER at APCER ≤ 5% | 2.6% | 85.0% |
| APCER, insight (face swap) | 6.7% | 7.6% |
| APCER, inpainting | 3.3% | 2.3% |
| APCER, text2img (synthetic) | 4.6% | 0.5% |
| APCER, masks and replays | 0% | 0% |
| attack_type accuracy | n/a | 61.6% |
| predict p95 | n/a | 291 ms |

Per-generator APCER is read at each model's own threshold. The decision model reaches APCER ≤ 5%
only by rejecting 85% of bona fide crops, so its low per-generator APCER reflects that threshold, not
better detection; compare models by AUC and BPCER.

Decision model calibration on 1,000 held-out training subjects: decision accuracy 69.4%, ECE 0.053.
The adapter and `report.json` are in the private model repository
`paulohenriquevn/face-decision-qwen35-4b`.

The comparison is not even: the probe trained on 41 times more rows. Two explanations remain open,
and the next runs separate them:

1. **Too little training.** Retrain the CLIP probe on the same 1,920 rows. If its AUC drops near
   0.75, data volume explains the gap and a longer decision model run is worth paying for.
2. **Frozen vision tower.** Unsloth trains LoRA on the language layers only. If the probe stays near
   0.99 on 1,920 rows, the frozen Qwen vision encoder is the more likely limit (experiment 4 in
   `docs/experiment-plan.md`).

## 2026-10-16: data volume, vision LoRA and unseen generators

Same protocol and test crops as above.

**Data volume does not explain the gap.** The CLIP probe trained on the same 1,920 rows as the
decision model (same seed, same `capped()` subset):

| Model | Training rows | AUC | EER | BPCER at APCER ≤ 5% |
|---|---|---|---|---|
| CLIP linear probe | 1,920 | 0.981 | 7.0% | 10.9% |
| Decision model, language LoRA | 1,920 | 0.751 | 31.6% | 85.0% |
| Decision model, language and vision LoRA | 1,920 | 0.850 | 22.7% | 75.1% |

Vision LoRA (`TRAIN_VISION = True`, 70.8 M trainable parameters against 64.5 M) raised AUC by
0.10, calibration decision accuracy from 69.4% to 72.0% and lowered ECE from 0.053 to 0.034, with
predict p95 at 326 ms. The frozen vision tower was a bottleneck, and the decision model with vision
LoRA still trails a linear probe on frozen CLIP features at equal data.

**Unseen generators are the harder test.** CLIP probe on the `unseen_*` protocols (trained without
one generator, tested on bona fide crops and that generator only):

| Protocol | Training rows | AUC | EER | BPCER at APCER ≤ 5% |
|---|---|---|---|---|
| subject_holdout | 79,087 | 0.994 | 3.5% | 2.6% |
| unseen_text2img | 58,686 | 0.956 | 11.7% | 24.0% |
| unseen_insight | 59,936 | 0.966 | 9.6% | 22.4% |
| unseen_inpainting | 59,768 | 0.995 | 3.8% | 2.6% |
| subject_holdout | 1,920 | 0.981 | 7.0% | 10.9% |
| unseen_text2img | 1,920 | 0.906 | 17.7% | 42.4% |
| unseen_insight | 1,920 | 0.946 | 12.8% | 33.4% |
| unseen_inpainting | 1,920 | 0.982 | 6.6% | 10.0% |

`unseen_text2img` and `unseen_insight` are where models should be compared from now on.

For outside context only: the DeepFakeFace paper (arXiv 2309.02218, Table 2) reports RECCE at AUC
35.1% on SD 1.5, 51.5% on inpainting and 63.1% on InsightFace. Its training data and split differ
from ours, so those numbers are not comparable to the tables above.
