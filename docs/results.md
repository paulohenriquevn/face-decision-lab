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

## 2026-10-16: LayerNorm-tuned CLIP detector

`notebooks/clip_layernorm_detector.ipynb`: CLIP ViT-L/14 with only its LayerNorm parameters
trained, L2-normalized features, a linear head and batches that keep each DeepFakeFace photo with
the fakes made from it (the recipe of GenD, arXiv 2508.06248, without its metric learning loss). One
epoch per protocol, batch 32, learning rate 3e-4, on a Colab L4. Compared with the frozen CLIP
linear probe on the same protocol and test crops:

| Protocol | Model | AUC | EER | BPCER at APCER ≤ 5% |
|---|---|---|---|---|
| subject_holdout | frozen CLIP probe | 0.994 | 3.5% | 2.6% |
| subject_holdout | LayerNorm-tuned CLIP | **0.998** | **1.6%** | **1.2%** |
| unseen_text2img | frozen CLIP probe | 0.956 | 11.7% | 24.0% |
| unseen_text2img | LayerNorm-tuned CLIP | **0.989** | **4.7%** | **4.5%** |
| unseen_insight | frozen CLIP probe | 0.966 | 9.6% | 22.4% |
| unseen_insight | LayerNorm-tuned CLIP | **0.975** | **6.9%** | **13.6%** |
| unseen_inpainting | frozen CLIP probe | 0.995 | 3.8% | 2.6% |
| unseen_inpainting | LayerNorm-tuned CLIP | **0.998** | **2.0%** | **0.4%** |

The tuned model beats the frozen probe on every protocol, most on unseen text2img, where the
BPCER at APCER ≤ 5% falls from 24.0% to 4.5%. Training took 15.9 to 21.5 minutes per protocol. The tuned parameters and each report are in the
private model repository `paulohenriquevn/face-decision-clip-ln`.

## 2026-10-16: decision model at 600 steps with vision LoRA

Same subject_holdout protocol and 4,000 test crops. 600 steps of 32 rows (19,200 training rows),
language and vision LoRA, calibrated on 1,000 held-out training subjects. Adapter and report:
`paulohenriquevn/face-decision-qwen35-4b`, folder `subject_holdout-vision-lora-600-steps`.

| Model | Training rows | AUC | EER | BPCER at APCER ≤ 5% |
|---|---|---|---|---|
| Decision model, vision LoRA, 60 steps | 1,920 | 0.850 | 22.7% | 75.1% |
| Decision model, vision LoRA, 600 steps | 19,200 | **0.996** | **3.0%** | **2.3%** |
| Frozen CLIP linear probe | 79,087 | 0.994 | 3.5% | 2.6% |
| LayerNorm-tuned CLIP | 79,087 | 0.998 | 1.6% | 1.2% |

Ten times more training rows took the decision model from AUC 0.850 to 0.996, past the frozen
CLIP probe trained on four times more rows, and its attack_type accuracy reached 94.1%. It stays
below the LayerNorm-tuned CLIP detector, with predict p95 at 326 ms on an L4. This passes the
gate set in the plan (AUC above 0.981, the CLIP probe at 1,920 rows), so the decision model remains
a candidate detector. The protocols that matter for generalization (`unseen_text2img`,
`unseen_insight`) have not been run for it yet.

## 2026-10-17: decision model on an unseen generator

`unseen_text2img`: trained without any Stable Diffusion 1.5 image, tested on bona fide crops and
SD 1.5 crops only (4,000). Decision model with language and vision LoRA, 600 steps (19,200 rows).
The run survived three Colab session losses by resuming from Hub checkpoints (every 100 steps).

| Model | Training rows | AUC | EER | BPCER at APCER ≤ 5% |
|---|---|---|---|---|
| Frozen CLIP linear probe | 58,686 | 0.956 | 11.7% | 24.0% |
| Decision model, vision LoRA, 600 steps | 19,200 | 0.972 | 8.6% | 13.1% |
| LayerNorm-tuned CLIP | 58,686 | **0.989** | **4.7%** | **4.5%** |

The decision model generalizes better than the frozen CLIP probe and worse than the
LayerNorm-tuned CLIP detector, at 317 ms predict p95. Its attack_type accuracy here is 46.4%: it
never saw the `fully_synthetic` label during training, so it names the unseen generator wrongly
even when it flags it as an attack.
