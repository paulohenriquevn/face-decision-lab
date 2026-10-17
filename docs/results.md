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

## 2026-10-17: GenD loss and longer training on unseen generators

`clip_layernorm_detector.ipynb` at commit 6578bbd: the GenD objective (cross entropy, 0.1
alignment, 0.5 uniformity) on the 1024-dimensional pooled features, three epochs, batch 128,
warm-up and cosine schedule, and augmentation (flip, affine, blur, color jitter, JPEG). Same
protocols and 4,000 test crops as the first LayerNorm-tuned run.

| Protocol | Model | AUC | EER | BPCER at APCER ≤ 5% |
|---|---|---|---|---|
| unseen_insight | Frozen CLIP probe | 0.966 | 9.6% | 22.4% |
| unseen_insight | LayerNorm-tuned CLIP, 1 epoch, batch 32, cross entropy | **0.975** | **6.9%** | **13.6%** |
| unseen_insight | LayerNorm-tuned CLIP, GenD loss, 3 epochs, batch 128, augmentation | 0.963 | 8.0% | 28.0% |
| unseen_text2img | Frozen CLIP probe | 0.956 | 11.7% | 24.0% |
| unseen_text2img | LayerNorm-tuned CLIP, 1 epoch, batch 32, cross entropy | **0.989** | **4.7%** | **4.5%** |
| unseen_text2img | LayerNorm-tuned CLIP, GenD loss, 3 epochs, batch 128, augmentation | 0.948 | 11.4% | 29.5% |

The GenD recipe did worse on both unseen generators. Four things changed at once (loss, epochs, batch size, augmentation),
so this run cannot say which one hurt. The notebook now takes `LOSS` and names each run after its
loss, epochs, batch size and augmentation, so the next runs can change one variable at a time.

## 2026-10-17: cross-dataset test on Celeb-DF v2

`cross_celebdf`: every crop of the 518 official Celeb-DF v2 test videos (8,285 crops, 16 frames per
video; RetinaFace kept 99.9% of real frames and 100% of fake ones). The lab detectors train on the
`subject_holdout` training side; the official GenD checkpoint (`yermandy/GenD_CLIP_L_14`, MIT)
trained on FaceForensics++ c23. Video AUC averages the frame scores of each video. The scores were
first measured in an interactive Colab session and saved to `paulohenriquevn/face-decision-clip-ln`
under `cross_celebdf/`; `notebooks/score_pretrained_detectors.ipynb` reproduces them and writes to
`pretrained/`.

| Model | Trained on | Frame AUC | Video AUC | EER | BPCER at APCER ≤ 5% |
|---|---|---|---|---|---|
| Frozen CLIP probe | lab data | 0.617 | 0.633 | 42.4% | 89.9% |
| LayerNorm-tuned CLIP v1 | lab data | 0.620 | 0.658 | 42.2% | 95.4% |
| GenD CLIP ViT-L/14, crop at 1.5x | FF++ c23 | 0.878 | 0.943 | 19.5% | 63.2% |
| GenD CLIP ViT-L/14, crop at 1.3x | FF++ c23 | **0.894** | **0.951** | **18.0%** | **50.5%** |

The GenD paper reports 0.960 video AUC on Celeb-DF v2. Its checkpoint reaches 0.951 on the lab's
crops, so the crop pipeline and the test set are sound, and the lab detectors' near-chance score
comes from their training data: press photos and generated images do not teach what a face swap
in a compressed video looks like. Videos from YouTube-real score almost as high as the fakes under
the v1 detector (mean 0.416 against 0.490; Celeb-real 0.245).

The same GenD checkpoint on the lab protocols (4,000 test crops each, 1.3x crop):

| Protocol | GenD (FF++) AUC | GenD BPCER at APCER ≤ 5% | LayerNorm-tuned CLIP v1 AUC | v1 BPCER at APCER ≤ 5% |
|---|---|---|---|---|
| subject_holdout | 0.898 | 70.8% | **0.998** | **1.2%** |
| unseen_text2img | 0.807 | 76.0% | **0.989** | **4.5%** |
| unseen_insight | 0.946 | 36.5% | **0.975** | **13.6%** |
| unseen_inpainting | 0.910 | 44.9% | **0.998** | **0.4%** |

Each detector fails on the other's domain: GenD misses diffusion images, the lab detectors miss
video face swaps. A detector that holds up on both is the gap worth working on.
