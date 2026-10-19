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

## 2026-10-17: fine-tuning the official GenD checkpoint on the lab data

`clip_layernorm_detector.ipynb` at commit 7c5529e with `INIT_FROM = "gend_official"`: the
official GenD weights, one epoch on the lab data at the 1.3x face margin, GenD loss, batch 128,
peak learning rate 1e-4 with 10% warm-up, augmentation. The `cross_celebdf` model also scores
`subject_holdout`, whose training side is the same. Run tag
`gend_official-gend-e1-b128-lr0.0001` in `paulohenriquevn/face-decision-clip-ln`.

| Model | subject_holdout AUC / BPCER | unseen_text2img AUC / BPCER | Celeb-DF video AUC / frame AUC |
|---|---|---|---|
| LayerNorm-tuned CLIP v1 (lab data) | **0.998 / 1.2%** | **0.989 / 4.5%** | 0.658 / 0.620 |
| Official GenD (FF++) | 0.898 / 70.8% | 0.807 / 76.0% | **0.951 / 0.894** |
| Official GenD + one epoch on lab data | 0.971 / 14.2% | 0.859 / 68.5% | 0.929 / 0.878 |

BPCER is at APCER ≤ 5%. The fine-tuned model is the first that works on both domains, at a cost on
each: Celeb-DF video AUC drops by 0.022, and on DeepFakeFace it stays far behind the detector
trained on the lab data alone, most of all on the unseen diffusion generator.

The first scoring of `subject_holdout` in this run was wrong: with Celeb-DF crops loaded next to the
lab data, the standard split drew its subjects over both datasets, so its test side held Celeb-DF
crops and subjects from the training side (it reported AUC 0.956). Commit 9ca1445 keeps test-only
datasets out of the standard splits, and the table shows the model rescored on the canonical
4,000 crops. The `unseen_text2img` run was stopped after 22 steps and restarted on the fixed split.

## 2026-10-17: max fusion of the lab detector and official GenD

`notebooks/score_pretrained_detectors.ipynb` at commit c5595bd. The fused attack score is the larger
of the two probabilities: the v1 LayerNorm-tuned CLIP detector (the checkpoint trained for each lab
protocol, and the `subject_holdout` one for `cross_celebdf`) and the official GenD checkpoint at the
1.3x margin. Nothing is fitted. The same run reproduced every single-model number above from the
versioned notebook. Reports: `paulohenriquevn/face-decision-clip-ln`, `pretrained/`.

AUC and BPCER at APCER ≤ 5%; Celeb-DF also gives video AUC.

| Model | subject_holdout | unseen_text2img | unseen_insight | unseen_inpainting | Celeb-DF frame / video AUC |
|---|---|---|---|---|---|
| v1 (lab data) | **0.998 / 1.2%** | **0.989 / 4.5%** | **0.975 / 13.6%** | **0.998 / 0.4%** | 0.620 / 0.658 |
| Official GenD (FF++) | 0.898 / 70.8% | 0.807 / 76.0% | 0.946 / 36.5% | 0.910 / 44.9% | **0.894 / 0.951** |
| Official GenD + one epoch on lab data | 0.971 / 14.2% | 0.859 / 68.5% | not run | not run | 0.878 / 0.929 |
| Max fusion of v1 and GenD | 0.994 / 2.7% | 0.953 / 30.7% | 0.969 / 18.0% | 0.991 / 1.2% | 0.848 / 0.905 |

No model wins every column. The fusion has the best worst case: its lowest AUC across the five
tests is 0.905 (Celeb-DF video), against 0.859 for the fine-tuned GenD, 0.807 for GenD and 0.658
for v1. Its cost is BPCER: GenD gives some DeepFakeFace bona fide photos high scores, and the
maximum passes them through, so on unseen text2img BPCER at APCER ≤ 5% rises from 4.5% (v1) to
30.7%. A fusion that learns how much to trust each detector, instead of taking the maximum, is the
next thing to try; it needs a calibration set with both domains, which FaceForensics++ would supply.

## 2026-10-18: training with FaceForensics++

FaceForensics++ c23 (`notebooks/build_faceforensics.ipynb`): 78,660 crops from 4,950 of the 5,000
videos (the 1,000 originals and four methods; RetinaFace kept 98.2% to 98.5% of frames in every
class), with the official splits: 56,664 training, 11,003 validation and 10,993 test crops. Each
run below is `clip_layernorm_detector.ipynb` with the GenD loss, 1.3x face margin and augmentation;
the `ffpp` column is the FF++ test split scored by the same model.

| Run | Init, epochs, peak LR | FF++ test frame / video AUC | Celeb-DF frame / video AUC | subject_holdout AUC / BPCER |
|---|---|---|---|---|
| `ffpp_to_celebdf`, whole pair groups | CLIP, 2, 3e-4 | 0.700 / 0.803 | 0.628 / 0.753 | not scored |
| `ffpp_to_celebdf`, pair groups capped at 10 | CLIP, 2, 3e-4 | 0.827 / 0.866 | 0.765 / 0.850 | not scored |
| `lab_ffpp_to_celebdf`, capped | official GenD, 1, 1e-4 | 0.951 / 0.971 | 0.869 / 0.929 | 0.968 / 14.6% |
| Official GenD, no training here | | not scored | **0.894 / 0.951** | 0.898 / 70.8% |
| LayerNorm-tuned CLIP v1, lab data only | | not scored | 0.620 / 0.658 | **0.998 / 1.2%** |

The first run kept whole pair groups: an FF++ group (an original and its four manipulations, 16
frames each) is 80 crops, so each batch held one scene. Commit 61a9dce caps group pieces at 10,
which lifted the FF++ test AUC from 0.70 to 0.83, still far from the roughly 0.99 the literature
reports in-domain. Two epochs of LayerNorm tuning from CLIP do not reproduce GenD, which the paper
trains far longer.

Starting from the official checkpoint and training one epoch on the lab data with FF++ alongside
gives the same trade-off as training on the lab data alone (0.929 Celeb-DF video AUC, 0.971
subject_holdout AUC, earlier section): adding FF++ to the training side did not keep the Celeb-DF
score at GenD's level. Neither pass criterion of experiment 2b is met (Celeb-DF video AUC ≥ 0.951
and subject_holdout AUC ≥ 0.99 in one model).

## 2026-10-19: learned fusion, GenD variants and aligned crops

Two runs at commits a058a65 (`score_pretrained_detectors.ipynb`) and 71df2a1
(`score_gend_variants.ipynb`). Reports: `paulohenriquevn/face-decision-clip-ln`, `pretrained/`.

The learned fusion is a logistic regression over the logits of the v1 detector and the official
GenD CLIP checkpoint, fitted on the lab calibration side of each protocol plus the FF++
validation split, with every domain and class weighing the same; Celeb-DF stays out of the fit.

| Test | v1 | Official GenD CLIP | Max fusion | Learned fusion |
|---|---|---|---|---|
| subject_holdout AUC / BPCER | 0.998 / 1.2% | 0.898 / 70.8% | 0.994 / 2.7% | 0.994 / **0.9%** |
| unseen_text2img AUC / BPCER | **0.989 / 4.5%** | 0.807 / 76.0% | 0.953 / 30.7% | 0.970 / 13.0% |
| unseen_insight AUC / BPCER | 0.975 / 13.6% | 0.946 / 36.5% | 0.969 / 18.0% | **0.975 / 10.8%** |
| unseen_inpainting AUC / BPCER | **0.998 / 0.4%** | 0.910 / 44.9% | 0.991 / 1.2% | 0.995 / 0.9% |
| FF++ test frame / video AUC | not scored | **0.946 / 0.966** | not scored | 0.935 / 0.956 |
| Celeb-DF frame / video AUC | 0.620 / 0.658 | **0.894 / 0.951** | 0.848 / 0.905 | 0.831 / 0.884 |

The learned fusion beats the max fusion on every lab protocol and loses to it on Celeb-DF: fitted
on lab and FF++ crops, it learns to trust v1 more than v1 deserves on video it has never seen.

GenD checkpoints on Celeb-DF, frame / video AUC, EER:

| Checkpoint | Square crop, 16 frames (lab pipeline) | Aligned crop, 32 frames (GenD pipeline) |
|---|---|---|
| GenD CLIP ViT-L/14 | 0.894 / 0.951, 18.0% | 0.885 / 0.947, 19.3% |
| GenD Perception Encoder L | 0.899 / 0.953, 18.8% | 0.886 / 0.945, 19.6% |
| Mean of the two probabilities | **0.916 / 0.967, 16.5%** | 0.905 / 0.958, 17.9% |

Aligning the faces the way GenD's preprocessing does (`faces.aligned_crop`, 32 frames) lowered
every score a little, so the gap between 0.951 here and the paper's 0.960 does not come from the
crop. The ensemble of the two public checkpoints is the best Celeb-DF result measured in the lab:
0.967 video AUC on the lab's crops, above the single-model numbers GenD reports (0.960 CLIP, 0.958
PE). The comparison is not exact: the paper's frames and detector settings differ, and an ensemble
of two models is a different class of entry than a single model.

## 2026-10-19: gating between the lab detector and the GenD ensemble

`score_gated_fusion.ipynb` at commit c34e7e2. Two experts: v1 (lab data) and the mean of the GenD
CLIP and Perception Encoder checkpoints. A linear gate on GenD CLIP's features, fitted on the lab
calibration side and the FF++ validation split, predicts the probability that a crop is from the
lab domain; the score is `gate * v1 + (1 - gate) * ensemble`. Celeb-DF never enters a fit.

| Test (AUC / BPCER at APCER ≤ 5%) | v1 | GenD ensemble | Max of the two | Gated |
|---|---|---|---|---|
| subject_holdout | **0.998 / 1.2%** | 0.891 / 64.3% | 0.994 / 3.0% | **0.998 / 1.1%** |
| unseen_text2img | **0.989 / 4.5%** | 0.810 / 70.3% | 0.958 / 26.2% | **0.989 / 4.5%** |
| unseen_insight | 0.975 / 13.6% | 0.954 / 28.1% | 0.972 / 15.4% | **0.976 / 13.4%** |
| unseen_inpainting | **0.998 / 0.4%** | 0.902 / 40.5% | 0.992 / 1.4% | **0.998 / 0.4%** |
| FF++ test, frame / video AUC | not scored | **0.964 / 0.976** | 0.959 / 0.972 | 0.963 / **0.976** |
| Celeb-DF, frame / video AUC | 0.620 / 0.658 | **0.916 / 0.967** | 0.856 / 0.910 | 0.872 / 0.937 |

On every lab protocol and on FF++ the gated fusion matches the better expert. On Celeb-DF it
loses 0.030 video AUC to the ensemble alone: the gate gives v1 a mean weight of 0.40 on real
Celeb-DF frames and 0.22 on fakes (0.995 and 0.998 on subject_holdout), so studio-quality
interview video partly looks like the lab's photos to it, and v1 scores those real frames high.
Switching hard at 0.5 (0.935) or sharpening the gate (0.935) did not help. Its lowest score across
the seven tests is 0.937, the best worst case so far, but it does not meet the pass criterion
(Celeb-DF video AUC ≥ 0.951 with subject_holdout AUC ≥ 0.99).
