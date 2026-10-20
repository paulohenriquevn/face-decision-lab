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

## 2026-10-19: photo expert retrained with real video (experiment 2c)

The gated fusion stopped at 0.937 because v1 scored real interview video as an attack (mean 0.416
on YouTube-real, 0.490 on fakes). v2 is `clip_layernorm_detector.ipynb` at commit b962286 under
`lab_ffpp_real`: the subject_holdout training side plus the FF++ original videos as bona fide,
from CLIP as released, cross entropy, one epoch, batch 32, no augmentation, 1.5x crop. Fusions use
`score_gated_fusion.ipynb` at b962286 with the GenD CLIP + PE ensemble as video expert.

Mean attack score per Celeb-DF source (criterion (a): real below fake):

| Detector | Celeb-real | YouTube-real | Celeb-synthesis (fake) |
|---|---|---|---|
| v1 | 0.245 | 0.416 | 0.490 |
| v2 | **0.071** | **0.127** | 0.242 |

| System | subject_holdout AUC / BPCER at APCER ≤ 5% | FF++ test frame / video AUC | Celeb-DF frame / video AUC |
|---|---|---|---|
| v2 alone | 0.9987 / 1.9% | 0.673 / 0.713 | 0.582 / 0.649 |
| GenD ensemble alone | 0.891 / 64.3% | 0.964 / 0.976 | 0.916 / 0.967 |
| Gated, v1 + ensemble | 0.998 / 1.1% | 0.963 / 0.976 | 0.872 / 0.937 |
| Gated, v2 + ensemble | **0.9986 / 1.3%** | 0.961 / 0.976 | 0.880 / 0.948 |
| Max, v2 + ensemble | 0.9965 / 2.9% | **0.964 / 0.976** | **0.903 / 0.957** |

Criterion (a) passes. Criterion (b) (Celeb-DF video AUC ≥ 0.951 with subject_holdout AUC ≥ 0.99)
passes for the max fusion of v2 and the GenD ensemble (0.957 and 0.9965) and misses by 0.003 for
the gated one (0.948). Once v2 stops scoring real video as an attack, taking the larger score no
longer lets the photo expert's false alarms through on video. The cost is photo BPCER: 2.9% for
the max against 1.3% for the gated fusion. v2 trained on every generator, so it is scored on
subject_holdout only; the unseen-generator protocols need a v2 trained per protocol.

## 2026-10-19: DiFF test fakes, before the pristine images (superseded below)

`notebooks/score_diff.ipynb` scored the 56,358 face crops of DiFF's test fakes (57,060 images;
RetinaFace found a face in 96.0% to 100% of each method's images). The pristine images are
granted through the authors' form and had not arrived, so there is no AUC yet: the table holds the
share of fakes each detector scores above 0.5, per subset of the paper. None of the detectors
trained on DiFF. v1 and v2 trained on DeepFakeFace, whose fakes include Stable Diffusion faces;
the GenD checkpoints trained on FaceForensics++ only.

| Detector | T2I | I2I | FS | FE |
|---|---|---|---|---|
| GenD CLIP | 0.717 | 0.598 | 0.978 | 0.732 |
| GenD PE | 0.935 | 0.872 | 0.956 | 0.703 |
| GenD ensemble (mean of CLIP and PE) | 0.921 | 0.850 | 0.972 | 0.745 |
| v1 | 0.714 | 0.575 | 0.996 | 0.968 |
| v2 | 0.851 | 0.756 | **1.000** | 0.994 |
| Max, v1 + ensemble | 0.961 | 0.930 | 0.999 | 0.981 |
| Max, v2 + ensemble | **0.976** | **0.956** | **1.000** | **0.997** |

A share of flagged fakes is half of what an AUC measures: a detector that flags every image,
real ones included, would top this table. The pristine side decides whether these detectors also
leave real DiFF photos alone. What the table does show: the photo experts and GenD fail on
different subsets (v1 and v2 on whole-face generation, T2I and I2I; GenD on face editing, FE), so
the max fusion covers both, and v2 flags more DiFF fakes than v1 in every subset.

## 2026-10-19: DiFF with its pristine images

`score_diffusion_benchmarks.ipynb` at commit 51caa62 scored DiFF's test split with its pristine
side: 56,358 fake crops and 23,723 pristine crops (RetinaFace found a face in 23,723 of the 23,731
pristine test images). AUC per subset is every pristine image against that subset's fakes, the
protocol of the DiFF paper. None of the detectors trained on DiFF; v1 and v2 trained on
DeepFakeFace, which holds Stable Diffusion faces, and the GenD checkpoints on FaceForensics++ only.

| Detector | T2I | I2I | FS | FE | Mean of subsets | AUC, all | BPCER at APCER ≤ 5% |
|---|---|---|---|---|---|---|---|
| GenD CLIP | 0.838 | 0.794 | 0.978 | 0.839 | 0.862 | 0.843 | 52.4% |
| GenD PE | **0.969** | **0.955** | 0.987 | 0.911 | 0.955 | **0.952** | 38.5% |
| GenD ensemble | 0.957 | 0.936 | 0.989 | 0.902 | 0.946 | 0.940 | 43.8% |
| v1 | 0.821 | 0.751 | 0.984 | 0.957 | 0.878 | 0.860 | 75.5% |
| v2 | 0.937 | 0.920 | 0.991 | **0.978** | **0.956** | 0.950 | 18.0% |
| Max, v1 + ensemble | 0.902 | 0.872 | **0.994** | 0.958 | 0.931 | 0.921 | 28.0% |
| Max, v2 + ensemble | 0.934 | 0.918 | 0.993 | 0.977 | 0.955 | 0.949 | **17.5%** |

v2 and the official GenD PE reach the same mean AUC over the four subsets (0.956 and 0.955) from
opposite sides: GenD PE is best on whole-face generation (T2I, I2I), v2 on face editing (FE), and
v2 leaves far fewer real photos flagged at the APCER 5% point (18.0% against 38.5%). Taking the
larger score adds nothing here. GenD PE's number is the fair one to compare with published
FF++-trained detectors, since it never saw a diffusion face; v2's training included Stable
Diffusion faces. We have not verified a published DiFF number to compare with: the per-subset
table of FS-VFM (arXiv 2510.10663, Table V) did not come through when we fetched the paper.

These are much better than the same detectors on DiffusionFace (AUC 0.69 to 0.76). DiFF's
pristine photos are not CelebA-HQ, which the diagnosis further down found to be what the lab's detectors
flag.

## 2026-10-19: DiffusionFace, the first AUC on unseen diffusion faces

`notebooks/score_diffusion_benchmarks.ipynb` (`TEST_SET = "cross_diffusionface"`) scored 20,935
crops of DiffusionFace part 1: 1,490 CelebA-HQ photos and the images 13 diffusion methods made
from or next to them (one sampled image number in 20, the same numbers in every folder). None of
the detectors trained on DiffusionFace. AUC is every real photo against all fakes, and against the
fakes of each method.

| Detector | AUC | BPCER at APCER ≤ 5% |
|---|---|---|
| GenD CLIP | 0.715 | 89.0% |
| GenD PE | 0.690 | 90.0% |
| GenD ensemble | 0.725 | 87.3% |
| v1 | 0.744 | 79.2% |
| v2 | 0.731 | 78.7% |
| Max, v1 + ensemble | **0.759** | **77.0%** |
| Max, v2 + ensemble | 0.737 | 77.3% |

| Method | GenD ensemble | v1 | Max, v1 + ensemble |
|---|---|---|---|
| DiffSwap | 0.988 | 0.888 | 0.962 |
| ADM | 0.920 | 0.865 | 0.889 |
| LDM | 0.894 | 0.682 | 0.754 |
| PNDM | 0.739 | 0.868 | 0.869 |
| DDIM | 0.696 | 0.851 | 0.850 |
| Inpaint | 0.666 | 0.805 | 0.803 |
| DDPM | 0.555 | 0.774 | 0.761 |
| SD 1.5 image to image, strength 0.3 / 0.5 / 0.7 | 0.668 / 0.748 / 0.619 | 0.744 / 0.746 / 0.663 | 0.749 / 0.753 / 0.674 |
| SD 2.1 image to image, strength 0.3 / 0.5 / 0.7 | 0.582 / 0.655 / 0.674 | 0.561 / 0.590 / 0.618 | 0.567 / 0.606 / 0.622 |

No detector here separates diffusion faces it never trained on from real photos well enough for
use: at the operating point that catches 95% of fakes, the best system flags 77% of real photos.
The DiFF table above, where the same systems flagged 93% to 100% of fakes, measured only half of
this; on DiffusionFace the real photos score high too. Stable Diffusion image to image, which
redraws a real photo while keeping its layout, is the weakest case for every detector, including
v1, which trained on Stable Diffusion 1.5 faces made from text. The two experts again fail on
different methods (GenD on DDPM and Inpaint, v1 on LDM), but taking the larger score gains little.

## 2026-10-19: why DiffusionFace's real photos score as fake

`notebooks/diagnose_bona_fide_scores.ipynb` scored up to 1,000 real crops per source, none from
any detector's training side, at native resolution and lowered with `lower_resolution` to faces
128, 96 and 64 px wide. The threshold is each detector's APCER 5% point on the DiffusionFace
fakes; the table gives the share of each source's reals above it. AxonData is left out: its
test side has too few real crops to read.

| Real crops | Sharpness | v1 | v2 | GenD ensemble |
|---|---|---|---|---|
| DiffusionFace (CelebA-HQ), native | 441 | 77.8% | 76.3% | 87.0% |
| DiffusionFace, 128 px | 93 | 50.5% | 13.9% | 87.8% |
| DiffusionFace, 96 px | 42 | 53.8% | 3.6% | 92.2% |
| DeepFakeFace, native | 278 | 2.8% | 1.1% | 53.2% |
| DeepFakeFace, 64 px | 16 | 1.6% | 0.2% | 78.9% |
| FaceForensics++, native | 95 | 15.9% | 0.3% | 29.0% |
| FaceForensics++, 64 px | 11 | 11.1% | 0.0% | 63.3% |
| Celeb-DF, native | 31 | 49.1% | 2.7% | 48.6% |
| Celeb-DF, 64 px | 7 | 8.6% | 0.0% | 67.9% |

Sharpness is the median variance of the Laplacian; native DiffusionFace crops are the sharpest
of all sources.

Low resolution does not explain the failure of the lab detectors. Lowering the resolution of
DeepFakeFace, FaceForensics++ or Celeb-DF reals never raises v1 or v2 above the threshold. It
is the CelebA-HQ photos themselves: v2 gives them a median score of 0.974, and removing their
fine detail brings it to 0.107 at 128 px. The likely cause is how CelebA-HQ was made: Karras et
al. (ICLR 2018, appendix C) passed every CelebA photo through a JPEG artifact removal network
and a 4x super-resolution network, and the detail those networks drew is what v1 and v2 flag.
GenD behaves the other way: it scores any face higher as its resolution drops, on every source,
and it already flags 87% of DiffusionFace reals at native resolution.

Two readings follow. Against ordinary real photos (DeepFakeFace, FaceForensics++, Celeb-DF), v2
catches 95% of the DiffusionFace fakes while flagging 0.3% to 2.7% of the reals, so the fakes
themselves are mostly caught. But a real photo that went through a generative network, such as
super-resolution or a phone's enhancement, looks fake to v2, and DiffusionFace's AUC measures
exactly that. Our hypothesis, still to test, is that the fix lies in training data (real photos
processed by such networks, labeled real) and not in fusion.

## 2026-10-19: super-resolved reals as bona fide (experiment 2d) fail

`clip_layernorm_detector.ipynb` at commit 7d4136a trained v2's recipe under `lab_ffpp_real_sr`:
`lab_ffpp_real`'s training side plus 30,308 super-resolved copies of its real crops, labeled bona
fide (`build_superresolved.ipynb`; Real-ESRGAN x4plus and Swin2SR, half each). Same seed and
settings as v2; one run.

| Test side | v2 | v2 + super-resolved reals |
|---|---|---|
| subject_holdout AUC / BPCER at APCER ≤ 5% | 0.9987 / 1.9% | 0.9985 / 3.2% |
| Celeb-DF frame / video AUC | 0.582 / 0.649 | 0.504 / 0.528 |
| FF++ test frame / video AUC | 0.673 / 0.713 | 0.609 / 0.633 |
| DiffusionFace AUC / BPCER at APCER ≤ 5% | 0.731 / 78.7% | 0.605 / 90.9% |

Criterion (a) fails: DiffusionFace BPCER rose instead of falling to 39% or below, and so does
(c). The copies did lower the score of DiffusionFace's real photos (mean 0.827 for v2, 0.626
now), but they lowered the fakes' scores more: LDM fell to an AUC of 0.302 and ADM to 0.495.
Our reading is that the fine detail a super-resolution network draws is the same cue v2 used to
catch diffusion faces, so teaching it that this detail is real removed the cue on both sides.
Fixing DiffusionFace's reals through augmentation alone does not work; the next step is training
on many more generators (DF40).

## 2026-10-20: intervals and a paired comparison on DiFF

`score_diffusion_benchmarks.ipynb` at commit 104be1e rescored DiFF's test split and saved every
crop's score (`scores/cross_diff.npz` in the model repository); the analysis of
`analyze_diffusion_benchmarks.ipynb` ran on those scores. The set is the one of the 2026-10-19
section: DiFF's test split as released by the authors, received in October 2026, with the
pristine archive the authors sent through their access form.

| Subset | Methods (crops) | Crops |
|---|---|---|
| T2I | HPS 7,787, Midjourney 5,620, SDXL 5,450, FreeDoM-T 2,501 | 21,358 |
| I2I | SDXL Refiner 5,578, DreamBooth 3,758, FreeDoM-I 2,257, LoRA 1,581 | 13,174 |
| FS | DCFace 4,629, DiffFace 1,576 | 6,205 |
| FE | Cycle Diffusion 7,282, Imagic 4,319, CoDiff 4,020 | 15,621 |
| Pristine | | 23,723 |

The pristine count is the 23,723 of 23,731 pristine test images in which RetinaFace found a face.
An external review of these results cited 23,661 pristine test images for DiFF, from CrossDF; we
have not checked that paper, so the counts here are ours. The archive itself repeats some images:
459 of the 5,620 Midjourney crops are byte for byte the crop of another Midjourney file, the same
picture filed under several identity folders, and 6 pristine crops repeat another. They are
scored as the archive lists them.

Intervals are 95% percentile intervals from 1,000 bootstrap resamples of whole subjects
(`bootstrap_interval`, seed 3407). The fakes come from 143 identities and are resampled by
identity. DiFF gives no identity for the pristine images, so each counts as its own subject; if
some show the same person, the intervals are narrower than they should be.

BPCER at APCER ≤ 5% takes the threshold from the scored test crops themselves, at the APCER of
the worst attack type. That is the usual reporting point but an optimistic one: a threshold fixed
beforehand on other data would give a higher BPCER.

| Detector | AUC, all subsets pooled | BPCER at APCER ≤ 5% |
|---|---|---|
| GenD PE | 0.952 [0.949, 0.954] | 38.5% [35.3%, 41.4%] |
| GenD ensemble | 0.940 [0.937, 0.944] | 43.8% [40.7%, 47.3%] |
| v2 | 0.950 [0.948, 0.953] | 18.0% [17.2%, 18.6%] |
| Max, v2 + ensemble | 0.949 [0.946, 0.951] | 17.5% [16.8%, 18.1%] |

Paired difference, v2 minus GenD PE, from the same resamples:

| Metric | Difference | 95% interval |
|---|---|---|
| AUC | -0.001 | [-0.005, +0.003] |
| BPCER at APCER ≤ 5% | -20.5 points | [-23.5, -17.4] |

v2 and GenD PE cannot be told apart by AUC on DiFF; the interval of the difference holds zero.
At the APCER 5% point v2 flags about 20 points fewer real photos, and that difference holds
across the resamples. Neither number makes either detector state of the art on DiFF: we have no
published result on the same split, pristine set and protocol to compare with, and v2 trained on
Stable Diffusion faces while the published DiFF baselines did not.

## 2026-10-20: real photos swapped between DiFF and DiffusionFace

Each benchmark's fakes were scored against its own real photos and against the other's, with the
scores of the section above and `scores/cross_diffusionface.npz`; the fakes stay fixed. AUC with a
95% interval over whole subjects, 1,000 resamples. DiffusionFace's 1,490 reals are CelebA-HQ
photos, one subject each; a fake made from a real photo shares its subject, and the sample has
1,575 subjects in all.

| Fakes | Reals | GenD PE | GenD ensemble | v2 | Max, v2 + ensemble |
|---|---|---|---|---|---|
| DiFF | DiFF | 0.952 [0.949, 0.954] | 0.940 [0.937, 0.944] | 0.950 [0.948, 0.953] | 0.949 [0.946, 0.951] |
| DiFF | DiffusionFace | 0.746 [0.732, 0.760] | 0.780 [0.766, 0.793] | 0.630 [0.614, 0.646] | 0.644 [0.627, 0.659] |
| DiffusionFace | DiFF | 0.936 [0.933, 0.938] | 0.913 [0.910, 0.916] | 0.968 [0.966, 0.969] | 0.965 [0.963, 0.967] |
| DiffusionFace | DiffusionFace | 0.690 [0.681, 0.700] | 0.725 [0.715, 0.734] | 0.731 [0.721, 0.743] | 0.737 [0.726, 0.748] |

The real side decides the result. DiFF's fakes drop from 0.950 to 0.630 for v2 once they are set
against DiffusionFace's real photos, and DiffusionFace's fakes rise from 0.731 to 0.968 against
DiFF's. Every detector moves the same way, and no interval of a swapped pair overlaps the
unswapped one. This agrees with the diagnosis of 2026-10-19: what the detectors flag in
DiffusionFace is its real photos. The swap also changes resolution and framing along with the
source, so it attributes the drop to DiffusionFace's real photos as a whole; the earlier
resolution test is what points at the detail CelebA-HQ's processing added.

DiffusionFace's low AUC therefore does not show that these detectors miss its fakes: against
ordinary real photos they separate them about as well as DiFF's. It shows that they flag
CelebA-HQ photos as fake, which matters for any input that went through a similar enhancement.

## 2026-10-20: DiFF per subset, against published results

Papers that report DiFF give one AUC per subset, each subset's test fakes against the pristine
test images, and average the four. The table below computes the same from the scores of the
2026-10-20 intervals section (`scores/cross_diff.npz`), with the lab's `auc`; the interval is the
95% percentile interval of the four-subset mean over 1,000 resamples of whole subjects
(`bootstrap_interval`, seed 3407). AUC in percent.

| Detector | Training data | T2I | I2I | FS | FE | Mean |
|---|---|---|---|---|---|---|
| v1 | DeepFakeFace, AxonData | 82.09 | 75.07 | 98.43 | 95.67 | 87.82 [87.36, 88.26] |
| GenD PE | FF++ c23 | 96.90 | 95.48 | 98.69 | 91.08 | 95.54 [95.29, 95.80] |
| GenD ensemble | FF++ c23 | 95.72 | 93.57 | 98.90 | 90.20 | 94.60 [94.31, 94.90] |
| v2 | DeepFakeFace, AxonData, FF++ originals | 93.67 | 92.02 | 99.06 | 97.83 | 95.65 [95.44, 95.84] |
| Max, v2 + ensemble | as its two parts | 93.40 | 91.75 | 99.30 | 97.70 | 95.54 [95.35, 95.72] |

Published results with no training on DiFF, as reported (AUC in percent, frame level):

| Method | Training data | T2I | I2I | FS | FE | Mean | Source |
|---|---|---|---|---|---|---|---|
| FS-VFM ViT-L/16 | FF++ DeepFakes c23 | 92.72 | 92.51 | 97.17 | 92.83 | 93.81 | arXiv 2510.10663, Table V |
| FS-VFM ViT-B/16 | FF++ DeepFakes c23 | 91.00 | 91.80 | 83.16 | 89.84 | 88.95 | arXiv 2510.10663, Table V |
| FSFM ViT-B/16 | FF++ DeepFakes c23 | 88.20 | 89.00 | 81.99 | 88.50 | 86.92 | arXiv 2510.10663, Table V |
| FSFM ViT-B/16 | FF++ DeepFakes c23 | 61.74 | 71.91 | 71.31 | 78.98 | 70.99 | arXiv 2412.12032 (CVPR 2025), Table 3 |
| EfficientNet | FF++ | 74.12 | 57.27 | 82.11 | 57.20 | 67.68 | DiFF paper, arXiv 2401.15859, Table 7 |

The two FSFM rows are the same method in two papers by the same group; the source gives no reason
for the gap. The EfficientNet mean is ours, from the paper's four values.

The real side matches. The DiFF paper describes a pristine collection of 23,661 images split
8:1:1, but the pristine archive the authors send holds `DiFF_real/test` (23,731 images),
`DiFF_real/train/real` (14,286) and `DiFF_real/val/real` (550). FSFM's preprocessing
(`datasets/finetune/preprocess/dataset_preprocess.py`, `make_DiFF_set_extract_face`) sets every
image under `DiFF_real/test` against each subset's test fakes, as we do. Its published test log
(`finetuned_models/FF++_DF_c23_32frames/experiments_test/test_results.txt` in the
`Wolowolo/fsfm-3c` model repository) counts 43,582, 35,227, 28,133 and 37,484 test images for
T2I, I2I, FS and FE, against our 45,081, 36,897, 29,928 and 39,344: DLIB, which FSFM uses with a
box 30% larger than the face, leaves 1,499 to 1,860 fewer test images per subset than
RetinaFace. The external review's 23,661 is the
paper's size of the whole pristine collection, not of the test split.

Two differences remain. FSFM picks a checkpoint per subset by the lowest loss on that subset's
DiFF validation split (`checkpoint-min_val_loss_T2I.pth` and so on in the same log); our
detectors never saw DiFF, validation included. The larger one is training data: the published
rows train on FF++ DeepFakes c23 alone, the GenD models on FF++ c23, and v1 and v2 on
DeepFakeFace, whose fakes include Stable Diffusion inpainting and text-to-image faces. GenD PE
above the best published mean shows that a detector trained on FF++ alone gets there with this
preprocessing; v2 above it does not show that the lab's method is better, since v2 saw diffusion
faces. A same-protocol comparison needs the lab's recipe trained on FF++ DeepFakes c23 alone and
scored the same way.

## 2026-10-20: the lab's recipe trained on FF++ DeepFakes alone, scored on DiFF

`clip_layernorm_detector.ipynb` at commit 8edf794 trained v2's recipe (CLIP ViT-L/14 as released,
LayerNorm tuning, cross entropy, one epoch, batch 32, peak learning rate 3e-4, 1.5x framing, no
augmentation) under `ffpp_deepfakes_to_diff`: the FF++ c23 originals and DeepFakes of the official
training split, 22,644 crops, calibration on the validation split, and DiFF's test split as the
test side. Training took 19.3 minutes on an L4. This is the protocol of the published rows above,
with no checkpoint selection on DiFF.

| Test side | AUC per subset (T2I / I2I / FS / FE) | Mean of the four | Frame / video AUC |
|---|---|---|---|
| DiFF | 91.78 / 90.24 / 98.10 / 93.70 | 93.46 | 0.926 pooled |
| Celeb-DF v2 test | | | 0.752 / 0.895 |
| FF++ c23 test, four methods | | | 0.780 / 0.843 |

The notebook keeps reports but not per-crop scores for this run, so the mean has no interval.
Against the published rows on the same training data, it sits above FSFM ViT-B/16 (86.92 and
70.99) and FS-VFM ViT-B/16 (88.95) and 0.35 points below FS-VFM ViT-L/16 (93.81). Without an
interval and with a different face detector and framing, 93.46 and 93.81 cannot be told apart;
it is no state-of-the-art result. It also shows where v2's 95.65 comes from: about two points
of it are the training data, not the recipe.

## 2026-10-20: detection at a fixed BPCER

A bank's face check works at a fixed BPCER: at most 0.5% of genuine customers rejected, and
usually 0.1% to 0.2%. The number that matters there is detection, 1 - APCER over all attacks,
at that BPCER. From the saved scores of the intervals section, with the threshold set on the
test's own bona fide crops (optimistic; a threshold fixed on other data would detect less):

| Detector | DiFF at BPCER 0.5% | 0.2% | 0.1% | DiffusionFace at 0.5% |
|---|---|---|---|---|
| v1 | 7.8% | 0% | 0% | 0% |
| GenD PE | 22.1% | 15.1% | 11.4% | 8.1% |
| GenD ensemble | 20.7% | 13.0% | 8.7% | 11.1% |
| v2 | 21.1% | 12.0% | 7.2% | 2.6% |
| Max, v2 + ensemble | 21.7% | 12.4% | 8.7% | 3.5% |

At BPCER 0.5% the max fusion detects 11% of T2I, 9% of I2I, 66% of FS and 29% of FE fakes.
v1 gives many bona fide crops its top score, so no threshold brings its BPCER under 0.2%. The
highest-scored real photos set the threshold at these points, and every detector here leaves
most fakes below it, AUC of 0.95 notwithstanding. DiffusionFace's 1,490 reals allow 7 errors at
0.5% and 1.5 at 0.1%, so its column is little more than noise. Experiments from here on report
detection at BPCER 0.1%, 0.2% and 0.5%, with the threshold fixed on a calibration side.

## 2026-10-20: published checkpoints on the lab's DiFF crops

`score_diffusion_benchmarks.ipynb` at commit f4b165b scored the released checkpoints behind Table V
of arXiv 2510.10663 (`Wolowolo/fsfm-3c`, CC BY-NC 4.0, all fine-tuned on FF++ DeepFakes c23) on the
same 80,081 DiFF crops as every lab detector, reframed to 1.3x as FSFM crops them. timm's ViT with
average pooling and `fc_norm` loads their weights with strict keys and gives outputs identical to
FSFM's `models_vit` (maximum difference 0.0 on random input). The section before last compared the
lab's numbers with these models' published ones, on their own DLIB crops; this one removes that
difference. AUC in percent, 95% interval over 300 resamples of whole subjects.

| Detector | Training data | T2I | I2I | FS | FE | Mean [95% CI] | Published mean |
|---|---|---|---|---|---|---|---|
| FS-VFM ViT-L/16 | FF++ DeepFakes c23 | 93.98 | 94.04 | 97.45 | 94.69 | 95.04 [94.80, 95.28] | 93.81 |
| FS-VFM ViT-B/16 | FF++ DeepFakes c23 | 92.76 | 93.53 | 93.34 | 93.66 | 93.33 [92.89, 93.69] | 88.95 |
| FSFM ViT-B/16 | FF++ DeepFakes c23 | 88.49 | 90.02 | 91.96 | 91.43 | 90.48 [89.88, 90.99] | 86.92 |
| Lab recipe, `ffpp_deepfakes_to_diff` | FF++ DeepFakes c23 | 91.78 | 90.24 | 98.10 | 93.70 | 93.46 [93.14, 93.75] | |
| GenD PE | FF++ c23, four methods | 96.90 | 95.48 | 98.69 | 91.08 | 95.54 [95.30, 95.80] | |
| v2 | DeepFakeFace, AxonData, FF++ originals | 93.67 | 92.02 | 99.06 | 97.83 | 95.65 [95.44, 95.84] | |

All three published models score higher on the lab's crops than in their paper. On the same crops
and the same training data, the lab's recipe is 1.58 points below FS-VFM ViT-L (paired difference
[-1.96, -1.25]): the earlier reading of 93.46 against 93.81 as a tie compared two preprocessings,
and it was wrong. GenD PE is 0.50 above FS-VFM ViT-L ([+0.12, +0.85]) and v2 0.61 above
([+0.37, +0.84]), on more training data than FF++ DeepFakes.

Detection at a fixed BPCER, with the threshold set two ways: on the FF++ validation originals
(2,200 crops; the BPCER it gives on DiFF's reals in parentheses), and on DiFF's own reals
(optimistic, but the same for every detector):

| Detector | FF++ threshold: 0.1% | 0.2% | 0.5% (DiFF BPCER) | DiFF threshold: 0.1% | 0.2% | 0.5% |
|---|---|---|---|---|---|---|
| FS-VFM ViT-L/16 | 49.7% | 60.6% | 79.3% (9.32%) | 25.6% | 27.7% | 30.2% |
| FS-VFM ViT-B/16 | 53.6% | 68.9% | 84.5% (11.41%) | 7.9% | 12.0% | 15.7% |
| FSFM ViT-B/16 | 21.0% | 24.8% | 36.0% (3.34%) | 6.3% | 9.3% | 15.3% |
| Lab recipe, `ffpp_deepfakes_to_diff` | 2.4% | 7.9% | 12.8% (0.38%) | 4.0% | 8.2% | 15.0% |
| GenD PE | 3.4% | 6.8% | 13.1% (0.13%) | 11.4% | 15.1% | 22.1% |
| v2 | 75.6% | 82.7% | 96.5% (17.36%) | 7.2% | 12.0% | 21.1% |

A threshold from FF++ video frames holds on DiFF's photos for the lab recipe and GenD PE, and not
for FS-VFM or v2, which flag 9% to 17% of DiFF's reals at a threshold meant for 0.5%; their high
detection in that column is bought with that BPCER. With the threshold on DiFF's reals, FS-VFM
ViT-L detects the most fakes at 0.1%, 25.6% against GenD PE's 11.4%.

Score fusions, each detector's logit standardized on the FF++ validation originals and averaged.
Four pairs were tried on this test side, so the best of them is chosen on it:

| Fusion | T2I | I2I | FS | FE | Mean [95% CI] | DiFF threshold: 0.1% | 0.2% | 0.5% |
|---|---|---|---|---|---|---|---|---|
| FS-VFM ViT-L + GenD PE | 98.25 | 97.87 | 99.35 | 96.62 | 98.03 [97.91, 98.15] | 30.5% | 35.3% | 47.6% |
| FS-VFM ViT-L + GenD PE + v2 | 96.30 | 95.09 | 99.75 | 98.53 | 97.42 [97.28, 97.53] | 32.1% | 38.1% | 46.0% |
| FS-VFM ViT-L + v2 | 94.84 | 93.70 | 99.27 | 97.85 | 96.41 [96.21, 96.58] | 24.3% | 25.7% | 29.0% |
| FS-VFM ViT-L + lab recipe | 94.87 | 94.38 | 99.17 | 96.77 | 96.30 [96.11, 96.50] | 28.4% | 32.0% | 39.4% |

FS-VFM ViT-L and GenD PE, both trained on FF++ c23 alone, reach a four-subset mean of 98.03, about
three points above either one; their thresholds from FF++ give a DiFF BPCER of 2.5% to 4.2%, so the
fusion still needs calibration on photos. Neither model was scored on Celeb-DF here.

## 2026-10-20: the same detectors on video, and one fusion for photos and video

`score_diffusion_benchmarks.ipynb` at commits c73421d (`cross_celebdf`, 8,285 crops of the 518
official Celeb-DF v2 test videos) and 0959204 (`ffpp`, the 10,993 crops of the official FF++ c23
test split) scored the detectors of the section above, with thresholds from the same 2,200 FF++
validation originals. Fusions average each detector's logit standardized on those originals.
Video AUC is the AUC of the mean frame score per video, its interval over 500 resamples of whole
subjects; detection is at BPCER 0.1%, 0.2% and 0.5% on the FF++ threshold, with the BPCER that
threshold gives on the test's reals in parentheses.

| Detector | Celeb-DF frame / video AUC [95% CI] | Celeb-DF detection (BPCER) | FF++ test frame / video AUC | DiFF mean AUC |
|---|---|---|---|---|
| FS-VFM ViT-L/16 | 0.626 / 0.645 [0.605, 0.691] | 2.6%, 3.2%, 5.0% (0.39%, 0.60%, 1.41%) | 0.706 / 0.721 | 95.04 |
| Lab recipe, `ffpp_deepfakes_to_diff` | 0.752 / 0.895 [0.868, 0.917] | 9.3%, 21.7%, 30.5% (0.04%, 0.25%, 0.67%) | 0.780 / 0.843 | 93.46 |
| GenD CLIP | 0.894 / 0.951 [0.937, 0.966] | 3.0%, 5.3%, 14.4% (0.04%, 0.04%, 0.46%) | 0.946 / 0.966 | |
| GenD PE | 0.899 / 0.953 [0.937, 0.968] | 3.6%, 7.9%, 17.8% (0.04%, 0.14%, 0.39%) | 0.968 / 0.976 | 95.54 |
| GenD ensemble | 0.916 / 0.967 [0.954, 0.978] | 2.4%, 3.1%, 29.7% (0%, 0%, 0.91%) | 0.964 / 0.976 | 94.60 |
| Max, v2 + GenD ensemble | 0.903 / 0.957 [0.942, 0.971] | 2.8%, 3.9%, 32.0% (0%, 0.04%, 1.69%) | 0.964 / 0.976 | 95.54 |
| FS-VFM ViT-L + GenD PE | 0.877 / 0.927 [0.908, 0.950] | 20.6%, 22.6%, 31.2% (1.44%, 1.55%, 2.46%) | 0.947 / 0.959 | 98.03 |
| FS-VFM ViT-L + GenD CLIP + GenD PE | 0.912 / 0.957 [0.942, 0.973] | 12.6%, 16.3%, 29.9% (0.67%, 0.88%, 1.51%) | 0.960 / 0.970 | 97.99 [97.87, 98.10] |

FS-VFM ViT-L detects its own training method, FF++ DeepFakes, at AUC 0.988 on these crops, and
Face2Face (0.575), NeuralTextures (0.544) and Celeb-DF poorly: this checkpoint saw one
manipulation method, and the crops are not what fails it. On DiFF the three-way fusion detects
44.3%, 48.4% and 60.2% at the FF++ thresholds, with a DiFF BPCER of 0.39%, 0.50% and 1.10%.

The three-way fusion, every part trained on FF++ c23 alone, holds the lab's video bar (Celeb-DF
video AUC 0.957, the max fusion's) and adds 2.45 points of DiFF mean over the best single
detector. It was picked after seeing DiFF and Celeb-DF, out of six fusions, so it still needs a
test set that played no part in the choice, such as DiffusionFace or DF40's unknown methods.
