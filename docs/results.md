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
