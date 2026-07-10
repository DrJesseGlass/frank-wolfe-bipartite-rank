# Follow-up experiments

## Exp 1: proper (C, t) selection for fw_lr (inner 2-fold CV, refit at chosen iterate)

| dataset | lr_balanced_cv | fw_lr_cv_m0 | fw_lr_cv_m2 | Δ(m0−bal) | m0 fold wins |
|---|---|---|---|---|---|
| synthetic_1to50 | 0.7902±0.023 | 0.7960±0.025 | 0.7923±0.024 | +0.0059 | 10/15 |
| satimage_4 | 0.7668±0.009 | 0.7662±0.009 | 0.7668±0.009 | -0.0006 | 0/15 |
| abalone_19 | 0.9189±0.016 | 0.9195±0.016 | 0.9189±0.016 | +0.0006 | 3/15 |
| german_credit | 0.7804±0.026 | 0.7810±0.024 | 0.7806±0.026 | +0.0005 | 6/15 |

## Exp 2: FW-BPR around gradient-boosted trees

| dataset | gbt_plain | gbt_balanced | fw_gbt | Δ(fw−balanced) | fw fold wins |
|---|---|---|---|---|---|
| synthetic_1to50 | 0.7950±0.026 | 0.7992±0.026 | 0.7992±0.026 | +0.0000 | 0/15 |
| spambase | 0.9865±0.003 | 0.9865±0.003 | 0.9865±0.003 | -0.0000 | 0/15 |
| satimage_4 | 0.9596±0.004 | 0.9586±0.004 | 0.9586±0.004 | +0.0000 | 0/15 |
| abalone_19 | 0.8410±0.024 | 0.8236±0.036 | 0.8236±0.036 | +0.0000 | 0/15 |

## Findings

1. **Exp 1 — with honest cross-fitted (C, t) selection, iterating becomes a
   free option.** On the one dataset with known headroom (synthetic 1:36) it
   recovers the ablation gain (+0.59 pts AUC, 10/15 fold wins, median chosen
   t = 6). On datasets where balanced is already optimal it *selects t = 0*
   (satimage, abalone medians) and therefore never hurts (worst mean Δ =
   −0.06 pts). This fixes the earlier fw_lr ≈ lr_balanced result, but the
   harvestable gains remain small and dataset-dependent.
2. **Exp 2 — the plug-in claim fails for interpolating learners as-is.**
   HistGradientBoosting reaches train AUC = 1.0, so margin-0 misordering
   counts vanish and FW-BPR provably degenerates to the balanced fit
   (identical AUCs to 4 decimals; verified counts = 0). Moreover static
   balancing itself does not help GBT here (plain ≥ balanced on 3/4,
   +1.7 pts on abalone). Rescuing the tree/deep case requires out-of-fold
   count estimation or margin-based counts on a calibrated scale — noted as
   an open extension, not attempted.
