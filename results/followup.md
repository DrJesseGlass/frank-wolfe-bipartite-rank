# Follow-up experiments

## Exp 1: proper (C, t) selection for fw_lr (inner 2-fold CV, refit at chosen iterate)

| dataset | lr_balanced_cv | fw_lr_cv_m0 | fw_lr_cv_m2 | Δ(m0−bal) | m0 fold wins |
|---|---|---|---|---|---|
| synthetic_1to50 | 0.7902±0.023 | 0.7960±0.025 | 0.7923±0.024 | +0.0059 | 10/15 |
| satimage_4 | 0.7668±0.009 | 0.7662±0.009 | 0.7668±0.009 | -0.0006 | 0/15 |
| abalone_19 | 0.9189±0.016 | 0.9195±0.016 | 0.9189±0.016 | +0.0006 | 3/15 |
| german_credit | 0.7804±0.026 | 0.7807±0.023 | 0.7805±0.026 | +0.0002 | 6/15 |

## Exp 2: FW-BPR around gradient-boosted trees

| dataset | gbt_plain | gbt_balanced | fw_gbt | Δ(fw−balanced) | fw fold wins |
|---|---|---|---|---|---|
| synthetic_1to50 | 0.7950±0.026 | 0.7992±0.026 | 0.7992±0.026 | +0.0000 | 0/15 |
| spambase | 0.9865±0.003 | 0.9865±0.003 | 0.9865±0.003 | -0.0000 | 0/15 |
| satimage_4 | 0.9596±0.004 | 0.9586±0.004 | 0.9586±0.004 | +0.0000 | 0/15 |
| abalone_19 | 0.8410±0.024 | 0.8236±0.036 | 0.8236±0.036 | +0.0000 | 0/15 |
