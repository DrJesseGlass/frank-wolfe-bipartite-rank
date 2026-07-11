# FW-BPR benchmark results

Test AUC, mean ± std over 3-fold × 5-repeat stratified CV. C tuned on a 25% validation split of each training fold; FW-BPR iterate also selected on validation.

| dataset | lr_plain | lr_balanced | fw_lr | svm_plain | svm_balanced | fw_svm | ranksvm | gbt |
|---|---|---|---|---|---|---|---|---|
| breast_cancer | 0.9934±0.005 | 0.9939±0.005 | 0.9939±0.005 | **0.9941±0.006** | 0.9931±0.006 | 0.9935±0.006 | 0.9906±0.005 | 0.9898±0.007 |
| synthetic_1to50 | 0.7929±0.028 | 0.7931±0.024 | 0.7929±0.026 | 0.7404±0.053 | 0.7940±0.023 | 0.7908±0.028 | 0.7911±0.024 | **0.7992±0.026** |
| pima_diabetes | 0.8278±0.017 | 0.8279±0.016 | 0.8270±0.015 | 0.8280±0.015 | 0.8275±0.015 | **0.8297±0.014** | 0.8282±0.014 | 0.7922±0.015 |
| german_credit | 0.7832±0.024 | **0.7836±0.024** | 0.7795±0.023 | 0.7776±0.023 | 0.7784±0.024 | 0.7766±0.026 | 0.7761±0.024 | 0.7637±0.029 |
| spambase | 0.9699±0.003 | 0.9702±0.003 | 0.9702±0.003 | 0.9703±0.003 | 0.9699±0.004 | 0.9695±0.003 | 0.9702±0.004 | **0.9865±0.003** |
| satimage_4 | 0.7676±0.010 | 0.7633±0.011 | 0.7635±0.011 | 0.6946±0.029 | 0.7615±0.011 | 0.7675±0.010 | 0.7688±0.009 | **0.9586±0.004** |
| abalone_19 | 0.9083±0.022 | 0.9211±0.016 | 0.9193±0.016 | 0.8385±0.070 | 0.9187±0.016 | 0.9190±0.019 | **0.9226±0.016** | 0.8236±0.036 |

## Paired comparison: fw_lr vs lr_balanced (per fold)

- breast_cancer: mean Δ = -0.0000, fw_lr wins 2/15 folds
- synthetic_1to50: mean Δ = -0.0003, fw_lr wins 6/15 folds
- pima_diabetes: mean Δ = -0.0008, fw_lr wins 3/15 folds
- german_credit: mean Δ = -0.0041, fw_lr wins 2/15 folds
- spambase: mean Δ = +0.0000, fw_lr wins 3/15 folds
- satimage_4: mean Δ = +0.0002, fw_lr wins 2/15 folds
- abalone_19: mean Δ = -0.0019, fw_lr wins 0/15 folds

## Findings (2026-07-10 run)

1. **fw_svm matches RankSVM everywhere** (mean |Δ| ≤ 0.4 AUC pts, per-fold
   wins split ~50/50 on all 7 datasets) — the reweighting meta-algorithm
   recovers pairwise-SVM ranking quality without ever forming pairs. This is
   the cleanest empirical confirmation of the FW-vertex = weighted-
   classification theory (cor:fw_bridge).
2. **fw_svm repairs the plain SVM under imbalance** (+5.0, +7.3, +8.1 pts on
   synthetic 1:36, satimage 1:9, abalone 1:43) and edges out static
   svm_balanced on pima (+0.22, 12/15 folds) and satimage (+0.60, 12/15).
3. **Balanced logistic already matches RankSVM** (wins 12/15 and 14/15 folds
   on breast_cancer and german_credit; ties elsewhere) — consistent with
   balanced classification being the first Frank–Wolfe subproblem plus the
   Kotlowski/Agarwal consistency results, and with the original observation
   that logistic regression often outperforms the SSVM at far lower cost.
4. **Validation-selected fw_lr did not improve on lr_balanced** (mean Δ
   between −0.4 and +0.0 pts). The fixed-C ablation shows the per-iteration
   test-AUC curve does rise on imbalanced data (+0.8 pts on synthetic), so
   the gap is a model-selection problem: the 25% validation splits contain
   too few positives (~27 on synthetic, ~23 on abalone) to reliably pick the
   best iterate. Better selection (larger val fraction, pooled/cross-fitted
   selection, or selecting on training pairwise margin per the sandwich
   certificate) is the obvious next experiment.
5. Boosted trees dominate where the problem is nonlinear (satimage 0.959,
   spambase 0.987) but collapse on abalone_19 (0.824 vs 0.919 linear) —
   the meta-algorithm's plug-in nature matters: it can wrap the tree model
   too (not yet run).
