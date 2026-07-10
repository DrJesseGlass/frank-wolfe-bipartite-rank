# FW-BPR benchmark results

Test AUC, mean ± std over 3-fold × 2-repeat stratified CV. C tuned on a 25% validation split of each training fold; FW-BPR iterate also selected on validation.

| dataset | lr_plain | lr_balanced | fw_lr | svm_plain | svm_balanced | fw_svm | ranksvm | gbt |
|---|---|---|---|---|---|---|---|---|
| breast_cancer | 0.9924±0.005 | 0.9937±0.005 | **0.9940±0.005** | 0.9940±0.007 | 0.9928±0.006 | 0.9923±0.006 | 0.9905±0.006 | 0.9899±0.006 |
| synthetic_1to50 | 0.7930±0.023 | 0.7913±0.016 | **0.7979±0.021** | 0.7410±0.022 | 0.7924±0.020 | 0.7932±0.018 | 0.7894±0.017 | 0.7952±0.025 |
| pima_diabetes | 0.8262±0.016 | 0.8261±0.014 | 0.8241±0.011 | **0.8266±0.015** | 0.8257±0.013 | 0.8262±0.014 | 0.8259±0.013 | 0.7899±0.016 |
| german_credit | 0.7817±0.028 | **0.7825±0.028** | 0.7786±0.027 | 0.7723±0.029 | 0.7769±0.027 | 0.7753±0.027 | 0.7764±0.028 | 0.7656±0.034 |
| spambase | 0.9704±0.003 | 0.9706±0.003 | 0.9707±0.003 | 0.9709±0.002 | 0.9705±0.002 | 0.9706±0.002 | 0.9703±0.004 | **0.9859±0.003** |

## Paired comparison: fw_lr vs lr_balanced (per fold)

- breast_cancer: mean Δ = +0.0003, fw_lr wins 2/6 folds
- synthetic_1to50: mean Δ = +0.0066, fw_lr wins 3/6 folds
- pima_diabetes: mean Δ = -0.0019, fw_lr wins 1/6 folds
- german_credit: mean Δ = -0.0040, fw_lr wins 1/6 folds
- spambase: mean Δ = +0.0001, fw_lr wins 2/6 folds
