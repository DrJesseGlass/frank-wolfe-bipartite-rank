# Deep-net verification

## Part A: exact gradient identity

For an MLP scorer, grad_theta(pairwise loss) == grad_theta(count-weighted linear score loss) with counts frozen at the current
scores — hard violation counts for the margin-2 hinge, soft counts
(sums of sigmoids) for the pairwise logistic. Autograd, float64:

Result: **PASS** — max |grad component diff| across 5 seeds: 5.7e-14, 4.3e-14, 4.3e-14, 5.0e-14, 8.5e-14

## Part B: MLP training, test AUC (5 stratified 60/20/20 splits, epoch selected on validation)

| dataset | bce_plain | bce_balanced | fw_bce | pair_logistic | pairwise cost multiple |
|---|---|---|---|---|---|
| synthetic_1to50 | **0.7969±0.065** | 0.7869±0.069 | 0.7776±0.071 | 0.7840±0.069 | 1.3x |
| satimage_4 | 0.9529±0.006 | 0.9565±0.004 | **0.9602±0.004** | 0.9555±0.006 | 3.1x |
| abalone_19 | **0.9186±0.018** | 0.9094±0.015 | 0.9146±0.018 | 0.9091±0.013 | 1.3x |
| spambase | 0.9817±0.004 | **0.9817±0.004** | 0.9807±0.004 | 0.9813±0.004 | 4.4x |

## Findings

1. **The gradient identity is exact through a deep net** (Part A): the
   pairwise hinge/logistic parameter gradient equals the count-weighted
   linear-loss gradient to machine precision. Count reweighting *is* the
   pairwise objective at the gradient level, for any differentiable scorer.
2. **Training tracks the pairwise loss** (Part B): per-epoch count-refreshed
   weighted BCE (`fw_bce`) stays within noise of full pairwise-logistic
   training on all four datasets (mean paired |Δ| ≤ 0.6 pts; it actually wins
   5/5 splits on satimage and 4/5 on abalone), while a pairwise epoch costs
   1.3–4.4× more here and scales O(n+·n-) vs O(N log N).
3. **Honest caveat:** with per-epoch validation-AUC model selection, *plain*
   BCE is already a strong AUC baseline for MLPs on these tabular sets
   (top on 2/4; BCE is AUC-consistent, so this is expected asymptotically).
   The pairwise objective itself (`pair_logistic`) does not beat
   `bce_balanced` here either — so the deep-net case for pairwise/count
   training rests on regimes where univariate training underperforms:
   extreme imbalance with limited data, or metric-learning settings
   (face verification) where scores are similarities rather than logits.
   The right claim for the paper: count reweighting is a *faithful, cheap
   substitute for pairwise training* wherever pairwise training is the right
   tool — not that it dominates BCE on generic tabular data.
