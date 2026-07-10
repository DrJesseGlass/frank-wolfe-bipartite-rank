# FW-BPR: Validation & Experiments Plan

*Created 2026-07-10. Companion to `dual_coefficient_mappings.tex` (the margin-2
rewrite of the Dual Coefficient Mappings section).*

## Context

The paper's claims, in dependency order:

1. **Theory (rewritten, needs machine verification):** Under the margin-2
   pairwise hinge, the RankSVM dual is exactly the balanced cost-sensitive SVM
   dual restricted to a capacitated transportation polytope `T = Φ(R)`
   (thm:equivalence, thm:image). Membership in `T` is a max-flow / Gale-prefix
   test (cor:membership); the balanced optimum lower-bounds the pairwise
   optimum (cor:sandwich); Frank–Wolfe vertices are misordering-count-weighted
   classification problems, and balanced classification is the first FW
   subproblem (cor:fw_bridge).
2. **Algorithm:** dual-free plug-in meta-algorithm — start from balanced
   class weights (n⁻ per positive, n⁺ per negative), refit, reweight each
   example by its count of misordered opposite-class examples, repeat.
3. **Empirical punchline:** applying the reweighting scheme to *logistic
   regression* beats the SSVM and competes with direct AUC optimizers,
   especially under class imbalance.

## Phase 1 — Numerical verification of the theorems  ✅ do first

Script: `code/verify_theorems.py`. Tiny problems (e.g. n⁺=5, n⁻=7, 2–6 dims,
several random seeds) where both duals are solvable to high accuracy with
scipy (L-BFGS-B for box-only QPs, SLSQP for the equality-constrained balanced
dual, HiGHS `linprog` for transportation feasibility LPs).

Checks, mapped to statements in the .tex:

- [x] **rem:margin2** — rescaling identity `margin-2(C) = 4 × unit-margin(C/2)`
      at random w; two-hinge domination inequality.
- [ ] **lem:holistic_surjection** — layer-cake construction reproduces random
      λ ∈ [0,C]^N with ≤ N nested vertices summing to C.
- [ ] **eq:margin_identity** — Δᵀα = 𝟙ᵀSα for random vertex mixtures.
- [ ] **thm:equivalence (i)–(iii)** — for random ρ ∈ [0,C̄]: same w, equal
      objectives (2·𝟙ᵀρ = 𝟙ᵀλ), image in B incl. Σtλ = 0.
- [ ] **Strong duality sanity** (catches convention/factor errors): pairwise
      primal(w*) = −(pairwise dual min); balanced primal(w*,b*) = −(balanced
      dual min), b* recovered from KKT.
- [ ] **thm:image (ii)** — every Φ(ρ) passes the Gale prefix conditions;
      Gale test ⟺ transportation-LP feasibility on random points of B
      (both directions exercised); ex:strict point (2C̄,0)/(2C̄,0) fails both.
- [ ] **thm:image (i)** — random convex combos of C̄·Se_s are in T.
- [ ] **cor:sandwich** — D(λ*_bal) ≤ D(λ*_pair) ≤ D(λ) for λ ∈ T; equality
      of pairwise optimum with min over T.
- [ ] **cor:fw_bridge** — FW linear-minimization over T at iterate w selects
      exactly ρ_ij = C̄·1{2 − ⟨w, x_i−x_j⟩ > 0}, i.e. λ = C̄ × misordering
      counts; at w = 0 this is the balanced weights (n⁻, n⁺).
- [ ] **rem:relaxation** — trust-region boxes [0, C̄ω] contain points outside T.

Exit criterion: all checks pass across seeds at tolerance ~1e-6 (1e-4 for
solver-accuracy-limited comparisons). Any failure → fix the .tex first.
**Status: PASSED 2026-07-10 — 241/241 checks over 5 seeds × sizes
(5,7), (4,4), (8,3). Duality gaps ≤ 1.6e-10; Gale ⟺ LP agreed on every
sample; sandwich strict in all instances (balanced optimum ∉ T generally).**
(Remaining `- [ ]` boxes above are covered by the same run; kept for the
checklist mapping.)

Parallel task: adversarial re-read of the thm:image max-flow proof (the one
genuinely new proof) by a fresh reviewer agent.
**Status: DONE — verdict SOUND-WITH-NITS; independent 209-sample numerical
check, 0 disagreements. Four prose nits applied to the .tex on 2026-07-10:
strictness padded to general n± ≥ 2, necessity clause added in proof of (ii),
equal-sum/nonnegativity caveat in cor:membership, L±(·) wording. Still open:
the self-flagged `\ref{sec:algorithms}` TODO.**

## Phase 2 — Proof-of-concept experiments

Code: `code/fwbpr.py` (meta-algorithm), `code/run_experiments.py`.

**Meta-algorithm (the deliverable of the paper):**
given any weight-accepting classifier `fit(X, y, sample_weight)`:
1. w⁰: weights = n⁻ for positives, n⁺ for negatives (balanced). Fit.
2. Score training set; for each positive, weight ← #negatives scored above
   it (margin-violating under the surrogate); for each negative, weight ←
   #positives scored below it. Refit. Repeat T times or until weights stall.
3. Model selection across iterates by validation AUC.

**Learners:** logistic regression (primary), linear SVM (SSVM comparator),
optionally gradient-boosted trees.

**Baselines (must-beat in bold):**
- plain logistic; **class-balanced logistic** (static weights — isolates the
  value of *iterating* beyond the known consistency result of
  Kotlowski et al. 2011 / Agarwal 2014);
- linear RankSVM (exact pairwise objective, small sets; SGD-pairwise for
  larger);
- XGBoost/LightGBM with scale_pos_weight (practitioner reference), if
  installed.

**Datasets** (small → medium, imbalance ratios ~1:2 to ~1:100):
- sklearn built-in: breast_cancer;
- OpenML (fetch_openml): german credit, spambase, pima diabetes, abalone
  (binarized), satimage (class 4 vs rest), letter (one-vs-rest);
- LIBSVM downloads if network allows: a9a, w8a, ijcnn1;
- synthetic imbalanced Gaussians as a controlled sanity case.

**Protocol:** repeated stratified train/val/test splits (5×), tune C /
regularization on val AUC, report test AUC mean ± std. Key ablation:
AUC vs. reweighting iteration t (t=0 is the balanced baseline).

**Status: DONE 2026-07-10** — full results in `results/experiments.md`
(+ `experiments_raw.json`, `ablation.md`, `ablation_raw.json`).
Headline findings (details in experiments.md §Findings):
- fw_svm ≈ RankSVM on all 7 datasets without forming pairs (best empirical
  support for cor:fw_bridge); repairs plain SVM by +5–8 pts under imbalance;
  edges static svm_balanced on pima/satimage (12/15 folds each).
- Balanced logistic already matches RankSVM (theory-consistent: first FW
  step + Kotlowski/Agarwal) — supports the "logistic beats SSVM" observation.
- fw_lr ≈ lr_balanced under validation selection even though the fixed-C
  iteration curve rises on imbalanced data (+0.8 pts): iterate selection on
  small validation splits is the bottleneck.

**Follow-up experiments: DONE 2026-07-10** → `results/followup.md`
(+ `followup_exp1.json`, `followup_exp2.json`, `followup.log`).
- Iterate selection fixed via inner-2-fold (C, t) cross-fitting: fw_lr now
  +0.59 pts on synthetic 1:36 (10/15 folds), selects t=0 (i.e. falls back to
  balanced, no harm) where balanced is already optimal. Margin 2 on log-odds
  adds nothing over margin 0.
- Tree wrap: GBT interpolates (train AUC 1.0) → counts vanish → fw_gbt ≡
  gbt_balanced exactly; static balancing itself doesn't help GBT on these
  sets. Open extension (not run): out-of-fold counts / calibrated margins.

**Overall empirical verdict for the paper:** count-reweighting is a faithful
O(N log N) substitute for pairwise training (exact at the gradient level,
matches RankSVM and pairwise-logistic nets everywhere), a safe free option
over static balancing with cross-fitted selection, but NOT a broad
improvement over balanced/plain BCE — consistent with AUC-consistency theory.
Frame as equivalence + geometry + certificate paper, not a new-SOTA method.

## Phase 2b — Deep-net verification (added 2026-07-10, user request)

Claim to verify: count-reweighting approximates the pairwise objective for
ANY differentiable scorer, because of a gradient identity — grad_theta of the
pairwise loss equals grad_theta of a count-weighted linear score loss (hard
counts for the margin hinge, sigmoid soft counts for pairwise logistic).
Script: `code/deepnet_check.py` (PyTorch).

- Part A (exact identity, float64 autograd, 5 seeds):
  **PASS — hinge diff exactly 0.0, logistic diff ~1e-13.**
- Part B (MLP 64-64, per-epoch count-refreshed weighted BCE vs static
  balanced BCE vs full pairwise-logistic training; 5 splits, epoch selection
  on validation; synthetic_1to50, satimage_4, abalone_19, spambase):
  **DONE** → `results/deepnet.md` §Findings. fw_bce tracks pair_logistic
  within noise on all 4 datasets (wins 5/5 satimage, 4/5 abalone splits) at
  1.3–4.4× lower per-epoch cost; caveat — plain BCE with validation-epoch
  selection is already a strong AUC baseline on tabular MLPs, and even true
  pairwise training doesn't beat it there, so the claim to make is
  "faithful cheap substitute for pairwise training," not "beats BCE".

Paper relevance: extends cor:fw_bridge beyond linear models; connects to
hard-example mining / ArcFace-style weighted-classification losses and deep
AUC maximization (cite Yang & Ying 2022, LibAUC).

## Phase 3 — Paper integration

**Status: DRAFT ASSEMBLED 2026-07-10 — `main.tex` compiles with tectonic
(14 pp., `main.pdf`), all citations resolved.** Structure: intro (4
contributions) → related work (Joachims/Chapelle–Keerthi/LambdaMART/LibAUC
positioning) → §3 = `dual_coefficient_mappings.tex` (input verbatim, TODOs
resolved, computed polytope-slice figure added) → §4 gradient-identity
proposition → §5 slim meta-algorithm (3 rigor readings, honest convergence)
→ §6 experiments (4 findings incl. negatives + the iterate-oscillation
phenomenon, Figure 2) → discussion. Bib: `references.bib` (23 entries).
Figures generated by `code/make_figures.py`.

Remaining polish for the user:
- [ ] Two overfull hboxes (dual_coefficient_mappings.tex lines ~284, ~291).
- [ ] Decide whether to salvage anything from the old Algorithms section
      (trust-region variants, kernel/density lemmas) — current §5 replaces it.
- [ ] Read-through for voice; author block/acknowledgements; venue formatting.
- [ ] Public code release if submitting (code/ is self-contained).

### Original phase-3 checklist (superseded by the above)

- Fold verification + experiment results into the draft; figures: AUC vs.
  iteration; polytope picture (fig:coeff_maps).
- Fix Algorithms section issues flagged in review: stopping-criterion units
  (∇D(λ)ᵀ(λ−ω) mixes spaces), non-convex bias pseudocode (`argsort` target,
  bias midpoint), `\if{False}` → `\iffalse`, honest convergence statement
  (relaxed vs. pairwise optimum).
- Citations: Joachims 2005/2006; Lacoste-Julien et al. 2013; Kotlowski,
  Dembczyński, Hüllermeier 2011; Agarwal 2014; Gale 1957; Yang & Ying 2022.
- Reconcile TODO labels in `dual_coefficient_mappings.tex` with the rest of
  the paper; compile check (no TeX on this machine — needs user's machine or
  tectonic install).
- Apply any fixes the adversarial reviewer of thm:image surfaces.

## File map

```
PLAN.md                       — this file
main.tex                      — full paper (compiles: `tectonic main.tex`)
references.bib                — bibliography
dual_coefficient_mappings.tex — theory section (input by main.tex)
Figures/                      — polytope_slice.pdf, auc_vs_iteration.pdf
code/verify_theorems.py       — Phase 1 checks (241/241 pass)
code/fwbpr.py                 — meta-algorithm implementation
code/run_experiments.py       — linear benchmark harness (checkpoint-resume)
code/run_ablation.py          — iteration ablation
code/deepnet_check.py         — gradient identity + MLP comparison (torch)
code/run_followup.py          — selection / margin / tree experiments
code/make_figures.py          — paper figures
results/                      — all experiment outputs (.md + raw .json)
```
