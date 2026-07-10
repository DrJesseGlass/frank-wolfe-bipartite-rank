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

**Status: not started** (blocked on Phase 1 passing). Results will land in
`results/experiments.md` + raw JSON.

## Phase 3 — Paper integration (after 1 & 2)

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
dual_coefficient_mappings.tex — rewritten theory section (drop-in)
code/verify_theorems.py       — Phase 1 checks
code/fwbpr.py                 — meta-algorithm implementation (Phase 2)
code/run_experiments.py       — Phase 2 benchmark harness
results/                      — experiment outputs
```
