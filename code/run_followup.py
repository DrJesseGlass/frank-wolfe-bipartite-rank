"""Follow-up experiments.

Exp 1 (selection + margin): does fw_lr beat balanced logistic once iterate
selection is done properly? Joint (C, t) selection by inner 2-fold CV curves
averaged across folds, refit on the full training fold at the chosen (C, t).
Margins 0 and 2 (on log-odds). Baseline: lr_balanced with the identical
inner-CV C selection and full-fold refit.
Datasets: synthetic_1to50, satimage_4, abalone_19 (imbalanced, where the
ablation showed headroom) + german_credit (near-balanced control).

Exp 2 (plug-in trees): FW-BPR wrapped around HistGradientBoosting vs plain /
balanced GBT, main-benchmark protocol (75/25 fit/val split, val selects the
iterate). Datasets: synthetic_1to50, spambase, satimage_4, abalone_19.

Writes results/followup.md, followup_raw.json.
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler

from fwbpr import (FWBPRanker, _score, balanced_counts, counts_to_weights,
                   pad_curve)
from run_experiments import load_datasets, make_lr

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
C_GRID = [0.1, 1.0, 10.0]
N_ITER = 8
EXP1_DATASETS = ["synthetic_1to50", "satimage_4", "abalone_19", "german_credit"]
EXP2_DATASETS = ["synthetic_1to50", "spambase", "satimage_4", "abalone_19"]


def fw_val_curve(C, margin, Xf, yf, Xv, yv):
    r = FWBPRanker(make_lr(C), n_iter=N_ITER, margin=margin)
    r.fit(Xf, yf, Xv, yv)
    return np.array(pad_curve([h["auc"] for h in r.history_], N_ITER))


def _inner_scaled(Xtr, ytr, seed, n_inner=2):
    """Inner-CV splits with the scaler fit on each inner-training part only
    (no leakage of inner-validation statistics into candidate models)."""
    inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
    for i, v in inner.split(Xtr, ytr):
        sc = StandardScaler().fit(Xtr[i])
        yield sc.transform(Xtr[i]), ytr[i], sc.transform(Xtr[v]), ytr[v]


def fw_lr_cvselect(Xtr, ytr, Xt, yt, margin, seed):
    """Joint (C, t) by inner-2-fold mean validation curve; refit at (C, t).
    Takes UNSCALED data; scaling is fit within each inner split and refit
    on the full training fold for the final model."""
    best = (-np.inf, None, None)
    for C in C_GRID:
        curves = [fw_val_curve(C, margin, Xf, yf, Xv, yv)
                  for Xf, yf, Xv, yv in _inner_scaled(Xtr, ytr, seed)]
        mean = np.mean(curves, axis=0)
        t = int(mean.argmax())
        if mean[t] > best[0]:
            best = (mean[t], C, t)
    _, C, t = best
    sc = StandardScaler().fit(Xtr)
    r = FWBPRanker(make_lr(C), n_iter=t + 1, margin=margin)
    r.fit(sc.transform(Xtr), ytr)
    model = r.models_[min(t, len(r.models_) - 1)]
    return roc_auc_score(yt, _score(model, sc.transform(Xt))), C, t


def lr_balanced_cvselect(Xtr, ytr, Xt, yt, seed):
    best = (-np.inf, None)
    for C in C_GRID:
        aucs = []
        for Xf, yf, Xv, yv in _inner_scaled(Xtr, ytr, seed):
            m = make_lr(C, balanced=True).fit(Xf, yf)
            aucs.append(roc_auc_score(yv, _score(m, Xv)))
        if np.mean(aucs) > best[0]:
            best = (np.mean(aucs), C)
    sc = StandardScaler().fit(Xtr)
    m = make_lr(best[1], balanced=True).fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yt, _score(m, sc.transform(Xt)))


def exp1(datasets):
    print("Exp 1: inner-CV iterate selection for fw_lr (+ margin sweep)")
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=42)
    methods = ["lr_balanced_cv", "fw_lr_cv_m0", "fw_lr_cv_m2"]
    raw = {}
    for dname in EXP1_DATASETS:
        X, y = datasets[dname]
        raw[dname] = {m: [] for m in methods}
        raw[dname]["chosen_t_m0"] = []
        t0 = time.time()
        for fold, (tr, te) in enumerate(rskf.split(X, y)):
            Xtr, Xt = X[tr], X[te]   # unscaled; helpers scale per split
            ytr, yt = y[tr], y[te]
            raw[dname]["lr_balanced_cv"].append(
                lr_balanced_cvselect(Xtr, ytr, Xt, yt, seed=fold))
            a, _, t_sel = fw_lr_cvselect(Xtr, ytr, Xt, yt, 0.0, seed=fold)
            raw[dname]["fw_lr_cv_m0"].append(a)
            raw[dname]["chosen_t_m0"].append(t_sel)
            a, _, _ = fw_lr_cvselect(Xtr, ytr, Xt, yt, 2.0, seed=fold)
            raw[dname]["fw_lr_cv_m2"].append(a)
        msg = f"  {dname} ({time.time()-t0:.0f}s): "
        for m in methods:
            v = np.array(raw[dname][m])
            msg += f"{m}={v.mean():.4f}±{v.std():.3f} "
        msg += f"median_t={int(np.median(raw[dname]['chosen_t_m0']))}"
        print(msg)
        _dump(raw, "followup_exp1")
    return raw


def make_gbt():
    return HistGradientBoostingClassifier(random_state=0)


def exp2(datasets):
    print("Exp 2: FW-BPR around gradient-boosted trees")
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=42)
    methods = ["gbt_plain", "gbt_balanced", "fw_gbt"]
    raw = {}
    for dname in EXP2_DATASETS:
        X, y = datasets[dname]
        raw[dname] = {m: [] for m in methods}
        t0 = time.time()
        for fold, (tr, te) in enumerate(rskf.split(X, y)):
            Xtr, ytr, Xt, yt = X[tr], y[tr], X[te], y[te]
            Xf, Xv, yf, yv = train_test_split(
                Xtr, ytr, test_size=0.25, stratify=ytr, random_state=fold)
            m = make_gbt().fit(Xf, yf)
            raw[dname]["gbt_plain"].append(
                roc_auc_score(yt, _score(m, Xt)))
            m = make_gbt().fit(
                Xf, yf, sample_weight=counts_to_weights(balanced_counts(yf)))
            raw[dname]["gbt_balanced"].append(
                roc_auc_score(yt, _score(m, Xt)))
            r = FWBPRanker(make_gbt(), n_iter=N_ITER, margin=0.0)
            r.fit(Xf, yf, Xv, yv)
            raw[dname]["fw_gbt"].append(
                roc_auc_score(yt, r.decision_function(Xt)))
        msg = f"  {dname} ({time.time()-t0:.0f}s): "
        for m in methods:
            v = np.array(raw[dname][m])
            msg += f"{m}={v.mean():.4f}±{v.std():.3f} "
        print(msg)
        _dump(raw, "followup_exp2")
    return raw


def _dump(obj, name):
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=1)


def paired(raw, a, b):
    x = np.array(raw[a], dtype=float)
    y = np.array(raw[b], dtype=float)
    return np.mean(x - y), int((x > y).sum()), len(x)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    datasets = load_datasets(quick=False)
    # --exp1-only / --exp2-only rerun one experiment and reuse the other's
    # checkpoint for the markdown summary
    def cached(name):
        with open(os.path.join(RESULTS_DIR, f"{name}.json")) as f:
            return json.load(f)
    r1 = cached("followup_exp1") if "--exp2-only" in sys.argv else exp1(datasets)
    r2 = cached("followup_exp2") if "--exp1-only" in sys.argv else exp2(datasets)
    lines = ["# Follow-up experiments", "",
             "## Exp 1: proper (C, t) selection for fw_lr "
             "(inner 2-fold CV, refit at chosen iterate)", "",
             "| dataset | lr_balanced_cv | fw_lr_cv_m0 | fw_lr_cv_m2 | "
             "Δ(m0−bal) | m0 fold wins |", "|---|---|---|---|---|---|"]
    for d, per in r1.items():
        cells = [f"{np.mean(per[m]):.4f}±{np.std(per[m]):.3f}"
                 for m in ["lr_balanced_cv", "fw_lr_cv_m0", "fw_lr_cv_m2"]]
        dm, wins, n = paired(per, "fw_lr_cv_m0", "lr_balanced_cv")
        lines.append(f"| {d} | " + " | ".join(cells)
                     + f" | {dm:+.4f} | {wins}/{n} |")
    lines += ["", "## Exp 2: FW-BPR around gradient-boosted trees", "",
              "| dataset | gbt_plain | gbt_balanced | fw_gbt | "
              "Δ(fw−balanced) | fw fold wins |", "|---|---|---|---|---|---|"]
    for d, per in r2.items():
        cells = [f"{np.mean(per[m]):.4f}±{np.std(per[m]):.3f}"
                 for m in ["gbt_plain", "gbt_balanced", "fw_gbt"]]
        dm, wins, n = paired(per, "fw_gbt", "gbt_balanced")
        lines.append(f"| {d} | " + " | ".join(cells)
                     + f" | {dm:+.4f} | {wins}/{n} |")
    with open(os.path.join(RESULTS_DIR, "followup.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote results/followup.md")


if __name__ == "__main__":
    main()
