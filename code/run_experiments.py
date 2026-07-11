"""Phase 2 benchmark: FW-BPR meta-algorithm vs. baselines on imbalanced
binary datasets. Protocol: RepeatedStratifiedKFold (3 folds x 5 repeats);
within each training fold, 75/25 fit/validation split used for tuning C and
(for FW-BPR) iterate selection; test AUC reported on the held-out fold.

Methods:
  lr_plain      LogisticRegression, unit weights
  lr_balanced   LogisticRegression, class_weight='balanced'  <- must-beat
  fw_lr         FW-BPR around logistic regression (margin 0)
  svm_plain     LinearSVC (hinge), unit weights
  svm_balanced  LinearSVC, class_weight='balanced'
  fw_svm        FW-BPR around LinearSVC (margin 2 = paper's convention)
  ranksvm       pairwise hinge on sampled difference vectors (exact objective,
                sampled pairs) -- the SSVM comparator
  gbt           HistGradientBoostingClassifier w/ balanced weights (reference)

Usage: python3 run_experiments.py [--quick]
Writes results/experiments_raw.json and results/experiments.md.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from fwbpr import FWBPRanker, _score, balanced_counts, counts_to_weights

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
C_GRID = [0.01, 0.1, 1.0, 10.0]
N_ITER = 12


# ------------------------------------------------------------------ datasets

def load_datasets(quick=False):
    ds = {}
    bc = load_breast_cancer()
    ds["breast_cancer"] = (bc.data, (bc.target == 0).astype(int))  # malignant=+

    Xs, ys = make_classification(
        n_samples=4000, n_features=20, n_informative=8, n_redundant=4,
        weights=[0.98, 0.02], flip_y=0.01, class_sep=0.8, random_state=7)
    ds["synthetic_1to50"] = (Xs, ys)

    def openml(name, version=1):
        d = fetch_openml(name, version=version, as_frame=False,
                         parser="liac-arff")
        return d.data, d.target

    try:
        X, t = openml("diabetes")
        ds["pima_diabetes"] = (X, (t == "tested_positive").astype(int))
        X, t = openml("credit-g")
        ds["german_credit"] = (X, (t == "bad").astype(int))
        X, t = openml("spambase")
        ds["spambase"] = (X, (t == "1").astype(int))
        if not quick:
            X, t = openml("satimage")
            ds["satimage_4"] = (X, (t == "4.").astype(int)
                                if "4." in set(t) else (t == "4").astype(int))
            X, t = openml("abalone")
            rings = t.astype(float)
            ds["abalone_19"] = (X, (rings >= 19).astype(int))
    except Exception as e:  # network gone: proceed with what we have
        print(f"[warn] openml fetch failed ({e}); continuing with built-ins")

    out = {}
    for name, (X, y) in ds.items():
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        keep = ~np.isnan(X).any(axis=1)
        X, y = X[keep], y[keep]
        n_pos = int(y.sum())
        out[name] = (X, y)
        print(f"  {name}: n={len(y)}, d={X.shape[1]}, "
              f"pos={n_pos} (1:{(len(y)-n_pos)/max(n_pos,1):.1f})")
    return out


# ------------------------------------------------------------------- methods

def make_lr(C, balanced=False):
    return LogisticRegression(
        C=C, max_iter=5000,
        class_weight="balanced" if balanced else None)


def make_svm(C, balanced=False):
    return LinearSVC(C=C, loss="hinge", max_iter=50000, tol=1e-4,
                     class_weight="balanced" if balanced else None)


def fit_ranksvm(X, y, C, rng, max_pairs=30000):
    """Pairwise hinge on difference vectors (RankSVM). Exact pair set when
    small, uniform pair sample otherwise. Mirrored +/- pairs give LinearSVC
    the two classes it requires; fit_intercept=False since bias cancels."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pairs = len(pos) * len(neg)
    if n_pairs <= max_pairs:
        ii, jj = np.meshgrid(pos, neg, indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
    else:
        ii = rng.choice(pos, size=max_pairs)
        jj = rng.choice(neg, size=max_pairs)
    Z = X[ii] - X[jj]
    Zb = np.vstack([Z, -Z])
    tb = np.concatenate([np.ones(len(Z)), -np.ones(len(Z))])
    m = LinearSVC(C=C, loss="hinge", fit_intercept=False,
                  max_iter=50000, tol=1e-4)
    m.fit(Zb, tb)
    return m


def eval_method(method, Xf, yf, Xv, yv, Xt, yt, rng):
    """Tune C on validation AUC; return test AUC of the selected model."""
    best_auc_v, best_model = -np.inf, None
    grid = C_GRID[:1] if method == "gbt" else C_GRID  # no C grid for trees
    for C in grid:
        if method == "lr_plain":
            m = make_lr(C).fit(Xf, yf)
        elif method == "lr_balanced":
            m = make_lr(C, balanced=True).fit(Xf, yf)
        elif method == "svm_plain":
            m = make_svm(C).fit(Xf, yf)
        elif method == "svm_balanced":
            m = make_svm(C, balanced=True).fit(Xf, yf)
        elif method == "fw_lr":
            m = FWBPRanker(make_lr(C), n_iter=N_ITER, margin=0.0)
            m.fit(Xf, yf, Xv, yv)
        elif method == "fw_svm":
            m = FWBPRanker(make_svm(C), n_iter=N_ITER, margin=2.0)
            m.fit(Xf, yf, Xv, yv)
        elif method == "ranksvm":
            m = fit_ranksvm(Xf, yf, C, rng)
        elif method == "gbt":
            m = HistGradientBoostingClassifier(random_state=0)
            m.fit(Xf, yf,
                  sample_weight=counts_to_weights(balanced_counts(yf)))
        else:
            raise ValueError(method)
        # ranksvm's LinearSVC has fit_intercept=False, so _score (its
        # decision_function) equals Xv @ coef_ -- no special case needed
        if isinstance(m, FWBPRanker):
            auc_v = m.best_auc_  # already computed on (Xv, yv) during fit
        else:
            auc_v = roc_auc_score(yv, _score(m, Xv))
        if auc_v > best_auc_v:
            best_auc_v, best_model = auc_v, m
    return roc_auc_score(yt, _score(best_model, Xt))


METHODS = ["lr_plain", "lr_balanced", "fw_lr",
           "svm_plain", "svm_balanced", "fw_svm",
           "ranksvm", "gbt"]


def main():
    quick = "--quick" in sys.argv
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading datasets...")
    datasets = load_datasets(quick=quick)
    n_repeats = 2 if quick else 5
    n_folds = 3 * n_repeats
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=n_repeats,
                                   random_state=42)
    # resume from checkpoint: skip datasets already fully evaluated
    ckpt_path = os.path.join(RESULTS_DIR, "experiments_raw.json")
    results = {}
    if "--fresh" not in sys.argv and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            prev = json.load(f)
        for dname, per in prev.items():
            if (dname in datasets and set(per) == set(METHODS)
                    and all(len(v) == n_folds for v in per.values())):
                results[dname] = per
                print(f"  [resume] {dname}: loaded {n_folds} folds from checkpoint")
    for dname, (X, y) in datasets.items():
        if dname in results:
            continue
        results[dname] = {m: [] for m in METHODS}
        t0 = time.time()
        for fold, (tr, te) in enumerate(rskf.split(X, y)):
            rng = np.random.default_rng(1000 + fold)
            Xtr, ytr, Xt, yt = X[tr], y[tr], X[te], y[te]
            Xf, Xv, yf, yv = train_test_split(
                Xtr, ytr, test_size=0.25, stratify=ytr,
                random_state=fold)
            sc = StandardScaler().fit(Xf)
            Xf, Xv, Xt_ = sc.transform(Xf), sc.transform(Xv), sc.transform(Xt)
            for m in METHODS:
                try:
                    auc = eval_method(m, Xf, yf, Xv, yv, Xt_, yt, rng)
                except Exception as e:
                    print(f"  [warn] {dname}/{m} fold {fold}: {e}")
                    auc = np.nan
                results[dname][m].append(auc)
        print(f"{dname}: done in {time.time()-t0:.0f}s")
        for m in METHODS:
            v = np.array(results[dname][m], dtype=float)
            print(f"    {m:13s} AUC = {np.nanmean(v):.4f} +- {np.nanstd(v):.4f}")
        # checkpoint after each dataset
        with open(ckpt_path, "w") as f:
            json.dump(results, f, indent=1)

    write_markdown(results, n_repeats)
    print("Wrote results/experiments.md")


def write_markdown(results, n_repeats):
    lines = ["# FW-BPR benchmark results",
             "",
             f"Test AUC, mean ± std over 3-fold × {n_repeats}-repeat "
             "stratified CV. C tuned on a 25% validation split of each "
             "training fold; FW-BPR iterate also selected on validation.",
             ""]
    header = "| dataset | " + " | ".join(METHODS) + " |"
    lines += [header, "|" + "---|" * (len(METHODS) + 1)]
    for dname, per in results.items():
        cells = []
        means = {m: np.nanmean(np.array(per[m], dtype=float)) for m in METHODS}
        top = max(means.values())
        for m in METHODS:
            v = np.array(per[m], dtype=float)
            s = f"{means[m]:.4f}±{np.nanstd(v):.3f}"
            if means[m] >= top - 1e-9:
                s = f"**{s}**"
            cells.append(s)
        lines.append(f"| {dname} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired comparison: fw_lr vs lr_balanced (per fold)", ""]
    for dname, per in results.items():
        a = np.array(per["fw_lr"], dtype=float)
        b = np.array(per["lr_balanced"], dtype=float)
        ok = ~(np.isnan(a) | np.isnan(b))
        wins = int(((a - b)[ok] > 0).sum())
        lines.append(f"- {dname}: mean Δ = {np.nanmean(a-b):+.4f}, "
                     f"fw_lr wins {wins}/{ok.sum()} folds")
    with open(os.path.join(RESULTS_DIR, "experiments.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
