"""Ablation: test AUC as a function of the FW-BPR reweighting iteration.

Iteration 0 is the balanced fit (static class weights), so the curve directly
shows what *iterating* adds over the known-consistent balanced baseline.
Fixed C per dataset (the mid-grid value 1.0 unless overridden); 3-fold x 5
repeats; per-iteration test AUC averaged over folds.

Writes results/ablation.md and results/ablation_raw.json.
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from fwbpr import FWBPRanker, _score, pad_curve
from run_experiments import load_datasets, make_lr, make_svm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
N_ITER = 12


def trace(make_model, margin, X, y, rskf):
    """Per-iteration test AUC, padded with the last value if converged early."""
    curves = []
    for fold, (tr, te) in enumerate(rskf.split(X, y)):
        Xtr, ytr, Xt, yt = X[tr], y[tr], X[te], y[te]
        sc = StandardScaler().fit(Xtr)
        Xtr, Xt = sc.transform(Xtr), sc.transform(Xt)
        r = FWBPRanker(make_model(), n_iter=N_ITER, margin=margin)
        r.fit(Xtr, ytr)
        aucs = [roc_auc_score(yt, _score(m, Xt)) for m in r.models_]
        curves.append(pad_curve(aucs, N_ITER))
    return np.array(curves)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    datasets = load_datasets(quick=False)
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=42)
    raw, lines = {}, [
        "# FW-BPR iteration ablation",
        "",
        "Mean test AUC by reweighting iteration (t=0 is the balanced fit),",
        "C=1.0 fixed, 3-fold x 5-repeat CV. `fw_lr`: logistic, margin 0;",
        "`fw_svm`: hinge, margin 2.",
        ""]
    for dname, (X, y) in datasets.items():
        raw[dname] = {}
        lines += [f"## {dname}", "",
                  "| t | " + " | ".join(str(t) for t in range(N_ITER)) + " |",
                  "|---|" + "---|" * N_ITER]
        for label, mk, margin in [("fw_lr", lambda: make_lr(1.0), 0.0),
                                  ("fw_svm", lambda: make_svm(1.0), 2.0)]:
            c = trace(mk, margin, X, y, rskf)
            mean = c.mean(axis=0)
            raw[dname][label] = c.tolist()
            best_t = int(mean.argmax())
            cells = []
            for t, v in enumerate(mean):
                s = f"{v:.4f}"
                cells.append(f"**{s}**" if t == best_t else s)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
            print(f"{dname}/{label}: t0={mean[0]:.4f} -> "
                  f"best t={best_t} ({mean[best_t]:.4f}), "
                  f"delta={mean[best_t]-mean[0]:+.4f}")
        lines.append("")
        with open(os.path.join(RESULTS_DIR, "ablation_raw.json"), "w") as f:
            json.dump(raw, f)
    with open(os.path.join(RESULTS_DIR, "ablation.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote results/ablation.md")


if __name__ == "__main__":
    main()
