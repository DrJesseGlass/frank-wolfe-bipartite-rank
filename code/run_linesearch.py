"""Line-searched FW-BPR (ensemble variant) vs raw refits.

Traces per-iteration test AUC of FWBPREnsembleRanker on the two ablation
panels (synthetic/logistic margin 0; satimage/hinge margin 2), same
3-fold x 5-repeat protocol, and records the chosen step sizes gamma.
The test set is passed as the trace set (no selection happens on it).

Writes results/linesearch_raw.json and prints a comparison against the raw
(gamma = 1) curves from results/ablation_raw.json.
"""

import json
import os

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from fwbpr import FWBPREnsembleRanker, pad_curve
from run_experiments import load_datasets, make_lr, make_svm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
N_ITER = 12

PANELS = [("synthetic_1to50", "fw_lr", lambda: make_lr(1.0), 0.0),
          ("satimage_4", "fw_svm", lambda: make_svm(1.0), 2.0)]


def main():
    datasets = load_datasets(quick=False)
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=42)
    with open(os.path.join(RESULTS_DIR, "ablation_raw.json")) as f:
        abl = json.load(f)
    out = {}
    for dname, key, mk, margin in PANELS:
        X, y = datasets[dname]
        curves, gammas = [], []
        for tr, te in rskf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xt = sc.transform(X[tr]), sc.transform(X[te])
            r = FWBPREnsembleRanker(mk(), n_iter=N_ITER, margin=margin)
            r.fit(Xtr, y[tr], Xt, y[te])   # test passed as trace set only
            curves.append(pad_curve([h["val_auc"] for h in r.history_],
                                    N_ITER))
            gammas.append([h["gamma"] for h in r.history_])
        c = np.array(curves)
        out[dname] = dict(method=key, curves=c.tolist(), gammas=gammas)
        abl_c = np.array(abl[dname][key])
        raw = abl_c.mean(0)
        m = c.mean(0)
        gmed = [round(float(np.median([g[t] for g in gammas
                                       if len(g) > t])), 3)
                for t in range(1, min(6, N_ITER))]
        print(f"{dname} ({key}, margin={margin}):")
        print(f"  raw  (gamma=1): " + " ".join(f"{v:.4f}" for v in raw))
        print(f"  line-searched : " + " ".join(f"{v:.4f}" for v in m))
        print(f"  median gamma t=1..5: {gmed}")
        print(f"  final: raw best-so-far {np.maximum.accumulate(abl_c, 1).mean(0)[-1]:.4f}, "
              f"LS final {m[-1]:.4f}, balanced {m[0]:.4f}")
    with open(os.path.join(RESULTS_DIR, "linesearch_raw.json"), "w") as f:
        json.dump(out, f)
    print("Wrote results/linesearch_raw.json")


if __name__ == "__main__":
    main()
