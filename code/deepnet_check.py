"""Deep-net verification of the count-reweighting <-> pairwise-loss claim.

Part A (exact): for an MLP scorer s_theta, the parameter gradient of the
pairwise loss equals the gradient of a per-example *linear* score loss whose
weights are the (frozen) violation counts:
    hinge:    d/ds_i sum_ij [m - (s_i - s_j)]_+  = -#{j : s_i - s_j < m}
    logistic: d/ds_i sum_ij softplus(s_j - s_i)  = -sum_j sigmoid(s_j - s_i)
so grad_theta(pairwise) == grad_theta(sum_j c_j s_j - sum_i c_i s_i).
Verified with autograd to float64 precision across random nets/data.

Part B (practical): training an MLP with per-epoch count-reweighted BCE
approaches the test AUC of training on the full pairwise logistic loss,
and beats static balanced weights, at O(N log N) per epoch instead of
O(n+ * n-). Protocol: 5 stratified 60/20/20 splits; per-epoch validation
AUC selects the reported model for every method identically.

Writes results/deepnet.md and results/deepnet_raw.json.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fwbpr import violation_counts
from run_experiments import load_datasets

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
torch.set_num_threads(max(1, os.cpu_count() - 2))


def make_mlp(d, hidden=(64, 64), seed=0):
    g = torch.Generator().manual_seed(seed)
    layers, prev = [], d
    for h in hidden:
        lin = torch.nn.Linear(prev, h)
        torch.nn.init.kaiming_uniform_(lin.weight, generator=g)
        layers += [lin, torch.nn.ReLU()]
        prev = h
    layers.append(torch.nn.Linear(prev, 1))
    return torch.nn.Sequential(*layers)


# ------------------------------------------------------------------ Part A

def part_a():
    print("Part A: gradient identity through an MLP (float64 autograd)")
    rows, ok_all = [], True
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n_pos, n_neg, d = 20, 30, 7
        X = torch.tensor(rng.normal(size=(n_pos + n_neg, d)))
        y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
        net = make_mlp(d, seed=seed).double()

        def grads(loss_fn):
            net.zero_grad()
            s = net(X).squeeze(1)
            loss_fn(s).backward()
            return torch.cat([p.grad.flatten().clone()
                              for p in net.parameters()])

        pos, neg = torch.tensor(y == 1), torch.tensor(y == 0)

        # hinge, margin 2, vs hard-count weighted linear loss
        def pair_hinge(s):
            return torch.clamp(2.0 - (s[pos][:, None] - s[neg][None, :]),
                               min=0).sum()

        with torch.no_grad():
            s0 = net(X).squeeze(1).numpy()
        c = violation_counts(s0, y, margin=2.0)
        cw = torch.tensor(c)

        def lin_hard(s):
            return (cw[neg] * s[neg]).sum() - (cw[pos] * s[pos]).sum()

        # ties in the hinge subgradient would break exact equality; skip if any
        margins = s0[y == 1][:, None] - s0[y == 0][None, :]
        if np.any(np.abs(2.0 - margins) < 1e-9):
            continue
        d_h = (grads(pair_hinge) - grads(lin_hard)).abs().max().item()

        # logistic, vs soft-count weighted linear loss
        def pair_logistic(s):
            return F.softplus(-(s[pos][:, None] - s[neg][None, :])).sum()

        sp, sn = s0[y == 1], s0[y == 0]
        soft_pos = 1.0 / (1.0 + np.exp(-(sn[None, :] - sp[:, None])))
        c_soft = np.zeros(y.size)
        c_soft[y == 1] = soft_pos.sum(axis=1)
        c_soft[y == 0] = soft_pos.sum(axis=0)
        cs = torch.tensor(c_soft)

        def lin_soft(s):
            return (cs[neg] * s[neg]).sum() - (cs[pos] * s[pos]).sum()

        d_l = (grads(pair_logistic) - grads(lin_soft)).abs().max().item()
        rows.append((seed, d_h, d_l))
        ok_all &= d_h < 1e-10 and d_l < 1e-10
        print(f"  seed {seed}: max|grad diff| hinge={d_h:.2e}, "
              f"logistic={d_l:.2e}")
    print(f"Part A: {'PASS' if ok_all else 'FAIL'}")
    return ok_all, rows


# ------------------------------------------------------------------ Part B

def train(method, Xtr, ytr, Xv, yv, Xt, yt, seed, epochs=400, margin=2.0):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    net = make_mlp(d, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)
    pos = torch.tensor(ytr == 1)
    neg = torch.tensor(ytr == 0)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    N = len(ytr)
    w = torch.tensor(np.where(ytr == 1, n_neg, n_pos), dtype=torch.float32)
    w = w * (N / w.sum())  # balanced init for fw_bce
    best_val, test_at_best, t_train = -np.inf, np.nan, 0.0
    for ep in range(epochs):
        t0 = time.time()
        net.train()
        opt.zero_grad()
        s = net(Xtr_t).squeeze(1)
        if method == "bce_plain":
            loss = F.binary_cross_entropy_with_logits(s, ytr_t)
        elif method == "bce_balanced":
            wb = torch.where(pos, float(n_neg), float(n_pos))
            loss = F.binary_cross_entropy_with_logits(
                s, ytr_t, weight=wb * (N / wb.sum()))
        elif method == "fw_bce":
            loss = F.binary_cross_entropy_with_logits(s, ytr_t, weight=w)
        elif method == "pair_logistic":
            loss = F.softplus(-(s[pos][:, None] - s[neg][None, :])).mean()
        else:
            raise ValueError(method)
        loss.backward()
        opt.step()
        if method == "fw_bce":  # refresh counts from the new scores
            with torch.no_grad():
                s_now = net(Xtr_t).squeeze(1).numpy()
            c = violation_counts(s_now, ytr, margin=margin)
            if c.sum() > 0:
                w = torch.tensor(c * (N / c.sum()), dtype=torch.float32)
        t_train += time.time() - t0
        net.eval()
        with torch.no_grad():
            auc_v = roc_auc_score(yv, net(Xv_t).squeeze(1).numpy())
            if auc_v > best_val:
                best_val = auc_v
                test_at_best = roc_auc_score(
                    yt, net(Xt_t).squeeze(1).numpy())
    return test_at_best, t_train / epochs


METHODS = ["bce_plain", "bce_balanced", "fw_bce", "pair_logistic"]
DATASETS = ["synthetic_1to50", "satimage_4", "abalone_19", "spambase"]


def part_b(n_splits=5):
    print("\nPart B: MLP training comparison")
    all_ds = load_datasets(quick=False)
    raw = {}
    for dname in DATASETS:
        X, y = all_ds[dname]
        raw[dname] = {m: dict(auc=[], sec_per_epoch=[]) for m in METHODS}
        for split in range(n_splits):
            Xtr, Xtmp, ytr, ytmp = train_test_split(
                X, y, test_size=0.4, stratify=y, random_state=split)
            Xv, Xt, yv, yt = train_test_split(
                Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=split)
            sc = StandardScaler().fit(Xtr)
            Xtr, Xv, Xt = sc.transform(Xtr), sc.transform(Xv), sc.transform(Xt)
            for m in METHODS:
                auc, spe = train(m, Xtr, ytr, Xv, yv, Xt, yt, seed=split)
                raw[dname][m]["auc"].append(auc)
                raw[dname][m]["sec_per_epoch"].append(spe)
        line = f"  {dname}: "
        for m in METHODS:
            v = np.array(raw[dname][m]["auc"])
            line += f"{m}={v.mean():.4f}±{v.std():.3f} "
        print(line)
    return raw


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ok_a, rows_a = part_a()
    raw_b = part_b()
    with open(os.path.join(RESULTS_DIR, "deepnet_raw.json"), "w") as f:
        json.dump(dict(part_a=dict(passed=ok_a, rows=rows_a), part_b=raw_b),
                  f, indent=1)
    lines = [
        "# Deep-net verification", "",
        "## Part A: exact gradient identity",
        "",
        "For an MLP scorer, grad_theta(pairwise loss) == grad_theta("
        "count-weighted linear score loss) with counts frozen at the current",
        "scores — hard violation counts for the margin-2 hinge, soft counts",
        "(sums of sigmoids) for the pairwise logistic. Autograd, float64:",
        "",
        f"Result: **{'PASS' if ok_a else 'FAIL'}** — max |grad component "
        "diff| across 5 seeds: "
        + ", ".join(f"{max(r[1], r[2]):.1e}" for r in rows_a),
        "",
        "## Part B: MLP training, test AUC "
        "(5 stratified 60/20/20 splits, epoch selected on validation)", "",
        "| dataset | " + " | ".join(METHODS) + " | pairwise cost multiple |",
        "|---|" + "---|" * (len(METHODS) + 1)]
    for dname, per in raw_b.items():
        means = {m: np.mean(per[m]["auc"]) for m in METHODS}
        top = max(means.values())
        cells = []
        for m in METHODS:
            v = np.array(per[m]["auc"])
            s = f"{v.mean():.4f}±{v.std():.3f}"
            cells.append(f"**{s}**" if means[m] >= top - 1e-9 else s)
        ratio = (np.mean(per["pair_logistic"]["sec_per_epoch"])
                 / np.mean(per["fw_bce"]["sec_per_epoch"]))
        lines.append(f"| {dname} | " + " | ".join(cells)
                     + f" | {ratio:.1f}x |")
    with open(os.path.join(RESULTS_DIR, "deepnet.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote results/deepnet.md")


if __name__ == "__main__":
    main()
