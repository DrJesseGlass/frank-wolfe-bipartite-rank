"""Deep-net regime test: does the count-reweighted loss beat BCE where
NNs are the right architecture and imbalance is severe?

Setup (mirrors the deep-AUC-maximization literature): CIFAR-10 binarized
(automobile vs rest), training positives subsampled to ~1:50, natural test
set (1000 pos / 9000 neg) for low-variance AUC. Small CNN, Adam, epoch
selected on validation AUC identically for every method.

Methods:
  bce        plain binary cross-entropy
  bce_bal    static balanced weights (the consistency-theory baseline)
  fw_hard    per-epoch hard violation counts (margin 2 on logits) as weights
  fw_soft    per-epoch soft counts sum_j sigmoid(s_j - s_i) as weights
             (= pairwise-logistic gradient magnitudes, Prop. gradient identity)
  pair_batch true pairwise logistic on stratified batches (16 pos / 112 neg)

Writes results/cifar_raw.json (checkpointed per seed/method) and prints a
summary. Usage: python3 cifar_experiment.py [--seeds 3] [--epochs 15]
"""

import argparse
import json
import os
import tarfile
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

from fwbpr import violation_counts, soft_violation_counts

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
DATA_DIR = os.path.join(HERE, "..", "data")
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
# PNG distribution of CIFAR-10 (the cs.toronto.edu pickle mirror is
# unreliably slow; this one is S3-backed)
CIFAR_URL = "https://pjreddie.com/media/files/cifar.tgz"
N_POS_TRAIN = 450      # ~1:50 against 22500 negatives
N_NEG_TRAIN = 22500
VAL_FRAC = 0.15
BATCH = 128


def _ensure_cifar():
    d = os.path.join(DATA_DIR, "cifar")
    if os.path.isdir(d):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    tgz = os.path.join(DATA_DIR, "cifar.tgz")
    if not os.path.exists(tgz):
        print(f"downloading {CIFAR_URL} ...", flush=True)
        urllib.request.urlretrieve(CIFAR_URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(DATA_DIR)


def _load_split_pngs(split):
    """Load the pjreddie PNG distribution (data/cifar/{train,test}/
    NNNNN_label.png) into tensors, cached as .pt after first decode."""
    cache = os.path.join(DATA_DIR, f"cifar_{split}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    _ensure_cifar()
    from PIL import Image
    d = os.path.join(DATA_DIR, "cifar", split)
    files = sorted(os.listdir(d))
    X = torch.empty(len(files), 3, 32, 32)
    labels = []
    to_t = T.ToTensor()
    for i, f in enumerate(files):
        X[i] = to_t(Image.open(os.path.join(d, f)).convert("RGB"))
        labels.append(f.rsplit("_", 1)[1].split(".")[0])
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    X = (X - mean) / std
    y = (np.array(labels) == "automobile").astype("float32")
    out = (X, torch.tensor(y))
    torch.save(out, cache)
    return out


def load_cifar_binary(seed, flip_frac=0.0):
    Xtr, ytr = _load_split_pngs("train")
    Xte, yte = _load_split_pngs("test")
    rng = np.random.default_rng(seed)
    pos = np.where(ytr.numpy() == 1)[0]
    neg = np.where(ytr.numpy() == 0)[0]
    keep = np.concatenate([rng.choice(pos, N_POS_TRAIN, replace=False),
                           rng.choice(neg, N_NEG_TRAIN, replace=False)])
    rng.shuffle(keep)
    Xtr, ytr = Xtr[keep], ytr[keep]
    # label noise: pair-swap flips keep class sizes fixed; the flip mask is
    # ground truth for the mislabel-detection analysis only, never training
    flip_mask = np.zeros(len(ytr), bool)
    if flip_frac > 0:
        k = int(round(flip_frac * ytr.sum().item()))
        pos = np.where(ytr.numpy() == 1)[0]
        neg = np.where(ytr.numpy() == 0)[0]
        fp = rng.choice(pos, k, replace=False)   # true pos labeled neg
        fn = rng.choice(neg, k, replace=False)   # true neg labeled pos
        ytr[fp] = 0.0
        ytr[fn] = 1.0
        flip_mask[fp] = flip_mask[fn] = True
    n_val = int(VAL_FRAC * len(ytr))
    # stratified val split
    pos = np.where(ytr.numpy() == 1)[0]
    neg = np.where(ytr.numpy() == 0)[0]
    nvp = max(10, int(VAL_FRAC * len(pos)))
    vidx = np.concatenate([rng.choice(pos, nvp, replace=False),
                           rng.choice(neg, n_val - nvp, replace=False)])
    vmask = np.zeros(len(ytr), bool)
    vmask[vidx] = True
    return (Xtr[~vmask], ytr[~vmask], Xtr[vmask], ytr[vmask], Xte, yte,
            flip_mask[~vmask])


class SmallCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return [torch.nn.Conv2d(cin, cout, 3, padding=1),
                    torch.nn.BatchNorm2d(cout), torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2)]
        self.net = torch.nn.Sequential(
            *block(3, 32), *block(32, 64), *block(64, 128),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


@torch.no_grad()
def scores(model, X, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(DEVICE)).cpu())
    return torch.cat(out)


def counts_hard(s, y, margin):
    return violation_counts(s.numpy().astype(float), y.numpy().astype(int),
                            margin=margin)


def counts_soft(s, y):
    return soft_violation_counts(s.numpy().astype(float),
                                 y.numpy().astype(int))


def detection_auroc(s_fit, yf, flip_mask):
    """AUROC of the normalized violation count (1 - per-point AUC) for
    identifying flipped labels. Computed within each labeled class then
    pooled, since counts are class-conditional."""
    if flip_mask.sum() == 0:
        return float("nan")
    c = counts_hard(s_fit, yf, 0.0)
    denom = np.where(yf.numpy() == 1, (yf == 0).sum().item(),
                     (yf == 1).sum().item())
    stat = c / denom
    return roc_auc_score(flip_mask.astype(int), stat)


def train_method(method, data, seed, epochs):
    Xf, yf, Xv, yv, Xt, yt, flip_mask = data
    torch.manual_seed(seed)
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    N = len(yf)
    n_pos, n_neg = int(yf.sum()), int((1 - yf).sum())
    if method == "bce":
        w = torch.ones(N)
    else:  # balanced init for all weighted variants
        w = torch.where(yf == 1, float(n_neg), float(n_pos))
        w = w * (N / w.sum())
    pos_idx = torch.where(yf == 1)[0]
    neg_idx = torch.where(yf == 0)[0]
    best_val, test_at_best, best_ep = -1.0, float("nan"), -1
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        if method == "pair_batch":
            n_steps = len(neg_idx) // 112
            pperm = pos_idx[torch.randperm(len(pos_idx), generator=g)]
            nperm = neg_idx[torch.randperm(len(neg_idx), generator=g)]
            pi = 0
            for st in range(n_steps):
                nb = nperm[st * 112:(st + 1) * 112]
                pb = pperm[pi:pi + 16]
                pi += 16
                if pi + 16 > len(pperm):
                    pperm = pos_idx[torch.randperm(len(pos_idx), generator=g)]
                    pi = 0
                xb = torch.cat([Xf[pb], Xf[nb]]).to(DEVICE)
                s = model(xb)
                sp, sn = s[:len(pb)], s[len(pb):]
                loss = F.softplus(-(sp[:, None] - sn[None, :])).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        else:
            perm = torch.randperm(N, generator=g)
            for i in range(0, N, BATCH):
                b = perm[i:i + BATCH]
                xb, yb = Xf[b].to(DEVICE), yf[b].to(DEVICE)
                s = model(xb)
                loss = F.binary_cross_entropy_with_logits(
                    s, yb, weight=w[b].to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()
            if method in ("fw_hard", "fw_soft"):
                s_tr = scores(model, Xf)
                c = (counts_hard(s_tr, yf, 2.0) if method == "fw_hard"
                     else counts_soft(s_tr, yf))
                if c.sum() > 0:
                    w = torch.tensor(c * (N / c.sum()), dtype=torch.float32)
        auc_v = roc_auc_score(yv.numpy(), scores(model, Xv).numpy())
        if auc_v > best_val:
            best_val, best_ep = auc_v, ep
            test_at_best = roc_auc_score(yt.numpy(),
                                         scores(model, Xt).numpy())
            # detection needs a full-train forward pass; skip on clean runs
            det_at_best = (detection_auroc(scores(model, Xf), yf, flip_mask)
                           if flip_mask.any() else float("nan"))
    return test_at_best, best_val, best_ep, det_at_best


METHODS = ["bce", "bce_bal", "fw_hard", "fw_soft", "pair_batch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--flip", type=float, default=0.0)
    args = ap.parse_args()
    seeds, epochs, flip = args.seeds, args.epochs, args.flip
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"_flip{int(flip*100)}" if flip > 0 else ""
    path = os.path.join(RESULTS_DIR, f"cifar_raw{tag}.json")
    raw = json.load(open(path)) if os.path.exists(path) else {}
    for seed in range(seeds):
        data = load_cifar_binary(seed, flip_frac=flip)
        print(f"seed {seed} (flip={flip}): train {int(data[1].sum())}+/"
              f"{int((1-data[1]).sum())}- val {int(data[3].sum())}+ "
              f"test {int(data[5].sum())}+ flipped {int(data[6].sum())}",
              flush=True)
        for m in METHODS:
            key = f"{m}_s{seed}"
            if key in raw:
                continue
            t0 = time.time()
            auc, val, ep, det = train_method(m, data, seed, epochs)
            raw[key] = dict(test_auc=auc, val_auc=val, best_epoch=ep,
                            detect_auroc=det)
            print(f"  {m:10s} test AUC {auc:.4f} (val {val:.4f}, ep {ep}, "
                  f"detect {det:.3f}, {time.time()-t0:.0f}s)", flush=True)
            json.dump(raw, open(path, "w"), indent=1)
    print("\nSummary (mean +- std over seeds):")
    for m in METHODS:
        v = [raw[f"{m}_s{s}"]["test_auc"] for s in range(seeds)
             if f"{m}_s{s}" in raw]
        d = [raw[f"{m}_s{s}"].get("detect_auroc") for s in range(seeds)
             if f"{m}_s{s}" in raw]
        d = [x for x in d if x is not None and not np.isnan(x)]
        if v:
            msg = f"  {m:10s} AUC {np.mean(v):.4f} +- {np.std(v):.4f}"
            if d:
                msg += f"   detect {np.mean(d):.3f}"
            print(msg + f"  (n={len(v)})")


if __name__ == "__main__":
    main()
