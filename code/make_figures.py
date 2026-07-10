"""Paper figures.

Fig 1 (Figures/polytope_slice.pdf): exact 2-D slice of the dual domains for
n+ = n- = 2, Cbar = 1, at one-sided mass Lambda = 2. Coordinates
(x, y) = (lambda^+_1, lambda^-_1) with lambda^+_2 = 2 - x, lambda^-_2 = 2 - y.
B-slice is the square [0,2]^2; the Gale condition L+(1)+L-(1) <= Lambda + 1
becomes |x-1| + |y-1| <= 1: T is the inscribed diamond. FW vertices
(Cbar * S e_s with integer counts) in the slice are the diamond's corners and
center; ex:strict is the corner (2,2).

Fig 2 (Figures/auc_vs_iteration.pdf): test AUC by reweighting iteration from
results/ablation_raw.json (mean +- s.e. over 15 folds).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIGDIR = os.path.join(ROOT, "Figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9,
                     "axes.labelsize": 9, "legend.fontsize": 8})

# ------------------------------------------------------------------- Fig 1
fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.add_patch(plt.Rectangle((0, 0), 2, 2, facecolor="#d7e3f4",
                           edgecolor="#3b6ea5", linewidth=1.2))
diamond = np.array([[1, 0], [2, 1], [1, 2], [0, 1], [1, 0]])
ax.add_patch(plt.Polygon(diamond[:-1], facecolor="#f6d7b0",
                         edgecolor="#c05f00", linewidth=1.2))
verts = np.array([[1, 0], [2, 1], [1, 2], [0, 1], [1, 1]])
ax.plot(verts[:, 0], verts[:, 1], "o", color="#c05f00", ms=5, zorder=5)
ax.plot([2], [2], "x", color="#b00020", ms=9, mew=2.2, zorder=6)
ax.annotate(r"$\mathcal{B}$ (box $\cap$ hyperplane)", (0.08, 1.86),
            color="#3b6ea5")
ax.annotate(r"$\mathcal{T}$ (transportation polytope)", (1.0, 0.98),
            color="#c05f00", ha="center")
ax.annotate("Example 1:\n" r"$\lambda\in\mathcal{B}\setminus\mathcal{T}$",
            (1.97, 1.97), color="#b00020", ha="right", va="top", fontsize=8)
ax.annotate(r"$\bar{C}\,\mathcal{S}e_s$ (FW vertices)", (1.52, 0.30),
            color="#c05f00", fontsize=8)
ax.set_xlabel(r"$\lambda^+_1$")
ax.set_ylabel(r"$\lambda^-_1$")
ax.set_xlim(-0.15, 2.15)
ax.set_ylim(-0.15, 2.15)
ax.set_aspect("equal")
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "polytope_slice.pdf"))
plt.close(fig)
print("wrote Figures/polytope_slice.pdf")

# ------------------------------------------------------------------- Fig 2
with open(os.path.join(ROOT, "results", "ablation_raw.json")) as f:
    abl = json.load(f)

panels = [("synthetic_1to50", "fw_lr", "logistic, synthetic (1:36)"),
          ("satimage_4", "fw_svm", "hinge, satimage (1:9)")]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharex=True)
for ax, (dname, method, title) in zip(axes, panels):
    c = np.array(abl[dname][method])          # folds x iterations
    m, se = c.mean(0), c.std(0) / np.sqrt(len(c))
    b = np.maximum.accumulate(c, axis=1)      # best-so-far per fold
    bm, bse = b.mean(0), b.std(0) / np.sqrt(len(b))
    t = np.arange(len(m))
    ax.axhline(m[0], color="#888", lw=0.9, ls="--",
               label="balanced ($t=0$)")
    ax.plot(t, m, "-o", ms=2.6, lw=0.9, color="#c05f00", alpha=0.6,
            label="raw iterate")
    ax.plot(t, bm, "-s", ms=3.2, lw=1.6, color="#7a3b00",
            label="best iterate so far")
    ax.fill_between(t, bm - bse, bm + bse, color="#7a3b00", alpha=0.2, lw=0)
    ax.set_title(title)
    ax.set_xlabel("reweighting iteration $t$")
ax.legend(loc="lower right")
axes[0].set_ylabel("test AUC")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "auc_vs_iteration.pdf"))
print("wrote Figures/auc_vs_iteration.pdf")
