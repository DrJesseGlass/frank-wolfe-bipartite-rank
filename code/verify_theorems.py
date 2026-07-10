"""Numerical verification of the claims in dual_coefficient_mappings.tex.

Checks, on tiny random problems solved to high accuracy with scipy:
  1. rem:margin2        -- rescaling identity and two-hinge domination
  2. lem:holistic_surjection -- layer-cake construction
  3. eq:margin_identity -- Delta^T alpha = 1^T S alpha
  4. thm:equivalence    -- same w, equal objectives, image inside B
  5. strong duality     -- pairwise and balanced primal/dual gaps ~ 0
  6. thm:image          -- Gale prefix test == transportation-LP feasibility;
                           ex:strict counterexample; vertex hull inside T
  7. cor:sandwich       -- D(bal*) <= D(pair*) <= D(lam) for lam in T
  8. cor:fw_bridge      -- FW LMO over T = violated-pair counts; w=0 gives
                           the balanced weights
  9. rem:relaxation     -- trust-region box [0, Cbar*omega] not inside T

Exit code 0 iff every check passes.
"""

import sys
import numpy as np
from scipy.optimize import minimize, linprog, minimize_scalar

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------- problem gen

class Problem:
    def __init__(self, n_pos, n_neg, d, C, rng):
        self.n_pos, self.n_neg, self.d, self.C = n_pos, n_neg, d, C
        self.N = n_pos + n_neg
        self.Cbar = C / self.N ** 2
        mu = rng.normal(size=d)
        mu *= 1.0 / np.linalg.norm(mu)
        self.Xp = rng.normal(size=(n_pos, d)) + mu
        self.Xn = rng.normal(size=(n_neg, d)) - mu
        self.X = np.vstack([self.Xp, self.Xn])          # positives first
        self.t = np.concatenate([np.ones(n_pos), -np.ones(n_neg)])
        # pair difference vectors z_(i,j) = x_i - x_j, row-major over (i, j)
        self.Z = (self.Xp[:, None, :] - self.Xn[None, :, :]).reshape(-1, d)
        self.Qt = self.Z @ self.Z.T                     # pairwise Gram
        self.Q = np.outer(self.t, self.t) * (self.X @ self.X.T)
        self.caps = np.concatenate([
            np.full(n_pos, self.Cbar * n_neg),
            np.full(n_neg, self.Cbar * n_pos)])

    def phi(self, rho):
        R = rho.reshape(self.n_pos, self.n_neg)
        return np.concatenate([R.sum(axis=1), R.sum(axis=0)])

    def w_from_rho(self, rho):
        return self.Z.T @ rho

    def w_from_lam(self, lam):
        return self.X.T @ (self.t * lam)

    def D(self, lam):
        return 0.5 * lam @ self.Q @ lam - lam.sum()

    def pair_dual_obj(self, rho):
        return 0.5 * rho @ self.Qt @ rho - 2.0 * rho.sum()

    def pair_primal(self, w):
        return 0.5 * w @ w + self.Cbar * np.maximum(0.0, 2.0 - self.Z @ w).sum()

    def bal_primal(self, w, b):
        sp = np.maximum(0.0, 1.0 - (self.Xp @ w + b)).sum()
        sn = np.maximum(0.0, 1.0 + (self.Xn @ w + b)).sum()
        return 0.5 * w @ w + self.Cbar * (self.n_neg * sp + self.n_pos * sn)


# ------------------------------------------------------------------- solvers

def solve_pair_dual(p):
    """min 0.5 r'Qt r - 2 1'r  s.t. 0 <= r <= Cbar  (L-BFGS-B, analytic grad)."""
    n = p.n_pos * p.n_neg
    fun = lambda r: (p.pair_dual_obj(r), p.Qt @ r - 2.0)
    best = None
    for x0 in (np.zeros(n), np.full(n, p.Cbar / 2)):
        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       bounds=[(0.0, p.Cbar)] * n,
                       options=dict(maxiter=20000, ftol=1e-16, gtol=1e-12))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun


def solve_bal_dual(p):
    """min D(lam) s.t. 0 <= lam <= caps, sum(t*lam) = 0  (SLSQP)."""
    fun = lambda l: (p.D(l), p.Q @ l - 1.0)
    cons = [dict(type="eq", fun=lambda l: p.t @ l, jac=lambda l: p.t)]
    best = None
    for frac in (0.0, 0.5, 1.0):
        res = minimize(fun, frac * p.caps, jac=True, method="SLSQP",
                       bounds=[(0.0, c) for c in p.caps], constraints=cons,
                       options=dict(maxiter=5000, ftol=1e-14))
        if res.success and (best is None or res.fun < best.fun):
            best = res
    return best.x, best.fun


def gale_ok(p, lam, tol=1e-9):
    """thm:image(ii): nonneg, equal sums, L+(a) + L-(b) <= Lam + Cbar*a*b."""
    lp, ln = lam[:p.n_pos], lam[p.n_pos:]
    if lam.min() < -tol or abs(lp.sum() - ln.sum()) > tol:
        return False
    Lam = lp.sum()
    cp = np.cumsum(np.sort(lp)[::-1])
    cn = np.cumsum(np.sort(ln)[::-1])
    ab = np.arange(1, p.n_pos + 1)[:, None] * np.arange(1, p.n_neg + 1)[None, :]
    return bool(np.all(cp[:, None] + cn[None, :] <= Lam + p.Cbar * ab + tol))


def lp_ok(p, lam, tol=1e-8):
    """Transportation feasibility via max-flow LP: max total rho with
    row sums <= lam+, col sums <= lam-, 0 <= rho <= Cbar; feasible iff
    the maximum equals Lam (and the one-sided sums agree)."""
    lp_, ln_ = lam[:p.n_pos], lam[p.n_pos:]
    if lam.min() < -tol or abs(lp_.sum() - ln_.sum()) > tol:
        return False
    n = p.n_pos * p.n_neg
    A = np.zeros((p.n_pos + p.n_neg, n))
    for i in range(p.n_pos):
        A[i, i * p.n_neg:(i + 1) * p.n_neg] = 1.0
    for j in range(p.n_neg):
        A[p.n_pos + j, j::p.n_neg] = 1.0
    res = linprog(-np.ones(n), A_ub=A, b_ub=lam, bounds=(0.0, p.Cbar),
                  method="highs")
    return res.status == 0 and (lp_.sum() + res.fun) < tol  # -res.fun = max flow


def random_point_in_B(p, rng, sparse=False):
    """Random lam in B (box + equal one-sided sums)."""
    if sparse:  # concentrate mass to stress the Gale conditions
        lp_ = p.caps[:p.n_pos] * rng.dirichlet(np.full(p.n_pos, 0.15))
        ln_ = p.caps[p.n_pos:] * rng.dirichlet(np.full(p.n_neg, 0.15))
        lp_, ln_ = np.minimum(lp_, p.caps[:p.n_pos]), np.minimum(ln_, p.caps[p.n_pos:])
    else:
        lp_ = rng.uniform(0, p.caps[:p.n_pos])
        ln_ = rng.uniform(0, p.caps[p.n_pos:])
    s_p, s_n = lp_.sum(), ln_.sum()
    if s_p > s_n:
        lp_ *= s_n / s_p
    else:
        ln_ *= s_p / s_n
    return np.concatenate([lp_, ln_])


# --------------------------------------------------------------- test groups

def test_margin2(p, rng):
    for _ in range(20):
        w = rng.normal(size=p.d) * rng.uniform(0.1, 5.0)
        lhs = p.pair_primal(w)
        v = w / 2.0
        rhs = 4.0 * (0.5 * v @ v + (p.Cbar / 2.0)
                     * np.maximum(0.0, 1.0 - p.Z @ v).sum())
        if not np.isclose(lhs, rhs, rtol=1e-12):
            return check("rem:margin2 rescaling identity", False,
                         f"{lhs} != {rhs}")
        two_hinge = (np.maximum(0.0, 1.0 - p.Xp @ w)[:, None]
                     + np.maximum(0.0, 1.0 + p.Xn @ w)[None, :]).ravel()
        pair_hinge = np.maximum(0.0, 2.0 - p.Z @ w)
        if not np.all(two_hinge >= pair_hinge - 1e-12):
            return check("rem:margin2 two-hinge domination", False)
    check("rem:margin2 (rescaling + two-hinge domination)", True)


def test_layer_cake(p, rng):
    C = p.C
    for _ in range(20):
        lam = rng.uniform(0, C, size=p.N)
        vals = np.sort(np.unique(lam[lam > 0]))
        alpha, recon, total = [], np.zeros(p.N), 0.0
        prev = 0.0
        for v in vals:
            s = (lam >= v - 1e-15).astype(float)
            recon += (v - prev) * s
            total += v - prev
            alpha.append(v - prev)
            prev = v
        ok = (np.allclose(recon, lam, atol=1e-12) and total <= C + 1e-12
              and len(alpha) <= p.N)
        if not ok:
            return check("lem:holistic_surjection layer-cake", False)
    check("lem:holistic_surjection layer-cake", True)


def test_margin_identity(p, rng):
    n_pairs = p.n_pos * p.n_neg
    for _ in range(20):
        m = rng.integers(1, 6)
        subsets = [rng.random(n_pairs) < rng.uniform(0.1, 0.9)
                   for _ in range(m)]
        a = rng.dirichlet(np.ones(m)) * p.C
        lam = sum(ai * p.phi(s.astype(float)) for ai, s in zip(a, subsets))
        delta_a = sum(ai * 2.0 * s.sum() for ai, s in zip(a, subsets))
        if not np.isclose(delta_a, lam.sum(), rtol=1e-12):
            return check("eq:margin_identity", False)
    check("eq:margin_identity (Delta'a = 1'S a = 1'lam)", True)


def test_equivalence(p, rng):
    n = p.n_pos * p.n_neg
    for _ in range(50):
        rho = rng.uniform(0, p.Cbar, size=n)
        lam = p.phi(rho)
        if not np.allclose(p.w_from_rho(rho), p.w_from_lam(lam), atol=1e-12):
            return check("thm:equivalence (i) weight vectors", False)
        if not np.isclose(p.pair_dual_obj(rho), p.D(lam), rtol=1e-10):
            return check("thm:equivalence (ii) objectives", False)
        in_B = (lam.min() >= -1e-12 and np.all(lam <= p.caps + 1e-12)
                and abs(p.t @ lam) < 1e-10)
        if not in_B:
            return check("thm:equivalence (iii) image in B", False)
    check("thm:equivalence (i)-(iii) on random rho", True)


def test_strong_duality(p):
    rho_s, dval = solve_pair_dual(p)
    w = p.w_from_rho(rho_s)
    gap = abs(p.pair_primal(w) + dval) / max(1.0, abs(dval))
    check("pairwise strong duality (primal = -dual)", gap < 1e-6,
          f"rel gap {gap:.2e}")

    lam_s, bval = solve_bal_dual(p)
    wb = p.w_from_lam(lam_s)
    b = minimize_scalar(lambda b: p.bal_primal(wb, b),
                        bounds=(-50, 50), method="bounded",
                        options=dict(xatol=1e-12)).x
    gap_b = abs(p.bal_primal(wb, b) + bval) / max(1.0, abs(bval))
    check("balanced strong duality (primal = -dual)", gap_b < 1e-5,
          f"rel gap {gap_b:.2e}")
    return rho_s, dval, lam_s, bval


def test_image(p, rng):
    # every image point satisfies Gale
    ok = all(gale_ok(p, p.phi(rng.uniform(0, p.Cbar, p.n_pos * p.n_neg)))
             for _ in range(200))
    check("thm:image: Phi(rho) always satisfies Gale", ok)

    # Gale test <=> LP feasibility on random points of B
    agree, n_in, n_out = True, 0, 0
    for k in range(150):
        lam = random_point_in_B(p, rng, sparse=(k % 2 == 0))
        g, l = gale_ok(p, lam), lp_ok(p, lam)
        if g != l:
            agree = False
            check("thm:image(ii) Gale == LP feasibility", False,
                  f"disagreement at lam={lam}")
            break
        n_in += g
        n_out += (not g)
    if agree:
        check("thm:image(ii) Gale == LP feasibility", n_in > 0 and n_out > 0,
              f"{n_in} feasible / {n_out} infeasible samples agreed")

    # ex:strict for n+=n-=2 (build a dedicated 2x2 problem)
    rng2 = np.random.default_rng(0)
    p2 = Problem(2, 2, 2, p.C, rng2)
    lam_bad = np.array([2 * p2.Cbar, 0.0, 2 * p2.Cbar, 0.0])
    in_B = (np.all(lam_bad <= p2.caps + 1e-15) and abs(p2.t @ lam_bad) < 1e-15)
    check("ex:strict: point lies in B but fails Gale and LP",
          in_B and not gale_ok(p2, lam_bad) and not lp_ok(p2, lam_bad))

    # vertex description: random convex combos of Cbar*S e_s are in T
    n_pairs = p.n_pos * p.n_neg
    ok = True
    for _ in range(50):
        m = rng.integers(1, 5)
        ws = rng.dirichlet(np.ones(m))
        lam = np.zeros(p.N)
        for w_ in ws:
            s = (rng.random(n_pairs) < rng.uniform(0.2, 0.8)).astype(float)
            lam += w_ * p.phi(p.Cbar * s)
        if not (gale_ok(p, lam) and lp_ok(p, lam)):
            ok = False
            break
    check("thm:image(i): conv combos of Cbar*S e_s lie in T", ok)


def test_sandwich(p, rng, rho_s, dval, lam_s, bval):
    lam_pair = p.phi(rho_s)
    ok1 = bval <= dval + 1e-8 * max(1, abs(dval))
    check("cor:sandwich: D(bal*) <= D(pair*)", ok1,
          f"D_bal={bval:.8f}, D_pair={dval:.8f}")
    ok2 = all(dval <= p.D(p.phi(rng.uniform(0, p.Cbar, p.n_pos * p.n_neg)))
              + 1e-8 for _ in range(100))
    check("cor:sandwich: D(pair*) <= D(lam) for lam in T", ok2)
    if gale_ok(p, lam_s, tol=1e-7):
        check("cor:sandwich: bal* in T => equal optima",
              np.isclose(bval, dval, rtol=1e-6),
              "balanced solution landed in T")


def test_fw_bridge(p, rng):
    n_pairs = p.n_pos * p.n_neg
    for trial in range(20):
        w = np.zeros(p.d) if trial == 0 else rng.normal(size=p.d) * rng.uniform(0.2, 3)
        lam_cur = p.phi(rng.uniform(0, p.Cbar, n_pairs)) if trial else np.zeros(p.N)
        # rebuild w consistently from a current dual point
        if trial:
            w = p.w_from_lam(lam_cur)
        g = p.Q @ lam_cur - 1.0
        # LMO over T via rho-space LP
        cost = np.add.outer(g[:p.n_pos], g[p.n_pos:]).ravel()
        res = linprog(cost, bounds=(0.0, p.Cbar), method="highs")
        # predicted: rho = Cbar on violated pairs (2 - <w,z> > 0)
        margins = 2.0 - p.Z @ w
        if np.any(np.abs(margins) < 1e-9):
            continue  # skip ties
        pred = p.Cbar * (margins > 0).astype(float)
        if not np.allclose(res.x, pred, atol=1e-9):
            return check("cor:fw_bridge: LMO = violated-pair counts", False,
                         f"trial {trial}")
        if trial == 0:
            lam0 = p.phi(pred)
            expect = np.concatenate([np.full(p.n_pos, p.Cbar * p.n_neg),
                                     np.full(p.n_neg, p.Cbar * p.n_pos)])
            if not np.allclose(lam0, expect):
                return check("cor:fw_bridge: w=0 gives balanced weights", False)
            check("cor:fw_bridge: w=0 LMO = balanced weights (n-, n+)", True)
    check("cor:fw_bridge: LMO over T = violated-pair counts", True)


def test_relaxation(p, rng):
    # omega = counts at a random w with a nontrivial violated set
    for _ in range(50):
        w = rng.normal(size=p.d) * rng.uniform(0.5, 3)
        s = (2.0 - p.Z @ w > 0).astype(float)
        if 0 < s.sum() < s.size:
            break
    omega = p.phi(s)
    box_hi = p.Cbar * omega
    found_outside = False
    for _ in range(500):
        lam = rng.uniform(0, box_hi)
        if not (gale_ok(p, lam) and lp_ok(p, lam)):
            found_outside = True
            break
    check("rem:relaxation: box [0, Cbar*omega] not contained in T",
          found_outside)
    check("rem:relaxation: vertex Cbar*omega itself lies in T",
          gale_ok(p, box_hi) and lp_ok(p, box_hi))


# --------------------------------------------------------------------- main

def run(seed, n_pos, n_neg, d=3, C=2.0):
    print(f"\n=== seed={seed}, n+={n_pos}, n-={n_neg}, d={d}, C={C} ===")
    rng = np.random.default_rng(seed)
    p = Problem(n_pos, n_neg, d, C, rng)
    test_margin2(p, rng)
    test_layer_cake(p, rng)
    test_margin_identity(p, rng)
    test_equivalence(p, rng)
    rho_s, dval, lam_s, bval = test_strong_duality(p)
    test_image(p, rng)
    test_sandwich(p, rng, rho_s, dval, lam_s, bval)
    test_fw_bridge(p, rng)
    test_relaxation(p, rng)


if __name__ == "__main__":
    for seed in range(5):
        for (np_, nn_) in [(5, 7), (4, 4), (8, 3)]:
            run(seed, np_, nn_)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'=' * 60}\nTOTAL: {len(RESULTS)} checks, {n_fail} failures")
    sys.exit(1 if n_fail else 0)
