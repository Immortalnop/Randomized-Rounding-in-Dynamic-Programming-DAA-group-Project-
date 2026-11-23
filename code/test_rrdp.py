"""Tests for RRDP-Rank against brute force. Run: python3 test_rrdp.py"""
import itertools, math
import numpy as np
from rrdp_rank import (rank_estimate, dp_tables, count, sample, round_costs,
                       key_cost, dyer_grid)

rng = np.random.default_rng(2026)


def brute(c, key):
    m, K = c.shape
    grids = np.meshgrid(*[c[j] for j in range(m)], indexing="ij")
    tot = sum(grids).ravel()
    return int((tot <= key_cost(c, key)[0]).sum()), tot


def rand_instance(m, K):
    c = rng.integers(0, 10_000, size=(m, K)).astype(np.int64)
    return c, rng.integers(0, K, m)


def test_dp_exact():
    """Lemma 2: DP count == brute-force count of rounded instance."""
    for _ in range(30):
        c, _ = rand_instance(3, 20)
        D = int(rng.integers(1, 500))
        r = round_costs(c, D, "floor")
        t = int(rng.integers(0, r.sum(axis=1).max()))
        F = dp_tables(r, t)
        bf = (sum(np.meshgrid(*r, indexing="ij")).ravel() <= t).sum()
        assert count(F, t) == bf
    print("ok  DP count exact (Lemma 2)")


def test_sandwich():
    """Lemma 1/3: L <= R <= U, deterministic, both rounding modes."""
    n = 0
    for _ in range(200):
        c, key = rand_instance(4, 12)
        R, _ = brute(c, key)
        for mode in ("floor", "stoch"):
            D = int(rng.integers(1, 3000))
            res = rank_estimate(c, key, D, mode=mode, integer_costs=True,
                                n_max=0, seed=int(rng.integers(1 << 30)))
            assert res["lower"] <= R <= res["upper"], (res, R)
            n += 1
    print(f"ok  certified sandwich L<=R<=U held in {n}/{n} cases")


def test_sampler_uniform():
    """Lemma 4: back-tracking sampler is uniform on S' (chi-square)."""
    c, _ = rand_instance(2, 8)
    r = round_costs(c, 700, "floor")
    t0 = int(np.median(r.sum(axis=0)) + r.max() // 2)
    F = dp_tables(r, t0)
    S = [(a, b) for a, b in itertools.product(range(8), repeat=2) if r[0, a] + r[1, b] <= t0]
    assert count(F, t0) == len(S)
    N = 200_000
    ks = sample(F, r, t0, N, rng)
    idx = {s: i for i, s in enumerate(S)}
    obs = np.zeros(len(S))
    for a, b in ks:
        obs[idx[(a, b)]] += 1          # KeyError => sample outside S'
    exp = N / len(S)
    chi2 = ((obs - exp) ** 2 / exp).sum()
    dof = len(S) - 1
    z = (chi2 - dof) / math.sqrt(2 * dof)
    assert abs(z) < 4, (chi2, dof)
    print(f"ok  sampler uniform on |S'|={len(S)} (chi2={chi2:.1f}, dof={dof}, z={z:+.2f})")


def test_estimator_floor():
    """Theorem 1: unbiased; |R_hat/R - 1| <= eps w.p. >= 1 - delta."""
    c, key = rand_instance(4, 16)
    R, _ = brute(c, key)
    D, eps, delta, trials = 400, 0.1, 0.1, 300
    est = np.array([rank_estimate(c, key, D, eps=eps, delta=delta, integer_costs=True,
                                  seed=s)["estimate"] for s in range(trials)])
    fail = np.mean(np.abs(est / R - 1) > eps)
    se = est.std(ddof=1) / math.sqrt(trials)
    assert fail <= delta, fail
    assert abs(est.mean() - R) < 4 * se, (est.mean(), R, se)
    print(f"ok  floor estimator: R={R}, mean={est.mean():.1f} (+-{se:.1f}), "
          f"fail rate {fail:.3f} <= delta={delta}")


def test_stoch_rounding():
    """Lemma 6: E[D r~] = c (unbiased) and Hoeffding tail for a fixed key."""
    m, K, D, eta = 16, 4, 1000, 0.05
    c = rng.integers(0, 10**6, size=(m, K)).astype(np.int64)
    key = rng.integers(0, K, m)
    reps = 20000
    dev = np.empty(reps); one = np.empty(reps)
    for i in range(reps):
        r = round_costs(c, D, "stoch", rng)
        one[i] = D * r[0, 0] - c[0, 0]
        dev[i] = D * key_cost(r, key)[0] - key_cost(c, key)[0]
    tau = D * math.sqrt(m * math.log(1 / eta) / 2)
    assert abs(one.mean()) < 4 * one.std() / math.sqrt(reps)
    tail = np.mean(dev > tau)
    assert tail <= eta, tail
    print(f"ok  stochastic rounding unbiased (mean err {one.mean():+.1f}, D={D}); "
          f"P[dev>tau]={tail:.4f} <= eta={eta}; |dev|max={np.abs(dev).max():.0f} "
          f"vs worst case mD={m*D}")


def test_estimator_stoch():
    """Theorem 2: (1-eps)(1-eta/delta1) R <= R_hat <= (1+eps) R w.p. >= 1-delta1-delta2."""
    c, key = rand_instance(4, 16)
    R, _ = brute(c, key)
    D, eps, d2, eta, d1, trials = 400, 0.1, 0.1, 0.01, 0.1, 300
    est = np.array([rank_estimate(c, key, D, eps=eps, delta=d2, mode="stoch", eta=eta,
                                  integer_costs=True, seed=s)["estimate"]
                    for s in range(trials)])
    lo, hi = (1 - eps) * (1 - eta / d1) * R, (1 + eps) * R
    fail = np.mean((est < lo) | (est > hi))
    assert fail <= d1 + d2, fail
    print(f"ok  stoch estimator: R={R}, mean={est.mean():.1f}, fail rate {fail:.3f} "
          f"<= {d1 + d2}")


def test_dyer_bound():
    """Lemma 5: with D <= T/(m(m-1)), |S(t*)| <= (m(K-1)+1) R. Includes skewed costs."""
    worst, n = 0.0, 0
    for it in range(300):
        m, K = int(rng.integers(2, 5)), int(rng.integers(2, 12))
        c = rng.integers(0, 10_000, size=(m, K)).astype(np.int64)
        if it % 2:
            c = (c.astype(float) ** 3 / 1e8).astype(np.int64)   # heavy-tailed costs
        c -= c.min(axis=1, keepdims=True)
        key = rng.integers(0, K, m)
        R, _ = brute(c, key)
        T = int(key_cost(c, key)[0])
        D = dyer_grid(T, m)
        res = rank_estimate(c, key, D, integer_costs=True, n_max=0)
        ratio = res["proposal"] / R
        assert ratio <= m * (K - 1) + 1, (ratio, m, K)
        worst = max(worst, ratio / (m * (K - 1) + 1)); n += 1
    print(f"ok  Dyer-type bound |S'|<=(m(K-1)+1)R held in {n}/{n} (max used {worst:.2f} of bound)")


if __name__ == "__main__":
    test_dyer_bound()
    test_dp_exact()
    test_sandwich()
    test_sampler_uniform()
    test_estimator_floor()
    test_stoch_rounding()
    test_estimator_stoch()
    print("ALL TESTS PASSED")
