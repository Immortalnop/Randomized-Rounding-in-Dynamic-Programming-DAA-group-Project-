"""
RRDP-Rank: Rounding + Dynamic Programming + Randomized correction
for side-channel key-rank estimation (a #multiple-choice-knapsack problem).

Problem
-------
m subkeys, K candidates each, integer costs c[j, v] >= 0 (e.g. scaled -log posterior).
True key k* has cost T = sum_j c[j, k*_j].
    rank R = #{ k in [K]^m : sum_j c[j, k_j] <= T }      (ties counted against us)

Algorithm (see report, Section 2)
---------------------------------
1. Round costs to a grid of width D   (mode 'floor'  : r = c // D,
                                        mode 'stoch'  : r = (c + U_j) // D, U_j ~ Unif{0..D-1})
2. Exact DP over integer budgets: F_j[t] = #{(k_j..k_{m-1}) : sum r <= t}
   -> certified sandwich  L <= R <= U.
3. Sample uniformly from proposal S' = {sum r <= t0} by DP back-tracking,
   accept iff true cost <= T. Estimate R_hat = |S'| * accepted / N.

All cost arithmetic is exact int64; counts are float64 (non-negative sums, relative
error <= ~(m*K)*2^-53, and exact while counts < 2^53).
"""
from __future__ import annotations

import math
import numpy as np

SCALE = 2 ** 20  # fixed-point resolution for real-valued scores (2^-20 nats)


def quantize(costs: np.ndarray) -> np.ndarray:
    """Real non-negative costs -> exact int64 fixed point. Rank is defined on these."""
    c = np.asarray(costs, dtype=np.float64)
    c = c - c.min(axis=1, keepdims=True)          # shift per subkey: rank unchanged
    return np.rint(c * SCALE).astype(np.int64)


def key_cost(c: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Exact integer cost of keys (shape (N, m) or (m,))."""
    keys = np.atleast_2d(keys)
    return c[np.arange(c.shape[0])[None, :], keys].sum(axis=1)


def round_costs(c: np.ndarray, D: int, mode: str, rng=None) -> np.ndarray:
    if mode == "floor":
        return c // D
    if mode == "stoch":
        U = rng.integers(0, D, size=(c.shape[0], 1))  # one shared shift per subkey
        return (c + U) // D
    raise ValueError(mode)


def dp_tables(r: np.ndarray, tmax: int) -> list[np.ndarray]:
    """F[j][t] = number of suffixes (k_j..k_{m-1}) with sum r <= t, t = 0..tmax."""
    m = r.shape[0]
    F = [None] * (m + 1)
    F[m] = np.ones(tmax + 1)
    for j in range(m - 1, -1, -1):
        bins, cnt = np.unique(r[j], return_counts=True)
        nxt, cur = F[j + 1], np.zeros(tmax + 1)
        for b, h in zip(bins, cnt):
            if b <= tmax:
                cur[b:] += h * nxt[: tmax + 1 - b]
        F[j] = cur
    return F


def count(F, t: int) -> float:
    return 0.0 if t < 0 else float(F[0][t])


def sample(F, r: np.ndarray, t0: int, N: int, rng, batch: int = 20000) -> np.ndarray:
    """N i.i.d. uniform samples from {k : sum_j r[j,k_j] <= t0} (requires F[0][t0] > 0)."""
    m, K = r.shape
    out = np.empty((N, m), dtype=np.int64)
    for s in range(0, N, batch):
        n = min(batch, N - s)
        t = np.full(n, t0, dtype=np.int64)
        for j in range(m):
            rem = t[:, None] - r[j][None, :]                 # (n, K) remaining budget
            w = np.where(rem >= 0, F[j + 1][np.clip(rem, 0, None)], 0.0)
            cdf = np.cumsum(w, axis=1)
            u = rng.random(n) * cdf[:, -1]
            v = (cdf < u[:, None]).sum(axis=1)
            v = np.minimum(v, K - 1)
            out[s:s + n, j] = v
            t -= r[j, v]
    return out


def dyer_grid(T: int, m: int) -> int:
    """Grid width for the a-priori bound |S'| <= (m(K-1)+1) R  (Lemma 5)."""
    return max(1, T // (m * (m - 1))) if m >= 2 else 1


def rank_estimate(costs, key, D: int, eps=0.05, delta=0.01, mode="floor",
                  eta=0.01, n_max=2_000_000, seed=None, integer_costs=False):
    """
    Returns dict with certified bounds [lower, upper], randomized estimate, diagnostics.
    mode='floor': Theorem 1 (unbiased, (1+-eps) w.p. >= 1-delta).
    mode='stoch': Theorem 2 (proposal threshold T + tau, tau = D*sqrt(m ln(1/eta)/2)).
    """
    rng = np.random.default_rng(seed)
    if integer_costs:
        c = np.asarray(costs, dtype=np.int64)
        c = c - c.min(axis=1, keepdims=True)                # rank-invariant shift
    else:
        c = quantize(costs)
    m, K = c.shape
    T = int(key_cost(c, np.asarray(key))[0])
    r = round_costs(c, D, mode, rng)

    t_low = T // D - m                                     # Lemma 1 / 3: certified-in set
    if mode == "floor":
        t_up = T // D                                      # A subset of {sum r <= T//D}
        t0 = t_up                                          # proposal = upper set
    else:
        t_up = (T + m * (D - 1)) // D                      # A subset of {sum r~ <= t_up}
        tau = math.ceil(D * math.sqrt(m * math.log(1 / eta) / 2))
        t0 = min((T + tau) // D, t_up)
    F = dp_tables(r, t_up)

    lower, upper, prop = max(count(F, t_low), 1.0), count(F, t_up), count(F, t0)
    if mode == "floor" and D == 1:
        lower = upper                                       # no rounding: DP is exact
    res = dict(T=T, D=D, mode=mode, width=t_up + 1, lower=lower, upper=upper,
               proposal=prop, estimate=None, p_lo=None, n=0, accepted=0)
    if prop == 0:
        return res
    p_lo = count(F, t_low) / prop
    if mode == "floor" and m >= 2 and D * m * (m - 1) <= T:
        p_lo = max(p_lo, 1.0 / (m * (K - 1) + 1))           # Lemma 5 (Dyer-type bound)
    res["p_lo"] = p_lo
    if p_lo <= 0:
        return res
    N = math.ceil(3 * math.log(2 / delta) / (eps ** 2 * p_lo))
    if N > n_max:
        return res                                          # fall back to sandwich only
    ks = sample(F, r, t0, N, rng)
    acc = int((key_cost(c, ks) <= T).sum())
    res.update(n=N, accepted=acc, estimate=prop * acc / N)
    return res


# ----------------------------------------------------------------- simulation helpers
SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0x16,0xbb],
    dtype=np.int64)
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)


def simulate_template_attack(n_traces: int, sigma: float, seed=0):
    """AES-128 first-round HW(Sbox(p^k)) leakage + Gaussian noise.
    Returns (costs[16,256] = -log posterior per byte, true key[16])."""
    rng = np.random.default_rng(seed)
    key = rng.integers(0, 256, 16)
    costs = np.empty((16, 256))
    v = np.arange(256)
    for j in range(16):
        p = rng.integers(0, 256, n_traces)
        leak = HW[SBOX[p ^ key[j]]] + rng.normal(0, sigma, n_traces)
        model = HW[SBOX[p[:, None] ^ v[None, :]]]          # (n_traces, 256)
        ll = -((leak[:, None] - model) ** 2).sum(axis=0) / (2 * sigma ** 2)
        ll -= ll.max()
        costs[j] = -(ll - np.log(np.exp(ll).sum()))          # -log posterior >= 0
    return costs, key
