"""Floor vs stochastic rounding as the number of subkeys m grows (same grid width W).
Metric: proposal size |S'| and empirical acceptance |A cap S'|/|S'| (sampling cost ~ 1/acc).
Run: python3 experiment_scaling_m.py -> ../results/scaling_m.csv"""
import csv, math, os
import numpy as np
from rrdp_rank import quantize, key_cost, round_costs, dp_tables, count, sample

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(11)
ETA, NS = 0.01, 20000
rows = []
for m in [8, 16, 32, 64, 128]:
    K = 16
    raw = rng.exponential(1.0, size=(m, K)); raw[:, 0] = 0.0
    key = np.where(rng.random(m) < 0.7, 0, rng.integers(0, K, m))   # partially leaked key
    c = quantize(raw); T = int(key_cost(c, key)[0])
    W = 16 * m
    D = max(1, T // W)
    row = dict(m=m, W=W, tau_over_D=round(math.sqrt(m * math.log(1 / ETA) / 2), 2),
               mean_floor_bias_over_D=m / 2)
    for mode in ("floor", "stoch"):
        r = round_costs(c, D, mode, rng)
        t0 = T // D if mode == "floor" else min(
            (T + math.ceil(D * math.sqrt(m * math.log(1 / ETA) / 2))) // D,
            (T + m * (D - 1)) // D)
        F = dp_tables(r, t0)
        ks = sample(F, r, t0, NS, rng)
        acc = float((key_cost(c, ks) <= T).mean())
        row[f"log2_prop_{mode}"] = round(math.log2(count(F, t0)), 3)
        row[f"accept_{mode}"] = round(acc, 4)
    rows.append(row); print(row)
with open(os.path.join(OUT, "scaling_m.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
