"""AES-128 key-rank experiment: grid width W vs bound tightness, estimate, time.
Run: python3 experiment_aes.py  -> ../results/aes_rank.csv, ../results/aes_rank.png"""
import csv, math, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rrdp_rank import simulate_template_attack, quantize, key_cost, rank_estimate

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
costs, key = simulate_template_attack(n_traces=6, sigma=1.0, seed=7)
T = int(key_cost(quantize(costs), key)[0])

rows = []
for W in [32, 64, 128, 256, 512, 1024, 4096, 16384]:
    D = max(1, T // W)
    for mode in ("floor", "stoch"):
        t = time.perf_counter()
        r = rank_estimate(costs, key, D, eps=0.05, delta=0.01, mode=mode, seed=1)
        dt = time.perf_counter() - t
        lg = lambda x: math.log2(x) if x else float("nan")
        rows.append(dict(W=W, mode=mode, log2_lower=lg(r["lower"]), log2_upper=lg(r["upper"]),
                         log2_estimate=lg(r["estimate"]) if r["estimate"] else float("nan"),
                         p_lo=r["p_lo"], samples=r["n"], seconds=round(dt, 3)))
        print(rows[-1])

with open(os.path.join(OUT, "aes_rank.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

fl = [x for x in rows if x["mode"] == "floor"]
Ws = [x["W"] for x in fl]
plt.figure(figsize=(6, 3.6))
plt.fill_between(Ws, [x["log2_lower"] for x in fl], [x["log2_upper"] for x in fl],
                 alpha=.25, label="certified sandwich (DP only)")
for mode, mk in (("floor", "o-"), ("stoch", "s--")):
    pts = [x for x in rows if x["mode"] == mode and not math.isnan(x["log2_estimate"])]
    plt.plot([x["W"] for x in pts], [x["log2_estimate"] for x in pts], mk,
             label=f"RRDP estimate ({mode})")
plt.xscale("log", base=2); plt.xlabel("DP grid width W (budget cells)")
plt.ylabel("log2 key rank"); plt.title("AES-128 key rank vs DP resolution")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(OUT, "aes_rank.png"), dpi=160)
print("saved results")
