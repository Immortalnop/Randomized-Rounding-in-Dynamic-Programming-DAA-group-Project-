# Problem 9 — Randomized Rounding in Dynamic Programming
**Round, Count, Correct: an FPRAS for side-channel key rank** · Aarush Mohan Mathur (2410110391)

## Contents
| Path | What |
|---|---|
| `report/report.pdf` | Full report (12 pp.) |
| `report/main.tex`, `report/sections/*.tex`, `refs.bib` | LaTeX source (Overleaf-ready) |
| `report/sections/01_problem_statement.tex` | Problem statement, background, #P-hardness |
| `report/sections/02_algorithm.tex` | RRDP-Rank algorithm |
| `report/sections/03_proof.tex` | Proof of correctness (Lemmas 1–6, Theorems 1–2, Corollary 1) |
| `report/sections/04_complexity.tex` | Complexity analysis |
| `report/sections/05_conclusion.tex` | Experiments and conclusion |
| `report/sections/06_gaps_future.tex` | Gaps and future directions |
| `code/rrdp_rank.py` | Implementation |
| `code/test_rrdp.py` | Brute-force verification of every lemma/theorem |
| `code/experiment_aes.py`, `code/experiment_scaling_m.py` | Experiments |
| `results/` | CSVs and figure |

## Main results
1. Rounding plus DP gives **certified bounds** L ≤ rank ≤ U.
2. DP-guided rejection sampling gives an **unbiased (1±ε) estimate**.
3. A Dyer-type bound turns this into an **FPRAS**, O(m³K + m²K²ε⁻²log δ⁻¹), for a #P-hard problem.
4. **Stochastic rounding** shrinks the error band from Θ(mD) to O(D√(m log 1/η)), giving 3.5× fewer samples at m=128.

## Run
```
cd code
python3 test_rrdp.py            # ~2 s, all tests pass
python3 experiment_aes.py       # ~15 s
python3 experiment_scaling_m.py # ~30 s
```
Requires Python 3, NumPy, Matplotlib. To build the PDF: `cd report && latexmk -pdf main.tex`.
