# Project Presentation Script
## Team Members: Aarush Mohan Mathur, Ojasw Kant, Breeti Bandyopadhyay, Ishaan Prakash, Shaurya Ajit Singh

---

### Aarush Mohan Mathur (2410110391)

**Topic: The Problem - Why Key Rank Matters**

"The problem we address is estimating the *key rank* after a side-channel attack on cryptographic devices like AES. When an attacker collects power traces or electromagnetic emissions, they can assign scores to each possible key byte. The key rank tells us: how many keys are still viable given these scores?

Formally, for m subkeys with K candidates each, we define the rank R as the count of keys whose total cost is below the true key's cost. This is a #P-hard problem - for AES-128 with 16 bytes and 256 candidates, the search space is 2^128, making exhaustive search impossible.

Our goal is to compute certified bounds L ≤ R ≤ U and an unbiased (1±ε) estimate in polynomial time, independent of the score magnitudes. This gives cryptographic evaluators a rigorous measure of residual security after an attack."

---

### Ojasw Kant (2410110226)

**Topic: The Core Algorithm - RRDP-Rank**

"The RRDP-Rank algorithm combines three ideas: rounding, dynamic programming, and randomization.

First, we *round* costs to a grid of width D. This shrinks the DP by factor D, but introduces bias. We implement two modes: 'floor' which always rounds down, and 'stochastic' which uses shared random shifts for unbiased rounding.

Second, we *count* exactly using a suffix DP. The table F[j][t] counts how many suffixes from position j have total rounded cost ≤ t. This gives deterministic certified bounds: for floor mode, N(t*-m) ≤ R ≤ N(t*) where t* = ⌊T/D⌋.

Third, we *correct* the bias using rejection sampling. We sample uniformly from the rounded superset S' = {k : ρ(k) ≤ t0} by walking back through the DP tables, then accept keys whose true cost ≤ T. The estimator R̂ = |S'| · (accepted/n) is unbiased."

---

### Breeti Bandyopadhyay (2410110105)

**Topic: Theoretical Guarantees - Lemmas and Theorems**

"Our work proves several theoretical results that form the foundation of RRDP-Rank:

*Lemma 1/2*: The per-entry rounding error for floor mode is 0 ≤ c - D·r ≤ D-1. For stochastic rounding, the error is bounded by ±(D-1) with the expectation exactly matching the original cost.

*Lemma 3*: The DP gives a certified sandwich L ≤ R ≤ U deterministically. For floor mode, L = N(t*-m) and U = N(t*). For stochastic mode, the bounds use a slightly wider range to account for the random shifts.

*Lemma 4*: The back-tracking sampler returns each key in S' with exactly 1/|S'| probability. We prove this by showing the telescoping product of conditional probabilities equals 1/N(t0).

*Theorem 1*: The floor estimator is unbiased with (1±ε) accuracy in probability ≥ 1-δ. The sample complexity is n ≥ 3·ln(2/δ)/(ε²·p_lo), where p_lo is a certified lower bound on the acceptance probability."

---

### Ishaan Prakash (2410110429)

**Topic: The Dyer-Type Bound and FPRAS**

"The key to making RRDP-Rank an FPRAS is the Dyer-type combinatorial bound, Lemma 5. It says: when D ≤ T/(m(m-1)), the proposal set size satisfies |S(t*)| ≤ (m(K-1)+1)·R.

The proof uses an injection argument: for any key k in S(t*) but not in A (the true accepted set), we can map it to a key in A by replacing its highest-cost coordinate with a zero-cost candidate. Each key in A has at most m(K-1) preimages, giving the bound.

With this guarantee, p_lo ≥ 1/(m(K-1)+1), so the sample complexity becomes polynomial in m and K. Combined with the DP running in O(mKW) time where W = O(m²), we get overall time O(m³K + m²K²ε⁻²logδ⁻¹), which is fully polynomial in the input size."

---

### Shaurya Ajit Singh (2410110318)

**Topic: Stochastic Rounding and Practical Results**

"Stochastic rounding with shared shifts provides a significant practical advantage. While floor rounding accumulates deterministic bias of Θ(mD), stochastic rounding has error concentration O(D√(m log(1/η))) with high probability. The price is a one-sided bias of at most ηR in the estimator.

Our experimental results confirm this: at m=128, stochastic rounding achieves 3.48× fewer samples than floor mode to reach the same accuracy. The acceptance probability stays around 52% vs 15% for floor.

We validate all theoretical claims against brute-force enumeration on small instances. The tests check: DP count exactness (Lemma 2), certified bounds (Lemmas 1/3), uniform sampling (Lemma 4), Dyer bound (Lemma 5), floor estimator properties (Theorem 1), and stochastic rounding properties (Theorem 2).

All tests pass: 400/400 cases for bounds, χ² test z=0.38 for uniformity, and zero failure rates for estimators within theoretical guarantees."

---

## Notes for presenters:
- Each section is approximately 200-250 words
- Focus on explaining the *why* and *how*, not just stating results
- Emphasize what makes your contribution unique and non-trivial
- Use the whiteboard/diagrams to illustrate key concepts
- Prepare to answer questions about the proofs and implementation details
