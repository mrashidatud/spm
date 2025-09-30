# Decision-Guided Sensitivity Outputs (CART + ANOVA Feedback)

This document explains every CSV produced by `sensitivity_decision.py`: what it
captures, the meaning of each column, and what to look for when interpreting the
results.

---

## 1) `workflow_rowwise_sensitivities.csv`

**Purpose:** Per-configuration (rowwise) attributions and labels used by all
downstream summaries.

**Key columns**
- `nodes`: node count for the configuration.
- `total`: observed total makespan (lower is better).
- `*_stor` (e.g., `individuals_stor`, …): storage placement per stage.
- `critical_path`: serialized path such as
  `stage_in_X(...)->read_Y(...)->write_Z(...)->stage_out_W(...)`.
- `lam_read_<store>`, `lam_write_<store>`: execution time on the critical path
  attributed to reads/writes on `<store>`.
- `lam_in_total`, `lam_out_total`: total stage-in and stage-out time on the
  critical path.
- `exec_beegfs_share`  
  \= (lam_read_beegfs + lam_write_beegfs) / total
- `exec_local_share`  
  \= Σ\_{s ∈ local stores} (lam_read_s + lam_write_s) / total
- `movement_share`  
  \= (lam_in_total + lam_out_total) / total
- `region_provisional`: cross-fitted CART leaf label (`fK_leafID`).
- `region`: **merged** region ID after ANOVA/Games–Howell feedback.

**What to look for**
- Shares diagnose *where* time is spent along the critical path. High
  `movement_share` ⇒ data movement dominates; high `exec_*_share` ⇒ execution on
  that store dominates.
- Check that shares are plausible (non-zero where expected).

---

## 2) `regions_by_total.csv`

**Purpose:** Merged-region summary with strict wildcards and share statistics.

**Columns**
- `region`, `size`
- `mean_total`, `std_total`, `cv_total` (= std/mean)
- `q25_total`, `median_total`, `q75_total`, `min_total`, `max_total`
- One column per `*_stor` stage: **strict wildcard**  
  • single value in region ⇒ that value; else `"*"`
- `nodes`: comma-separated node levels covered
- `mean_exec_beegfs_share`, `cv_exec_beegfs_share`
- `mean_exec_local_share`, `cv_exec_local_share`
- `mean_movement_share`, `cv_movement_share`

**What to look for**
- Regions with low `mean_total` and small `cv_total` are strong candidates.
- Wildcards show how tightly a stage’s storage choice identifies the region.
- Share means reveal the dominant cost component within the region.

---

## 3) `regret_per_region.csv`

**Purpose:** Region-level regret versus the *global best* configuration.

**Columns**
- `region`, `mean_total`
- `regret_vs_global_best`  
  \= `mean_total` − min over all rows of `total`

**What to look for**
- Small regret indicates near-optimal regions under average behavior.

---

## 4) `regret_per_config.csv`

**Purpose:** Configuration-level regret.

**Columns**
- `region`, `nodes`, `total`, all `*_stor`
- `regret_vs_global_best`  
  \= `total` − global best `total`

**What to look for**
- Useful for ranking concrete assignments and validating the region summary.

---

## 5) `top_regions_store_effects.csv`

**Purpose:** Within each of the top-K (lowest-mean) regions, quantify residual
effects of *per-stage storage choice* on `total`.

**Columns**
- `region`, `column` (a `*_stor` stage), `value` (storage), `mean_total`
- `delta_from_best` = `mean_total` − best `mean_total` among values of `column`
  **within the same region**
- `count`

**What to look for**
- `delta_from_best` close to 0 ⇒ storage choice at that stage is immaterial in
  that region. Larger values point to stages worth standardizing.

---

## 6) `node_scaling_by_region.csv`

**Purpose:** Summarize scaling curves per region across node counts.

**Columns**
- `region`, `nodes`, `count`, `mean_total`, `std_total`
- `delta_from_prev` = mean\_total(n) − mean\_total(prev\_n)
- `rel_delta` = |`delta_from_prev`| / mean\_total(n)
- `efficiency` = T(n₀) / [ T(n)·(n/n₀) ], with n₀ as the smallest node level
- `is_nstar`: True if `|rel_delta| < nstar_eps` (diminishing returns threshold)

**What to look for**
- Where `is_nstar` flips to True is a sensible stop-scaling point.
- Efficiency should decrease as nodes increase; sudden drops may indicate
  bottlenecks or imbalance.

---

## 7) `synergy_by_region.csv`

**Purpose:** Two-way interaction (“synergy”) between pairs of `*_stor` stages.

**Columns**
- `region`, `colA`, `valA`, `colB`, `valB`
- `interaction` = mean\_AB − mean\_A − mean\_B + overall\_mean  
  where means are region-conditional:  
  • `mean_total_combo` (= mean\_AB),  
  • `mean_total_A`, `mean_total_B`, and `mean_total_overall`
- `count`

**Interpretation**
- Because `total` is *lower is better*,  
  • `interaction < 0` ⇒ *better-than-additive* combo (beneficial synergy),  
  • `interaction > 0` ⇒ worse-than-additive (antagonism).

**What to look for**
- Strongly negative interactions worth standardizing; positive ones to avoid.

---

## 8) `critical_path_prevalence_by_region.csv`

**Purpose:** Which critical paths dominate in each region.

**Columns**
- `region`, `region_size`
- `prevalent_critical_path`: most frequent path in the region
- `critical_path`, `count`, `fraction`
- Representative configuration for that path:  
  `config_nodes` and each `*_stor` column (or store aliases)

**What to look for**
- A single dominant path ⇒ robust behavior; multiple paths ⇒ mixed regimes.

---

## 9) `anova_stats.csv`

**Purpose:** Welch one-way ANOVA on the pooled cross-fit (held-out) labels.

**Columns**
- `Source` (always “Region (Welch)”)
- `ddof1`, `ddof2`, `F`, `pval`

**Interpretation**
- `pval < alpha_global` ⇒ at least one region mean differs.

---

## 10) `pairwise_stats.csv`

**Purpose:** Region-pair comparisons (Games–Howell if `pingouin` is present;
fallback is Welch t-tests + BH-FDR).

**Columns**
- `A`, `B`: region labels compared (provisional labels)
- `mean_A`, `mean_B`, `n_A`, `n_B`
- `pval_adj`: adjusted p (GH or BH-FDR)
- `ci_low`, `ci_high`: 95% CI for mean difference (GH only; NaN in fallback)
- `hedges_g`: standardized effect size (small ≈ 0.2, medium ≈ 0.5, large ≈ 0.8)
- `glass_delta_A_vs_B`, `glass_delta_B_vs_A`: Glass’s Δ using reference SD
- `sig`: True if (`pval_adj` < alpha\_pair) **and** (|g| ≥ g\_min)

**What to look for**
- Use `sig` and `hedges_g` jointly: material differences should be both
  statistically significant and practically non-negligible.

---

## 11) `region_merge_map.csv`

**Purpose:** Mapping from provisional (cross-fit) region labels to merged region
IDs used everywhere downstream.

**Columns**
- `region_provisional` → `region`

**What to look for**
- Merges indicate CART splits that do not survive statistical scrutiny.

---

# Mathematical Notes

- **Hedges’ g**  
  Let `x`, `y` be samples with means `m_x`, `m_y`, sample variances `s_x^2`,
  `s_y^2`, and sizes `n_x`, `n_y`. Pooled variance  
  `s_p^2 = ((n_x-1)s_x^2 + (n_y-1)s_y^2) / (n_x + n_y - 2)`  
  Cohen’s d `= (m_x - m_y) / s_p`. Small-sample correction  
  `J = 1 - 3/(4(n_x + n_y) - 9)`. Then `g = J * d`.

- **Glass’s Δ**  
  `Δ = (m_x - m_y) / s_ref`, where `s_ref` is the SD of the chosen reference
  group (asymmetric, useful when variances differ markedly).

- **Efficiency**  
  `E(n) = T(n0) / [ T(n) * (n / n0) ]`, where `T(n)` is mean makespan at `n`
  nodes and `n0` is the smallest node level in the region.

- **Regret**  
  Config: `total - min(total)`. Region: `mean_total - min(total)`.

---

# Practical Reading Order

1. `regions_by_total.csv` → find low-mean, low-CV regions.  
2. `regret_per_region.csv` and `regret_per_config.csv` → shortlist.  
3. `node_scaling_by_region.csv` → pick node counts; find n*.  
4. `top_regions_store_effects.csv` → learn which stage placements matter.  
5. `synergy_by_region.csv` → avoid antagonistic combos; favor synergistic ones.  
6. `critical_path_prevalence_by_region.csv` → confirm dominant behavior.  
7. `anova_stats.csv`, `pairwise_stats.csv`, `region_merge_map.csv` → audit the
   statistical defensibility of the regions you’re basing decisions on.
