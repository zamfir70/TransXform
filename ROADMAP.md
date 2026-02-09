# TransXform Hardening Roadmap

What can still go wrong when the architecture is sound, the data is good, and the spec is reasonable?

## Current Coverage

| Layer | What It Catches | Stress Test |
|-------|----------------|-------------|
| V1 hard invariants | Cosine collapse, gradient death, variance collapse, loss explosion, gradient spikes | ICFU, CRUX |
| V1 readiness gate | Cliff transitions (advancing into thresholds the model can't satisfy) | SemanticNormalizer Run 7 vs 8 |
| V1 abort | Unsalvageable architectures (max interventions exhausted) | CRUX |
| V2 Signal 1 | Unused capacity — high variance + low loss improvement | — |
| V2 Signal 2 | Missing structural signal — cosine/variance not moving | — |
| V2 Signal 3 | Loss-representation mismatch — loss improving but representations stagnating | FROG (early) |
| V2 Signal 4 | Dynamically unlearnable regime — loss plateau + pathological gradients | — |
| V2 Signal 5 | Shortcut learning — variance explosion/collapse while loss improves | FROG |
| V2 Signal 6 | Missing expected metrics — spec declares them but they never appear | — |
| V2.2 Signal 7 | Loss stagnation — loss plateau + healthy gradients | MIRE |
| V2.3 Signal 8 | Threshold drift — metric trending monotonically toward threshold | — |
| V2.3 Signal 9 | Metric instability — high CV oscillation on invariant metrics | — |
| V2.3 Signal 10 | Intervention futility — repeated failed interventions on same component | CRUX |
| V2.4 Signal 11 | Gradient domination — one component monopolizes optimizer updates | — |
| V2.4 Signal 12 | Metric anomaly — NaN/Inf sentinel for corrupted training state | — |
| V2.4 Signal 13 | Train/val divergence — overfitting detection via loss divergence | — |

---

## Tier 1 — COMPLETE (V2.3)

### Signal 8: ThresholdDrift

**The problem:** A metric creeps 0.3 -> 0.5 -> 0.7 -> 0.95 -> 0.979 over thousands of steps. Never crosses 0.98. TransXform fires at 0.981 — by then the model is already in trouble and the intervention (reinit, LR halve) is a shock to a fragile system. Recovery is hard because the model has been drifting for so long.

**The signal:** Trend extrapolation on invariant-associated metrics. If a metric has been monotonically trending toward its hard threshold over a sustained window, project when it will cross and warn early.

**Evidence:** "encoder.pairwise_cosine has risen from 0.45 to 0.92 over the last 8,000 steps. At current rate, it will cross the 0.98 hard threshold in approximately 1,200 steps."

**Why it matters:** Turns TransXform from reactive (fire when crossed) to predictive (warn before it happens). Early warning gives the training loop time to adjust LR, increase regularization, or checkpoint before the violation cascade.

**Config:**
- `drift_window_steps` — how many steps of history to compute trend over (default 1000)
- `drift_crossing_horizon` — warn if projected crossing is within N steps (default 2000)
- `drift_monotonic_pct` — what fraction of recent samples must be trending (default 0.8)

### Signal 9: MetricInstability

**The problem:** A metric bounces rapidly between 0.4 and 0.9 every 50 steps. The mean is 0.65 — well within bounds. But the oscillation itself is pathological: the model is in an unstable regime, and any perturbation could tip it permanently over the threshold. TransXform sees each individual sample as "fine" and never flags the pattern.

**The signal:** Coefficient of variation (std/mean) on each invariant metric over the diagnostic history window. High CV means the metric is unstable even if its mean is safe.

**Evidence:** "encoder.pairwise_cosine has coefficient of variation 0.42 over the last 2,000 steps (mean 0.65, std 0.27). This level of instability suggests the model may be oscillating between representational states."

**Why it matters:** Oscillating metrics are a precursor to permanent threshold crossing. A metric with CV 0.4 centered at 0.65 will eventually sample above 0.98 by chance alone. Catching this early lets the training loop stabilize (e.g., reduce LR) before the inevitable violation.

**Config:**
- `instability_cv_threshold` — CV above this triggers (default 0.3)
- Uses existing `history_window` from DiagnosticConfig

### Signal 10: InterventionFutility

**The problem:** TransXform reinitializes the encoder. Cosine drops from 0.99 to 0.4. Twenty steps later, it's back at 0.97. TransXform reinitializes again. Same pattern. After 8 reinitializations, it finally aborts at `max_hard_interventions`. But the writing was on the wall after the second failed recovery — each reinit recovered for fewer steps than the last.

**The signal:** Track post-intervention metric recovery duration. If the last N interventions on the same component all produced recovery lasting fewer than M steps, the interventions are futile and the architecture may need fundamental changes, not repeated reinitializations.

**Evidence:** "Last 3 reinitializations of encoder produced cosine recovery lasting 18, 12, and 8 steps respectively. Interventions on this component are showing diminishing returns."

**Why it matters:** Saves compute by recognizing futility early instead of burning through the intervention budget. CRUX took 12 interventions across 221 steps to decide the architecture was broken — intervention futility detection could have flagged this by intervention 3 or 4.

**Config:**
- `futility_lookback_interventions` — how many past interventions to examine (default 3)
- `futility_min_recovery_steps` — if all recent recoveries lasted fewer than this, fire (default 50)

---

## Tier 2 — Tractable but needs new metric conventions from the user

### Catastrophic Forgetting

**The problem:** In multi-task or curriculum training, the model learns task B but forgets task A. Total loss stays flat because gains on B offset losses on A. TransXform sees stable loss and healthy structural metrics.

**What it would take:** Per-domain/per-task loss metrics (e.g., `loss_domain_A`, `loss_domain_B`). A diagnostic signal watches for one rising while another falls. Requires the user to report domain-specific losses — not hard, but a new convention to document and enforce.

### Attention Entropy Collapse

**The problem:** All attention heads converge to uniform or identical distributions. Attention becomes a no-op, but FFN layers compensate, so loss/cosine/variance all look fine. The model works but is wasting capacity and is fragile to distribution shift.

**What it would take:** User reports `{component}.attention_entropy` as a metric. TransXform watches for it dropping below a threshold or all heads converging to similar values. Feasible but ties TransXform to a specific architectural assumption (attention-based models).

---

## Tier 3 — Fundamentally hard with current architecture

### Normalization Masking

**The problem:** LayerNorm/BatchNorm rescales degenerate representations to look healthy. Post-norm cosine is 0.3 (diverse!), but pre-norm cosine is 0.999 (collapsed). TransXform measures post-norm because that's what the user's `component_metrics()` returns.

**What it would take:** User restructures metric reporting to include pre-norm measurements. More of a documentation/guidance issue than a signal to build.

### Inter-Component Gradient Interference

**The problem:** One head's gradients systematically cancel another's. Both report healthy individual grad norms, but the combined update is destructive. Net effective learning is near zero on shared parameters.

**What it would take:** Cross-component gradient correlation, which requires tensor-level access TransXform doesn't have. Would need a framework-specific extension (e.g., `tch_backend` computing gradient cosine between component parameter groups).

---

## Implementation Status

- [x] **Signal 8: ThresholdDrift** — V2.3
- [x] **Signal 9: MetricInstability** — V2.3
- [x] **Signal 10: InterventionFutility** — V2.3
- [x] Stress test for Tier 1 signals — CRUX re-run validates Signal 9 + Signal 10
- [x] **Signal 11: GradientDomination** — V2.4, 141 tests passing
- [x] **Signal 12: MetricAnomaly** — V2.4, 141 tests passing
- [x] **Signal 13: TrainValDivergence** — V2.4, 141 tests passing
- [ ] Tier 2 as conventions stabilize
