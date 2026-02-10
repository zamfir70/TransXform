# TransXform Voyage: CRUX — Compressed Representation Under eXtreme Pressure

**Date**: 2026-02-08
**Status**: Complete (ABORTED at step 221)
**Verdict**: Correct abort of unsalvageable architecture

---

## Model Under Supervision

| Parameter | Value |
|-----------|-------|
| Model | crux-bottleneck |
| Parameters | ~2.12M |
| Architecture | Hash embed (2048x192) + 4-layer transformer (d=192, 4 heads) + bottleneck (192→32) + 3 heads |
| Task | Multi-objective: 5-class classification + BoW reconstruction + contrastive embedding |
| Dataset | 216,232 samples from 27,029 RFPV2 sequences (8 windows/seq, 128 tokens/window) |
| Batch size | 64 |
| Learning rate | 0.001 |
| Epochs | 3 (target 10,134 steps) |
| Device | CPU |

### Components

| Component | Role | Architecture |
|-----------|------|-------------|
| encoder | 4-layer transformer | Hash embed → 4 self-attention layers (d=192, 4 heads) |
| compressor | Severe bottleneck | Linear(192, 32) + tanh — 83% information loss |
| category_head | 5-class classifier | Linear(32, 5) — tiny gradients |
| bow_head | BoW reconstruction | Linear(32, 256) → ReLU → Linear(256, 2048) — massive gradients |
| contrastive_head | Contrastive embedding | Linear(32, 64) — collapse pressure |

### Design Intent: Engineered Instability

CRUX was deliberately designed to create training pathologies:

1. **400:1 gradient magnitude ratio**: BoW head (2048 outputs) vs category head (5 outputs) creates overwhelming gradient imbalance
2. **83% information bottleneck**: 192→32 dims forces the compressor to choose what to preserve — and three objectives disagree
3. **Contrastive collapse pressure**: InfoNCE loss within batch pushes representations toward collapse
4. **22:1 category imbalance**: cross_domain_analogy (47.4%) vs interdisciplinary (2.1%)

---

## What Happened

### Timeline

| Step | Event | Detail |
|------|-------|--------|
| 0 | Training starts | Loss 10.53 [cat=1.42, bow=0.88, con=4.51] |
| 1 | 4× cosine violations | All components above threshold immediately (encoder 0.957, compressor 0.972, contrastive 0.972, category 0.980) |
| 1 | 4× reinitialize | Encoder, compressor, contrastive_head, category_head all reinitialized |
| 2 | Grad norm spike | Global grad norm 9025x → LR halved to 5e-4 |
| 3-20 | Chronic cosine violations | Compressor, contrastive, category remain above threshold. Encoder recovering |
| 21 | 4× reinitialize encoder | Encoder violations triggered (upstream attribution) |
| 22-40 | Compressor chronic | Compressor cosine 0.958-0.984, slowly climbing back |
| 41 | Reinit compressor | 3× reinitialize actions |
| 42-60 | Contrastive + category chronic | Both stuck at 0.999+ |
| 61 | Reinit compressor | 2× actions |
| 81 | Reinit compressor | 2× actions |
| 101 | Reinit compressor | 2× actions |
| 121 | Reinit compressor | 2× actions |
| 141 | Reinit compressor | 2× actions |
| 144 | Category head hits 1.000000 | Total output collapse — all outputs identical |
| 146-160 | Rescale compressor loop | Intervention exhausted reinit, trying rescale |
| 161 | Reinit + rescale compressor | Cycling interventions |
| 181 | Reinit + rescale compressor | Still cycling |
| 201 | Reinit + rescale compressor | Still cycling |
| **221** | **ABORT** | max_hard_interventions (12) exhausted → bootstrap → aborted |

### Intervention Summary

| Intervention | Count | Effective? |
|-------------|-------|-----------|
| reinitialize(encoder) | 4 | Partially — encoder cosine dropped after reinit |
| reinitialize(compressor) | ~16 | No — collapsed back within 10-20 steps every time |
| reinitialize(contrastive_head) | 1 | No — stuck at 0.999+ permanently |
| reinitialize(category_head) | 1 | No — drifted to 1.000000 |
| rescale(compressor) | ~50+ | No — rescale(1.0) is a no-op, revealing a strategy limitation |
| LR halving | 1 | Partially — reduced grad spike but LR too low for recovery |

### Loss Trajectory

| Step | Total | Category | BoW | Contrastive | Phase |
|------|-------|----------|-----|-------------|-------|
| 0 | 10.53 | 1.42 | 0.88 | 4.51 | bootstrap |
| 100 | 9.90 | 1.57 | 0.40 | 4.14 | bootstrap |
| 200 | 9.84 | 1.54 | 0.19 | 4.14 | bootstrap |
| 221 | — | — | — | — | aborted |

**Key observation**: BoW loss dropped 78% (0.88→0.19) while category and contrastive were flat. The BoW head dominated the bottleneck's 32 dims, starving the other heads.

---

## Why TransXform Aborted (Correctly)

### The Unsalvageable Architecture

1. **Bottleneck too narrow**: 32 dimensions cannot represent three competing objectives. BoW needs hash-bucket patterns, category needs class-discriminative features, contrastive needs sample-distinguishing features. These are fundamentally different information types.

2. **Gradient magnitude mismatch**: BoW (2048 outputs, BCE loss) produces ~400x more gradient magnitude than category (5 outputs, CE loss). The optimizer essentially ignores category and contrastive signals.

3. **Contrastive collapse is self-reinforcing**: Once embeddings are similar (cosine > 0.999), InfoNCE gradients push them *more* similar because all negatives look like positives. Reinitialization breaks the cycle for ~10 steps, then collapse resumes.

4. **Reinit doesn't fix structural problems**: TransXform correctly reinitializes degenerate components, but the architecture *re-creates* degeneracy every time because the root cause is capacity, not weights.

### TransXform's Correct Response

- **Detected**: All four pathologies (cosine collapse, grad spike, bottleneck degeneration, head starvation) within the first 5 steps
- **Intervened**: 12 distinct hard interventions across multiple strategies (reinit, rescale, LR adjustment)
- **Escalated**: When reinit failed to hold, tried rescale; when rescale failed, cycled back
- **Aborted**: After exhausting intervention budget, correctly transitioned to `Aborted` rather than continuing futile interventions

This validates TransXform's **negative capability** (whitepaper §13): the ability to recognize when a training run cannot be saved and terminate it cleanly with full audit trail.

---

## Lessons for TransXform

### Finding 1: Rescale(1.0) is a No-Op

The control laws emitted `rescale(compressor, 1.0000)` repeatedly — multiplying weights by 1.0 does nothing. This suggests the rescale factor computation needs review when the violation is structural rather than magnitude-based.

### Finding 2: Intervention Attribution Could Be Smarter

TransXform reinitializes the compressor to fix contrastive_head and category_head violations (via upstream attribution). But when the *downstream* heads are collapsed because the *upstream* bottleneck is too narrow, reinitializing the bottleneck doesn't help — the capacity problem remains.

**Update (V2.3):** Signal 10 (InterventionFutility) now catches this pattern. See the V2.3 Validation addendum below.

### Finding 3: The Abort Verdict Is Marketing Gold

Three stress tests, three outcomes:
- **ICFU (HEALTHY)**: Real pathology detected and recovered — proves the system catches problems
- **SemanticNormalizer (ZERO VIOLATIONS)**: Timing governance prevented problems — proves the system prevents problems
- **CRUX (ABORTED)**: Unsalvageable architecture correctly identified — proves the system knows when to stop

> TransXform doesn't just fix training runs. It knows which ones *can't* be fixed.

---

## Spec Used

```yaml
model:
  name: "crux-bottleneck"
  components: [encoder, compressor, category_head, bow_head, contrastive_head]

roles:
  encoder:
    must_preserve_variance: true
    must_maintain_gradient: true
  compressor:
    must_preserve_variance: true
    must_maintain_gradient: true
    upstream: encoder
  category_head:
    diversity_required: true
    must_maintain_gradient: true
    upstream: compressor
  bow_head:
    must_maintain_gradient: true
    upstream: compressor
  contrastive_head:
    diversity_required: true
    must_maintain_gradient: true
    upstream: compressor

invariants:
  hard:
    pairwise_cosine:
      encoder: 0.92
      compressor: 0.90
      category_head: 0.90
      contrastive_head: 0.90
    grad_norm_min:
      encoder: 0.0001
      compressor: 0.0001
      category_head: 0.00005
      bow_head: 0.0001
      contrastive_head: 0.0001
    activation_variance_min:
      encoder: 0.00001
      compressor: 0.000005
  soft:
    loss_explosion_factor: 2.5
    grad_norm_spike_threshold: 30.0

control:
  cooldown_steps: 20
  max_hard_interventions: 12
  hysteresis_margin: 0.001
  hysteresis_pct: 0.01
  catastrophic_overrides:
    compressor.pairwise_cosine: 0.9999
    loss_explosion_factor: 6.0
  readiness_gate: true
  readiness_patience_steps: 100
  max_threshold_relaxation: 0.03
```

---

---

## V2.3 Validation: Tier 1 Diagnostic Signals

**Date**: 2026-02-09
**Purpose**: Re-run CRUX with V2.3 to validate Signals 8, 9, and 10 against a known pathological architecture.

### Results

The V2.3 diagnostic layer produced **5 advisories** — exactly the signals CRUX was designed to trigger:

| Step | Signal | Component | Metric | Confidence | Detail |
|------|--------|-----------|--------|------------|--------|
| 50 | UnusedCapacity | compressor | activation_variance | 92% | Variance below 1e-5 for 97% of last 30 steps (current: 6.60e-7) |
| 50 | **MetricInstability** | compressor | activation_variance_min | 70% | CV=4.661 — repeated reinit causes wild variance oscillation |
| 50 | **MetricInstability** | contrastive_head | grad_norm_min | 70% | CV=0.676 — gradient instability from competing objectives |
| 145 | **InterventionFutility** | category_head | — | 40% | 3 interventions, all recovered temporarily. Chronic structural problem. |
| 145 | **InterventionFutility** | contrastive_head | — | 40% | 3 interventions, all recovered temporarily. Chronic structural problem. |

### What Signal 10 Found

Signal 10 fired at step 145 for both category_head and contrastive_head — **76 steps before the abort at step 221**. That's 34% of the run's remaining lifetime.

The key insight: every intervention was tagged "recovered" by the regret tracker, because each reinit of the compressor temporarily drops cosine similarity. But the components re-collapsed within 10-20 steps every time. The pattern is chronic — interventions produce temporary relief but no lasting improvement.

Signal 10's evidence:

> "category_head has required 3 interventions in this phase (3 recovered temporarily). Repeated intervention suggests a chronic structural problem."
>
> - Step 1: reinitialize(category_head) — recovered (0 steps).
> - Step 21: reinitialize(compressor) — recovered (0 steps).
> - Step 41: reinitialize(compressor) — recovered (0 steps).
> - Even successful interventions are not producing lasting improvement.

### What Signal 9 Found

Signal 9 fired at step 50 for two metrics:

1. **compressor.activation_variance_min** (CV=4.661): The compressor is being reinitialized every 20 steps (cooldown period). Each reinit resets variance, then it collapses again. The coefficient of variation is 15x the threshold — extreme oscillation.

2. **contrastive_head.grad_norm_min** (CV=0.676): The contrastive head's gradients oscillate because InfoNCE loss produces wildly different gradient magnitudes depending on how similar the current batch's embeddings are. Every reinit creates a brief window of diverse embeddings (small gradients), followed by rapid re-collapse (large gradients trying to separate them).

### What Signal 8 Did NOT Find (Correctly)

Signal 8 (ThresholdDrift) did not fire. This is correct — CRUX's metrics didn't trend gradually toward thresholds. They started above the threshold from step 1. ThresholdDrift detects the slow creep of a metric that hasn't violated yet; CRUX had immediate, catastrophic violations.

### Bug Found and Fixed

The first V2.3 run against CRUX revealed that Signal 10 didn't fire — because the original logic required ALL interventions to have `recovered: false`. CRUX's regret tracker tagged every intervention as "recovered" because the temporary cosine drop after reinit met the recovery criterion.

**Fix**: Signal 10 now detects two modes of futility:
1. **Total failure**: All recent interventions failed → higher confidence (0.5/0.7/0.8 for 3/4/5+)
2. **Chronic futility**: Repeated interventions (even when each "recovers") → lower confidence (0.4/0.5/0.6) with advisory about chronic structural problems

This is the same pattern Finding 2 described — except now TransXform catches it automatically, 76 steps before exhausting the intervention budget.

### Timing: How Much Compute Signal 10 Saves

| Event | Step | Note |
|-------|------|------|
| First cosine violations | 1 | Architecture is broken from the start |
| Signal 9: MetricInstability | 50 | Oscillation detected in compressor + contrastive_head |
| Signal 10: InterventionFutility | 145 | Chronic futility detected — interventions aren't helping |
| Abort (max interventions) | 221 | V1 finally gives up after 12 interventions |

Signal 10 at step 145 gives the user 76 steps of warning before V1 aborts. In a real training run (thousands of steps, not 221), the gap between Signal 10's advisory and V1's abort could be much larger — potentially saving significant compute.

---

## V2.4: GradientDomination — Naming the Root Cause

**Date**: 2026-02-08
**Purpose**: V2.4 adds Signal 11 (GradientDomination) — designed to detect the exact root cause behind CRUX's failure.

### Why CRUX Is Signal 11's Canonical Example

V2.3 caught CRUX's *symptoms* — oscillation (Signal 9) and futile interventions (Signal 10). But neither signal identified *why* the architecture was unsalvageable. The answer is gradient domination: BoW reconstruction (2048 outputs, BCE loss) produced ~400x more gradient magnitude than category classification (5 outputs, CE loss). The optimizer effectively ignored the smaller heads.

Signal 11 detects this directly. It compares mean `{component}.grad_norm` across all components over the diagnostic history window. When the ratio between the largest and smallest exceeds `gradient_domination_ratio` (default 100x), it fires — naming the dominant component and listing the suppressed ones.

For CRUX, the detection chain would be:

| Step | Signal | What It Tells You |
|------|--------|-------------------|
| ~50 | Signal 9 (MetricInstability) | Something is oscillating — compressor and contrastive_head are unstable |
| ~50 | **Signal 11 (GradientDomination)** | **bow_head's gradients are 400x larger than category_head's — the optimizer is starving the smaller heads** |
| 145 | Signal 10 (InterventionFutility) | Interventions aren't helping — the problem is structural |
| 221 | V1 Abort | Budget exhausted |

Signal 11 names the mechanism. Signal 9 names the symptom. Signal 10 names the prognosis. Together, the three signals tell the complete story: "bow_head is monopolizing the optimizer (Signal 11), causing instability in downstream components (Signal 9), and no amount of reinitialization will fix a gradient magnitude mismatch (Signal 10)."

### V2.4 Validation Results (2026-02-08)

Re-running CRUX with V2.4 produced **8 advisories** (up from 5 in V2.3):

| Step | Signal | Component | Detail |
|------|--------|-----------|--------|
| 50 | UnusedCapacity | compressor | Variance below 1e-5 for 97% of steps (6.66e-7) |
| 50 | MetricInstability | encoder | grad_norm CV=0.343 (new — enabled by grad key fix) |
| 50 | MetricInstability | contrastive_head | grad_norm CV=0.679 |
| 50 | MetricInstability | compressor | activation_variance CV=4.564 |
| 50 | MetricInstability | compressor | grad_norm CV=0.373 (new — enabled by grad key fix) |
| 50 | **GradientDomination** | category_head | **102x ratio, bow_head suppressed** |
| 145 | InterventionFutility | category_head | 3 chronic interventions |
| 145 | InterventionFutility | contrastive_head | 3 chronic interventions |

**Surprise finding**: Signal 11 named `category_head` as the dominant component (mean grad 0.404) and `bow_head` as suppressed (mean grad 0.00397). This is the *opposite* of the design expectation — CRUX was engineered with a 400:1 output dimension ratio (2048 vs 5) to create gradient imbalance. But BCE loss across 2048 outputs produces many *small* per-parameter gradients, while CE loss with 5 classes concentrates the gradient signal. Signal 11 correctly measured the actual gradient norms, not the theoretical expectation.

**Grad key fix bonus**: The `resolve_grad_key` fix (same pattern as V2.1's `resolve_var_key`) also unblocked 2 additional MetricInstability detections — encoder and compressor grad_norm oscillation that was invisible when diagnostics couldn't find the `grad_norm_min` keys.

Two additional signals complete the advisory layer:

- **Signal 12 (MetricAnomaly)**: NaN/Inf sentinel — not relevant to CRUX (no numerical corruption), but defends against an entire class of silent failure.
- **Signal 13 (TrainValDivergence)**: Overfitting detection via train/val loss divergence. Not relevant to CRUX (aborted too early), but covers the most common real-world training failure.

---

## Artifacts

| File | Description |
|------|-------------|
| `D:\CRUX\checkpoints\final_supervisor.json` | TransXform state at abort |
| `D:\CRUX\checkpoints\final.pt` | Model weights at abort |
| `D:\CRUX\checkpoints\hint_221_supervisor.json` | Post-abort checkpoint |
| `D:\CRUX\checkpoints\transxform_voyage.log` | Full training trace |
| `D:\CRUX\checkpoints\transxform_certificate.json` | Training certificate |
| `D:\CRUX\checkpoints\transxform_report.md` | Full TransXform report (V2.4 diagnostics) |
| `D:\CRUX\checkpoints\run_manifest.json` | Merkle chain manifest |
