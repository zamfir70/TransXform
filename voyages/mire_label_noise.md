# TransXform Voyage: MIRE — Model In Representation Equilibrium

**Date**: 2026-02-09
**Status**: Complete (33,780 steps)
**Verdict**: HEALTHY — LossStagnation signal detected label noise plateau with zero hard violations

---

## Model Under Supervision

| Parameter | Value |
|-----------|-------|
| Model | mire-label-noise |
| Parameters | ~525K |
| Architecture | Hash embed (2048x128) + 2-layer transformer (d=128, 2 heads, FFN=256) + classifier (128->5) |
| Task | 5-class text classification with progressive label noise |
| Dataset | 216,232 samples from 27,029 RFPV2 sequences (8 windows/seq, 2048 tokens/seq) |
| Batch size | 64 |
| Learning rate | 0.001 |
| Epochs | 10 (33,780 steps) |
| Device | CPU |

### Components

| Component | Role | Architecture |
|-----------|------|-------------|
| encoder | 2-layer transformer | Hash embed -> 2 self-attention layers (d=128, 2 heads, FFN=256) |
| classifier | 5-class output | Linear(128, 5) -- passive (observe only, no interventions) |

### Design Intent: The Invisible Failure

MIRE was designed to test V2.2's LossStagnation signal (Signal 7) — the ability to detect a model that looks healthy but can't learn.

The architecture is clean. The training loop is standard. No poison, no corruption, no structural sabotage. The only problem is the data: progressive label noise drowns the learning signal.

- **Steps 0-311**: Clean labels. Model learns quickly, loss drops 4.17 -> 0.58.
- **Steps 311+**: Noise ramps from ~2% toward 90%. Each sample has `min(step/15000, 0.9)` probability of receiving a random label.
- **Steps 15,000+**: 90% noise. Loss plateaus at ~1.60 (near random chance for 5 classes: ln(5) = 1.609). Gradients normal. Representations diverse. Cosine healthy.

Every V1 hard invariant passes. Signal 4 (DynamicallyUnlearnableRegime) doesn't fire because gradients are healthy — well above the vanishing threshold. The model is actively trying to learn, but 90% of its supervision is noise. Only Signal 7 catches this.

---

## What Happened

### Timeline

| Step | Noise | Loss | Cosine | Variance | Event |
|------|-------|------|--------|----------|-------|
| 0 | 0% | 4.17 | 0.87 | 6.68 | Training begins. |
| 49 | 0.3% | — | — | — | Phase: bootstrap -> representation_formation |
| 99 | 0.6% | — | — | — | Phase: representation_formation -> stabilization |
| 149 | 1.0% | — | — | — | Phase: stabilization -> refinement |
| 311 | 2.1% | 0.58 | — | — | **Best loss achieved.** Never beaten by >1%. |
| 500 | 3% | 0.71 | 0.60 | 6.50 | Loss already climbing from noise. |
| **2,400** | **14%** | **1.05** | **0.46** | **5.61** | **ADVISORY: LossStagnation** (2,089 steps since best, confidence 32%) |
| 5,000 | 30% | 1.05 | 0.25 | 4.49 | Loss drifting up, representations still healthy. |
| 10,000 | 60% | 1.44 | 0.26 | 3.30 | Loss near plateau. Cosine still low, variance healthy. |
| 15,000 | 90% | 1.62 | 0.07 | 2.34 | Noise maxed. Loss at theoretical floor. |
| 25,000 | 90% | 1.58 | 0.13 | 1.46 | Loss flat. Model in equilibrium with noise. |
| 33,780 | 90% | 1.60 | 0.16 | 1.25 | Run complete. |

### Diagnostic Advisory

**1 total** (0 acknowledged):

**Step 2,400 — LossStagnation** (32% confidence)
> "Loss has stagnated at 0.5796 for 2089 steps despite healthy gradient flow. Training may not be making meaningful progress."
>
> Evidence:
> - Loss has not improved beyond 0.5796 for 2089 steps (best seen at step 311). The improvement threshold is 1.0%.
> - Gradient norm in encoder averages 9.47e-1 — healthy flow, suggesting the model is actively trying to learn.
> - Gradient norm in classifier averages 9.83e-1 — healthy flow, suggesting the model is actively trying to learn.
> - Healthy gradients with stagnant loss suggests the model may be stuck in a flat loss basin, the data signal-to-noise ratio may be too low, or the architecture may lack capacity for the task.

The advisory correctly identified the root cause: "the data signal-to-noise ratio may be too low."

### Hard Violations & Interventions

| Metric | Violations | Interventions |
|--------|-----------|---------------|
| pairwise_cosine | 0 | 0 |
| grad_norm_min | 0 | 0 |
| activation_variance_min | 0 | 0 |
| loss_explosion_factor (soft) | 0 | 0 |
| grad_norm_spike_threshold (soft) | 0 | 0 |

**Zero violations. Zero interventions.** The architecture is healthy and the training loop is clean — the only problem is the data.

### Why Signal 4 Did NOT Fire

Signal 4 (DynamicallyUnlearnableRegime) detects loss plateau + **pathological** gradients: vanishing (< 1e-7) or oscillating (CV > 2.0). MIRE's gradients stayed healthy throughout:

| Step | Encoder Grad Norm | Classifier Grad Norm |
|------|-------------------|---------------------|
| 500 | ~0.95 | ~0.98 |
| 15,000 | ~0.07 | ~0.42 |
| 33,780 | 0.045 | 0.177 |

All values well above the 1e-7 vanishing floor. The model is actively computing gradients and trying to learn — the labels are just noise. This is exactly the gap Signal 7 was designed to fill.

---

## Analysis

### Why Signal 7 Fired at Step 2,400 (Not Step 17,000)

The plan predicted detection at ~step 17,000 (2,000 steps after the noise plateau at 15,000). In practice, the signal fired at step 2,400. The reason is straightforward:

1. The model learned fast — best loss 0.58 was achieved at step 311
2. The noise ramp starts at step 0, not step 5,000
3. By step 311, noise was only 2.1%, but the model had already extracted most learnable signal
4. From step 311 to step 2,400 (2,089 steps), the ramping noise prevented any >1% improvement
5. At step 2,400, patience (2,000 steps) expired and the signal fired

This is correct behavior. The signal detected the exact moment the noise began dominating the learning signal. The model hit its best loss early and never recovered — precisely because the noise was already corrupting the gradient signal enough to prevent further improvement.

### The Loss Trajectory

```
Loss vs. Step (with noise ramp)

4.0 |*
    |
3.0 |
    |
2.0 |                     ............................................
    |          ........''''
1.0 |     ..'''
    |   .'
0.5 |  *  <- best loss (step 311)
    |
0.0 +--------+--------+--------+--------+--------+--------+--------
    0      5000    10000   15000   20000   25000   30000   35000
              noise: 0%→30%→60%→90%→90%→90%→90%→90%
```

The loss follows the noise ramp almost exactly. Once noise hits 90% at step 15,000, loss plateaus at ~1.60 (random chance for 5 classes). The model is in equilibrium — its gradient signal is 90% noise, so it can't improve.

### Structural Health Throughout

| Metric | Step 0 | Step 311 (best) | Step 15,000 (plateau) | Step 33,780 (final) |
|--------|--------|-----------------|----------------------|---------------------|
| Loss | 4.17 | 0.58 | 1.62 | 1.60 |
| Cosine | 0.87 | — | 0.07 | 0.16 |
| Variance | 6.68 | — | 2.34 | 1.25 |
| Grad norm | — | — | 0.07 | 0.045 |

Cosine dropped steadily (healthy diversity). Variance decreased gradually (no collapse, just settling). Gradients remained nonzero. Every structural invariant was satisfied at every step. A human watching only V1 metrics would see a perfectly healthy training run that simply converged early.

---

## Lessons for TransXform

### Finding 1: Stagnation Is the Most Common Real-World Failure

The gap Signal 7 fills is the most common real-world training failure: the model trains but doesn't learn. Wrong learning rate, noisy data, capacity mismatch, flat loss basins — all produce the same pattern. Loss plateaus. Gradients flow. Representations evolve. Nothing improves. Without Signal 7, TransXform would give these runs a clean bill of health.

### Finding 2: Gradient Floor Check Prevents False Positives

Signal 7's gradient floor check (requires at least one component with grad_norm > 1e-5) is critical. Without it, the signal would fire on converged models — loss flat because training is done, not because training is stuck. MIRE's gradients stayed healthy throughout (0.045-0.98), so the floor check trivially passed. The important thing is that it *exists* for runs where convergence is real.

### Finding 3: Early Detection via Best-Loss Tracking

The stateful tracking approach (remembering `best_loss` and `best_loss_step` rather than comparing recent windows) enables detection at any timescale. MIRE's stagnation started at step 311 — far earlier than the 50-step history window could capture. By tracking the absolute best, the signal accumulates evidence over the entire phase.

### Finding 4: Signal 7 Completes the Diagnostic Suite

Five stress tests, five outcomes:
- **ICFU (HEALTHY)**: V1 caught structural violations, V2 caught epoch-boundary instability
- **SemanticNormalizer (ZERO VIOLATIONS)**: Readiness gate prevented cliff transitions
- **CRUX (ABORTED)**: V1 correctly identified unsalvageable architecture
- **FROG (HEALTHY, SHORTCUT DETECTED)**: V2 caught slow poison invisible to V1
- **MIRE (HEALTHY, STAGNATION DETECTED)**: V2.2 caught loss plateau invisible to V1 and Signals 1-6

MIRE is the first test where only Signal 7 detected the problem. Signals 1-6 had nothing to report. Hard invariants had nothing to enforce. The model was structurally healthy but making no progress — and TransXform now flags that.

---

## Spec Used

```yaml
model:
  name: "mire-label-noise"
  components: [encoder, classifier]

roles:
  encoder:
    must_preserve_variance: true
    must_maintain_gradient: true
  classifier:
    diversity_required: true
    must_maintain_gradient: true
    upstream: encoder
    passive: true

invariants:
  hard:
    pairwise_cosine:
      encoder: 0.98
    grad_norm_min:
      encoder: 0.00001
    activation_variance_min:
      encoder: 0.000001
  soft:
    loss_explosion_factor: 100.0
    grad_norm_spike_threshold: 500.0

phases:
  bootstrap:
    cosine: 1.0
    guard: 100 steps
  representation_formation:
    cosine: 0.95
    guard: 200 steps
  stabilization:
    cosine: 0.90
    guard: 300 steps
  refinement:
    guard: 500 steps

control:
  cooldown_steps: 50
  max_hard_interventions: 8
  hysteresis_margin: 0.002
  hysteresis_pct: 0.02
  readiness_gate: true
  readiness_patience_steps: 500
  max_threshold_relaxation: 0.02
```

---

## Artifacts

| File | Description |
|------|-------------|
| `D:\MIRE\checkpoints\final_supervisor.json` | TransXform state at completion |
| `D:\MIRE\checkpoints\final.pt` | Model weights at completion |
| `D:\MIRE\checkpoints\transxform_voyage.log` | Full training trace |
| `D:\MIRE\checkpoints\transxform_certificate.json` | Training certificate |
| `D:\MIRE\checkpoints\transxform_report.md` | Full TransXform report |
| `D:\MIRE\checkpoints\run_manifest.json` | Merkle chain manifest |
