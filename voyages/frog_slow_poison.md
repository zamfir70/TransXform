# TransXform Voyage: FROG — From Representations to Overfitting Gradually

**Date**: 2026-02-09
**Status**: Complete (27,024 steps)
**Verdict**: HEALTHY — diagnostic layer detected slow poison with zero hard violations

---

## Model Under Supervision

| Parameter | Value |
|-----------|-------|
| Model | frog-slow-poison |
| Parameters | ~525K |
| Architecture | Hash embed (2048x128) + 2-layer transformer (d=128, 2 heads, FFN=256) + classifier (128→5) |
| Task | 5-class text classification with gradual label leak injection |
| Dataset | 216,232 samples from 27,029 RFPV2 sequences (8 windows/seq, 128 tokens/window) |
| Batch size | 64 |
| Learning rate | 0.001 |
| Epochs | 8 (27,024 steps) |
| Device | CPU |

### Components

| Component | Role | Architecture |
|-----------|------|-------------|
| encoder | 2-layer transformer | Hash embed → 2 self-attention layers (d=128, 2 heads, FFN=256) |
| classifier | 5-class output | Linear(128, 5) — passive (observe only, no interventions) |

### Design Intent: Silent Corruption

FROG was designed to test the V2 diagnostic layer's ability to detect shortcut learning when all V1 hard invariants remain satisfied:

1. **Gradual label leak**: A class-specific unit-normalized embedding vector is injected at position 0, with strength growing linearly from 0.0 to 5.0 over 15,000 steps
2. **Loose hard thresholds**: cosine 0.98, grad_norm_min 0.00001, variance_min 0.000001 — deliberately wide enough that the poison never triggers hard violations
3. **Passive classifier**: 5-dim logits naturally have high cosine (~0.99) which is meaningless for diversity measurement
4. **Soft violations logged but not applied**: Growing poison causes loss ratio spikes, but LR adjustments are suppressed to let the poison fully develop

The model should learn normally from content tokens in early training, then gradually shift to exploiting the position-0 shortcut as leak strength increases. Loss should crash toward zero while encoder representations inflate — a pattern invisible to V1's structural invariants.

---

## What Happened

### Timeline

| Step | Leak | Loss | Cosine | Variance | Event |
|------|------|------|--------|----------|-------|
| 0-5,000 | 0→1.67 | 1.60→0.58 | 0.35→0.44 | 6.5→6.6 | Normal training. Model learning from content. |
| 5,000 | 1.67 | 0.58 | 0.44 | 6.6 | **Checkpoint saved** (pre-poison baseline) |
| 7,000 | 2.33 | 0.50 | 0.44 | 6.3 | Last normal step. Poison threshold about to cross. |
| 7,500 | 2.50 | 0.26 | 0.23 | 10.4 | **Inflection point.** Variance jumps 65%. Loss halves. |
| **7,900** | **2.63** | **0.39** | **—** | **10.3** | **ADVISORY: ShortcutLearning via variance explosion** (63% increase) |
| 9,000 | 3.00 | 0.09 | 0.19 | 34.8 | Variance 5.5x baseline. Loss approaching zero. |
| **10,900** | **3.63** | **0.03** | **—** | **—** | **ADVISORY: ShortcutLearning via cosine drift** (classifier) |
| 15,000 | 5.00 | 0.00 | 0.32 | 126 | Leak maxed. Model fully exploiting shortcut. |
| 27,024 | 5.00 | 0.00 | 0.23 | 356 | Run complete. Variance 57x baseline. |

### Diagnostic Advisories

**3 total** (0 acknowledged):

1. **Step 3,600 — LossRepresentationMisalignment** (30% confidence)
   > Early training noise. Loss improved 2.5% while classifier metrics stagnated. Low confidence, correct to flag but not alarming.

2. **Step 7,900 — ShortcutLearning** (50% confidence)
   > "Loss improved from 0.6181 to 0.3851 (37.7% decrease) over the last 2000 steps. Activation variance in encoder increased 63% (from 6.32e0 to 1.03e1). This rapid variance growth while loss improves suggests the model is amplifying a low-dimensional shortcut feature rather than learning distributed representations."

   This is the key detection. The diagnostic correctly identified that the variance *explosion* (not collapse) paired with loss improvement indicates shortcut exploitation. This signal was added specifically because FROG's first run revealed the gap.

3. **Step 10,900 — ShortcutLearning** (50% confidence)
   > Classifier cosine drift — cosine increased from 0.3194 to 0.3415 while loss improved 58.7%.

### Hard Violations & Interventions

| Metric | Violations | Interventions |
|--------|-----------|---------------|
| pairwise_cosine | 0 | 0 |
| grad_norm_min | 0 | 0 |
| activation_variance_min | 0 | 0 |
| loss_explosion_factor (soft) | 2,018 | 0 (logged only) |

**Zero hard violations. Zero interventions.** Exactly as designed — the poison operates entirely within the structural invariant boundaries.

### The Poison's Footprint

| Metric | Step 5,000 (pre-poison) | Step 27,024 (final) | Change |
|--------|------------------------|---------------------|--------|
| Loss | 0.58 | 0.000000056 | -100% (memorized) |
| Encoder variance | 6.6 | 356 | +5,290% (57x) |
| Encoder cosine | 0.44 | 0.23 | -48% (healthier!) |
| Grad norm | normal | 2.2e-6 | Near-zero (converged) |

The irony: cosine *improved* (dropped further from 0.98 threshold) because the model's outputs became dominated by the 5 unique class-specific leak vectors. The variance exploded because those vectors amplified the position-0 feature. A naive observer would see "good cosine, good loss" and declare the model healthy. Only the diagnostic's trend analysis caught the real story.

---

## Bugs Found and Fixed

### Bug 1: Variance Explosion Detection Missing (Run 1)

The original `detect_shortcut_learning()` only checked for variance *decrease* (representation collapse). FROG's poison causes variance *increase* (feature amplification). Added `shortcut_variance_explosion` config field and the corresponding check — if variance increases by more than the threshold (default 100%, FROG uses 50%) while loss improves, it fires ShortcutLearning.

### Bug 2: Metric Key Mismatch (Run 2)

The diagnostic layer looked for `{component}.activation_variance` but the standard invariant name (and what all stress tests report) is `{component}.activation_variance_min`. The `mean_metric()` helper uses exact key matching, so variance lookups silently returned `None` in all three signals that check variance (UnusedCapacity, LossRepresentationMisalignment, ShortcutLearning).

Added `resolve_var_key()` and `mean_var_metric()` helpers that try the `_min` suffix first, falling back to the plain key for backward compatibility with unit tests.

### Bug 3: tch VarStore Save/Load Format Mismatch

`vs.save()` uses `Tensor::save_multi()` (pickle format) but `vs.load()` uses `torch::jit::_load_parameters()` (JIT format). On some PyTorch versions these are incompatible. Workaround: use `Tensor::load_multi_with_device()` + manual `var.copy_()` for checkpoint resume.

---

## Iterative Improvement Across 3 Runs

| Run | Diagnostic Result | Fix Applied |
|-----|-------------------|-------------|
| 1 | **No detection.** ShortcutLearning only checked for collapse (variance down). Poison causes inflation (variance up). | Added `shortcut_variance_explosion` detection |
| 2 | **Detected at step 10,500** via cosine drift. Variance explosion code present but silently skipped due to metric key mismatch. | Fixed `activation_variance` → `activation_variance_min` resolution |
| 3 | **Detected at step 7,900** via variance explosion. First advisory 2,600 steps earlier than Run 2. Full detection with correct evidence. | Final version |

---

## Lessons for TransXform

### Finding 1: Shortcut Learning Manifests as Inflation, Not Just Collapse

The ML literature typically describes shortcut learning as "representation collapse" — outputs becoming similar, variance decreasing. FROG demonstrates the opposite: when a model discovers a trivial feature, it may *amplify* that feature, causing variance to explode while cosine similarity actually improves (drops).

TransXform's diagnostic layer now detects both patterns.

### Finding 2: Silent Metric Key Mismatches Are Dangerous

The `activation_variance` vs `activation_variance_min` mismatch caused three diagnostic signals to silently skip variance analysis. No error, no warning — just `None` from the HashMap lookup. This class of bug is particularly insidious because the system appears to work (other signals still fire) but key detection pathways are disabled.

The fix (try `_min` suffix first, fall back to plain key) is defensive and backward-compatible.

### Finding 3: Checkpoint Resume Enables Iterative Debugging

Being able to resume from step 5,000 (pre-poison) saved ~10 minutes per run. The diagnostic fix → rebuild → resume → verify cycle ran 3 times in under an hour. Without checkpointing, each iteration would have required a full 18-minute run from scratch.

### Finding 4: The Diagnostic Layer Adds Real Value

Four stress tests, four outcomes:
- **ICFU (HEALTHY)**: V1 caught structural violations, V2 caught epoch-boundary instability
- **SemanticNormalizer (ZERO VIOLATIONS)**: Readiness gate prevented cliff transitions
- **CRUX (ABORTED)**: V1 correctly identified unsalvageable architecture
- **FROG (HEALTHY, DIAGNOSTICS FIRED)**: V1 saw nothing wrong. V2 caught the slow poison at step 7,900.

FROG is the first test where **only V2 detected the problem**. This validates the diagnostic layer as a necessary complement to structural enforcement — it catches patterns that hard invariants cannot.

---

## Spec Used

```yaml
model:
  name: "frog-slow-poison"
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
| `D:\FROG\checkpoints\final_supervisor.json` | TransXform state at completion |
| `D:\FROG\checkpoints\final.pt` | Model weights at completion |
| `D:\FROG\checkpoints\step_5000_supervisor.json` | Pre-poison checkpoint (supervisor) |
| `D:\FROG\checkpoints\step_5000.pt` | Pre-poison checkpoint (model) |
| `D:\FROG\checkpoints\transxform_voyage.log` | Full training trace |
| `D:\FROG\checkpoints\transxform_certificate.json` | Training certificate |
| `D:\FROG\checkpoints\transxform_report.md` | Full TransXform report |
| `D:\FROG\checkpoints\run_manifest.json` | Merkle chain manifest |
