# TransXform: Boundary-Governed Transformer Training

## A Spec-Driven Supervisory Architecture That Makes Silent Training Failure Impossible

**Version 1.2 — February 2026**
**Authors:** John M. Kuykendall (Acorn KC, LLC), with recursive design contributions from GPT 5.2 and Claude
**Status:** Architecture Specification

---

## Abstract

Transformer training is an uncontrolled dynamical system. Practitioners define scalar objectives, apply local gradient descent, and hope the loss landscape cooperates. When it doesn't — when representations collapse, attention heads die, submodules enter zero-gradient subspaces, or shortcut learning produces degenerate solutions — the failure is silent. Loss decreases. Dashboards stay green. The model learns nothing useful.

TransXform eliminates silent training failure by making training closed-loop, stateful, and governed by enforceable invariants. It interposes a supervisory authority between the optimizer and the model, treating gradient updates as *provisional state changes* subject to authoritative post-commit correction. When invariants are violated, the supervisor intervenes with component-local actions — reinitializing frozen submodules, rescaling collapsed representations, dampening gradient spikes — while preserving healthy learned structure elsewhere.

The system borrows its authority model from EDGE (Explicit Delimitation of Generative Epistemics), a boundary-condition cognitive substrate designed for inference-time safety enforcement. TransXform applies the same principle to training: **illegal states cannot persist because they are detected and corrected in real time, not discovered post-hoc.**

A diagnostic layer (§12) supplements hard enforcement with predictive advisory signals — detecting gradient domination, shortcut learning, loss stagnation, overfitting, and intervention futility before they escalate to invariant violations. The diagnostic layer is strictly non-authoritative: it observes and advises, never intervenes.

The result is transformer training that is specifiable, auditable, and recoverable. Failures are localized. Recovery is guaranteed. Progress is explainable. Training stops being alchemy and becomes ordinary engineering.

---

## 1. The Problem: Training Doesn't Self-Correct

### 1.1 Open-Loop Optimization

Modern transformer training is open-loop. A loss function is defined, an optimizer is chosen, and gradient descent runs until a stopping criterion is met. The implicit assumption is that loss reduction correlates with intended behavior.

This assumption is routinely violated:

| Failure Mode | What the Dashboard Shows | What Actually Happened |
|---|---|---|
| Representational collapse | Loss decreasing | All representations converged to a single direction |
| Dead attention heads | Loss decreasing | Heads learned identity or constant functions |
| Emission head frozen | Loss stable | Submodule entered zero-gradient subspace |
| Shortcut learning | Loss decreasing rapidly | Model memorized surface patterns, not structure |
| Recurrence erasing signal | Loss stable | State tokens dominate input embeddings, erasing content |
| Stability loss trivial convergence | Auxiliary loss dropping | Loss rewarding degenerate fixed points |

In every case, the optimizer sees "this direction reduces loss" and proceeds. It cannot see that the direction leads to a degenerate attractor. The loss function doesn't encode what the practitioner actually wants.

### 1.2 Degeneracy Is a Stable Attractor

The critical insight is that degenerate solutions are not transient — they are **stable minima**. A collapsed emission head has zero gradient. A dead attention head has zero gradient. A one-dimensional projection that satisfies the loss has no pressure to become multi-dimensional.

From the optimizer's perspective, these states are converged. From the practitioner's perspective, the model is dead.

This gap — between formal optimization and intended behavior — is where TransXform lives.

### 1.3 Why "Let It Cook" Fails

The conventional advice for stalled training is patience: let the optimizer find its way out. This advice is correct only when the architecture is proven and the only question is convergence time.

For research architectures, novel training objectives, or multi-task loss recipes, patience is dangerous. Empirical evidence from training LOOM (a dual-process recurrent transformer):

| Step | Practitioner Response | Actual State |
|---|---|---|
| 4,000 | "Let it cook" | Stability loss rewarding trivial convergence |
| 6,000 | "Let it cook" | Emission collapse; downstream probes at random chance |
| 14,000 | "Let it cook" | Recurrence erasing input signal |
| 16,000 | "Let it cook" | Emissions re-collapsed despite normalization fix |
| 381 (after fix) | "Let it cook" | Emission head frozen in 1D; contrastive gradient = 0 |

Each failure was only caught because the practitioner built monitoring that checked the right things (emission cosine, not just loss), examined results regularly, and stopped to diagnose rather than hoping.

**The training didn't self-correct because there was no mechanism for self-correction.** This is an engineering problem with an engineering solution.

### 1.4 Why Nobody Has Built This

| Reason | Why It Persists |
|---|---|
| Big labs use human dashboards | Expensive, doesn't scale, catches failures late |
| Academics publish one-off runs | Don't need robustness for a single paper |
| "Just retrain" culture | Compute is cheap if you're Google |
| Failure modes are architecture-specific | No universal catalog... until now |
| ML culture treats training as art | Engineering discipline is unfashionable |

TransXform exists because none of these reasons are technical. They are cultural. The technical solution is straightforward.

---

## 2. Core Principles

### 2.1 Authority vs. Proposal Split

The optimizer and its gradients are **non-authoritative**. They produce state changes. The supervisor is **authoritative**. It observes committed state, evaluates it against declared invariants, and corrects violations through component-local intervention.

This is the same separation that EDGE enforces at inference time: generators propose trajectories; the boundary ledger decides which transitions are legal. TransXform applies this principle to training:

```
Inference (EDGE):     Generator proposes state → Ledger evaluates → Admit/Reject
Training (TransXform): Optimizer commits update → Supervisor evaluates → Accept/Correct
```

**A note on authority semantics.** TransXform v1 uses **post-commit authority**, not pre-commit gating. The optimizer step executes first; the supervisor then observes the resulting state and intervenes if invariants are violated. This is a deliberate engineering choice: pre-commit gating (holding the optimizer step pending approval) would require shadow-stepping or checkpoint-and-rollback on every training step, imposing unacceptable overhead. Post-commit correction is cheaper, simpler, and sufficient — the supervisor can reinitialize, rescale, or roll back any component within the same step boundary. The invariant is not "bad states never occur" but "bad states never *persist*." Future implementations may support shadow-step evaluation or delayed commit for high-risk phases where even transient illegal states are unacceptable.

**A limitation of post-commit correction.** The claim that "bad states never persist" is weaker than it appears for adaptive optimizers. When TransXform reinitializes a component's weights, optimizers like Adam retain momentum and variance EMAs computed from the pre-reinit parameter trajectory. These "momentum corpses" decay with a half-life of roughly β₂/(1−β₂) steps (~999 for standard β₂=0.999), meaning the optimizer continues pushing toward the old, discredited state for hundreds of steps after reinit. The supervisor corrects the *parameters* but not the *optimizer state* — it has no authority over the optimizer's internal accumulators.

Mitigations under investigation include **selective shadow-stepping** (running a trial optimizer step, evaluating the result against invariants, and only committing if it passes) for high-risk phases, and **optimizer state reset** (zeroing momentum/variance EMAs for the reinitialized component). Both add complexity; the current post-commit design is honest about this gap. See §15.6.

### 2.2 Loss Is Telemetry, Not Truth

Loss is observed, never trusted. Training success is defined as **invariant satisfaction**, not loss minimization. A run where loss decreases while representations collapse is a failed run. A run where loss plateaus but all invariants hold is a healthy run awaiting signal.

### 2.3 Component Locality

Interventions are per-component (head, layer, module). A frozen emission head does not require a full restart — it requires reinitialization of the emission head. A spiking gradient in one layer does not require global learning rate reduction — it requires per-component dampening.

Global restarts destroy learned structure. Component-local interventions preserve it.

### 2.4 Phase Awareness

Training is not a smooth curve. It is a sequence of **regime shifts**: representation formation, stabilization, refinement, and (sometimes) collapse and recovery. TransXform models these as explicit phases with phase-specific thresholds, transition guards, and allowed interventions.

### 2.5 Auditability

Every intervention has a reason. Every step is replayable. Silent failure is impossible by construction. The supervisor maintains an append-only boundary ledger recording every metric snapshot, invariant evaluation, and intervention taken.

### 2.6 Separation of Learning from Maintenance

The key reframe: most training failures (collapse, dead heads, 1D projections, entropy sinks) are not learning problems. They are **maintenance failures**. The optimizer handles learning. The supervisor handles structural maintenance — enforcing variance floors, mutual information between layers, per-head usefulness, and gradient liveliness. These are two coupled systems: a slow optimizer and a fast stabilizer.

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│              TransXform Supervisor                        │
│                                                          │
│  ┌─────────────────┐  ┌──────────────────┐               │
│  │ Invariant        │  │ Phase             │               │
│  │ Monitor          │  │ Controller        │               │
│  └────────┬────────┘  └────────┬─────────┘               │
│           │                    │                          │
│  ┌────────▼────────┐  ┌───────▼──────────┐               │
│  │ Control Laws     │  │ Intervention      │               │
│  │ (Hard/Soft)      │  │ Executor          │               │
│  └────────┬────────┘  └───────┬──────────┘               │
│           │                    │                          │
│  ┌────────▼────────────────────▼──────────┐               │
│  │         Boundary Ledger (Audit Log)     │               │
│  └─────────────────────────────────────────┘               │
└──────────────────────┬───────────────────────────────────┘
                       │  metrics up, corrections down (post-commit authority)
                       │
┌──────────────────────▼───────────────────────────────────┐
│         Optimizer + Model (Any Transformer)               │
│                                                          │
│  Forward → Loss → Backward → Optimizer step (commit) →    │
│  [Supervisor observes, evaluates, corrects if needed]      │
└──────────────────────────────────────────────────────────┘
```

### 3.1 Supervisor

The authoritative controller. Registers invariants from the training spec, collects metrics from model hooks, evaluates invariant satisfaction at configured cadence, decides interventions via control laws, advances or regresses phases, and halts training if the spec is unsatisfiable.

The supervisor does NOT: compute gradients, touch loss functions, or modify model architecture.

### 3.2 Metric Collector

Pure observability layer. Pulls tensors and statistics from model hooks, aggregates per-component metrics, and supports cadence control (every step for cheap metrics, every N steps for expensive ones).

Metrics are inputs to the supervisor — no logic lives here.

### 3.3 Phase Controller

A finite state machine over training regimes. Phases are condition-based, not time-based. Transitions require invariant satisfaction for T consecutive steps. Regression is allowed but limited (once per component). Hysteresis and cooldown timers prevent oscillation.

### 3.4 Intervention Executor

The actuator layer. Receives commands from the supervisor and executes them against the model and optimizer. The executor is stateless — all decisions come from the supervisor.

### 3.5 Boundary Ledger

Append-only audit log. Every entry records: `(step, phase, component, invariant, metric_snapshot, action, justification)`. This enables post-hoc analysis, replay, and proof of training correctness.

---

## 4. The Training Specification

Before training starts, the user provides a **Training Spec** — a declarative contract describing what must hold during training. This is the central innovation: training becomes specifiable, not empirical.

### 4.1 Model Declaration

```yaml
model:
  name: "loom_v2"
  layers: 24
  hidden_dim: 1024
  attention_heads: 16
  components:
    - backbone
    - compressor
    - emission_head
    - compression_decoder
    - attention_heads[0..15]
```

### 4.2 Declared Roles

Components can be annotated with declared intent:

```yaml
roles:
  attention_heads:
    diversity_required: true
    min_active_heads: 8
  emission_head:
    must_preserve_variance: true
    must_maintain_gradient: true
  compressor:
    output_diversity_required: true
```

Roles enable the supervisor to distinguish *intentional* low entropy (e.g., a routing head) from *pathological* low entropy (a dead head). Collapse is illegal only when it violates declared intent.

### 4.3 Invariants

Invariants are the physics of the training run. Violations trigger intervention.

```yaml
invariants:
  hard:
    emission_cosine_max: 0.95
    grad_norm_min:
      emission_head: 1e-3
      backbone: 1e-4
    activation_variance_min:
      all_layers: 1e-4

  soft:
    attention_entropy_min: 0.3
    attention_entropy_max: 3.5
    grad_norm_spike_threshold: 100
    loss_explosion_factor: 3.0
```

Hard invariants trigger immediate intervention. Soft invariants trigger gradual correction.

### 4.4 Phase Definitions

```yaml
phases:
  bootstrap:
    description: "Initial learning, expect high variance"
    thresholds:
      activation_variance_min: 1e-5  # loose
    max_duration_steps: 500

  representation_formation:
    description: "Enforce diversity, monitor collapse"
    thresholds:
      activation_variance_min: 1e-4  # tighter
      emission_cosine_max: 0.95
    transition_guard:
      all_hard_invariants_satisfied_for: 100  # steps

  stabilization:
    description: "Tighten bounds, reduce noise"
    thresholds:
      activation_variance_min: 5e-4
      grad_norm_spike_threshold: 50

  refinement:
    description: "Freeze healthy components, focus learning"
    allowed_interventions:
      - adjust_lr
      - freeze
```

### 4.5 Control Law Configuration

```yaml
control:
  cooldown_steps: 50          # min steps between hard interventions per component
  max_hard_interventions: 3   # per component per phase
  hysteresis_margin: 0.05     # threshold must be exceeded by this margin to trigger
  damping_factor: 0.5         # soft correction strength
```

### 4.6 Threshold Provenance

Thresholds are not free hyperparameters. They are **encoded domain knowledge**. TransXform does not discover thresholds automatically — it makes existing knowledge executable and auditable.

Thresholds are expected to originate from:

- **Prior successful runs:** Metric trajectories from healthy training provide empirical baselines.
- **Architecture profiles:** Reusable, versioned threshold bundles for common architectures (see §15.2).
- **Failure signatures:** Each documented failure mode (see §11) implies a detection threshold. These are proven, not guessed.
- **Theoretical bounds:** Where available (e.g., entropy bounds derived from head count, variance floors derived from hidden dimension).

All thresholds are explicit, logged, and versioned in the training spec. When a threshold proves wrong — too tight (false interventions) or too loose (missed failures) — the ledger provides the evidence to adjust it. The system's invariant is not "thresholds are always correct" but "threshold errors are always visible."

**This system does not remove judgment. It removes ambiguity.**

### 4.7 Threshold Discovery: Bootstrap as Observation

The preceding section describes where thresholds come from *once a practitioner has them*. But for novel architectures — the first run of the first variant — there is no prior data, no architecture profile, and no failure history. **The hardest problem is not enforcing thresholds. It is knowing what they should be.**

TransXform's bootstrap phase is designed to serve double duty. In its default mode, bootstrap enforces relaxed invariants while the model's random initialization settles. But it can also run in **observation-only mode**: all invariants are monitored but none are enforced, and the supervisor records the full metric distribution for every component.

At the end of the observation window, the system proposes thresholds derived from the observed data:

- **Hard floors/ceilings**: 1st/99th percentile of observed metric ranges, with a configurable safety margin.
- **Soft targets**: Median observed values, adjusted by the metric's natural variance.
- **Phase transitions**: Step numbers where metric distributions shift detectably.

These proposals are exactly that — proposals. They require human review before they become spec entries. The system does not claim to know what healthy training looks like for an architecture it has never seen. It claims to *measure* what training actually looks like, and to present those measurements in a form that makes threshold authorship tractable.

This is the distinction between **threshold enforcement** (V1) and **threshold discovery** — and for novel architectures, discovery is the harder problem. See §15.8.

---

## 5. Invariants: The Physics of Training

### 5.1 Metric Tiers

Not all metrics are equally important or equally cheap. TransXform defines three tiers:

**Tier 0 — Mandatory (computed every step):**
- Per-component gradient norms
- Activation variance per layer
- Representation pairwise cosine (where applicable)

**Tier 1 — Recommended (computed every N steps):**
- Attention entropy per head
- Approximate layer-to-layer mutual information
- Drift metrics (running mean/variance shift)

**Tier 2 — Optional / Advanced (computed on demand):**
- Spectrum analysis of weight matrices
- Curvature estimates
- Jacobian conditioning proxies

### 5.2 Core Invariant Catalog

These invariants are generic to all transformer architectures:

| Invariant | What It Detects | Metric | Default Threshold |
|---|---|---|---|
| **Variance Floor** | Representational collapse | Activation variance per layer | > 1e-4 |
| **Gradient Liveliness** | Dead submodules | Grad norm per component | > 1e-3 |
| **Representation Diversity** | Emission/embedding collapse | Pairwise cosine similarity | < 0.95 |
| **Entropy Bounds** | Attention head death or saturation | Attention entropy per head | 0.3 — 3.5 |
| **Loss Stability** | Gradient explosion or NaN | Loss delta vs. baseline | < 3x baseline |
| **Gradient Stability** | Gradient spikes | Grad norm | < 100 (configurable) |

### 5.3 Architecture-Specific Invariants

Users add invariants specific to their architecture. Examples from LOOM:

| Invariant | What It Detects | Metric |
|---|---|---|
| H0-Hfinal gap | Recurrence erasing signal | Cosine(H_0, H_final) < 0.5 |
| State token magnitude | State tokens dominating input | norm(state_tokens) / norm(input_embeds) < 5.0 |
| Compressor diversity | Compressor collapse | comp_cos < 0.7 |
| Contrastive gradient | Contrastive loss at plateau | contrastive_grad_norm > 1e-3 |

---

## 6. Control Laws

TransXform uses control-theoretic rules. Interventions are stabilized, not reactive.

### 6.1 Hard Boundaries (Immediate Intervention)

When a hard invariant is violated:

| Detection | Action |
|---|---|
| Representation cosine > threshold AND gradient norm < epsilon | Reinitialize the collapsed component |
| Gradient norm = 0 on any component for N steps | Reinitialize or unfreeze |
| Loss explosion (> 3x baseline) | Reduce LR globally, dump diagnostic state |
| Activation variance = 0 on any layer | Reinitialize layer, inject noise |
| Attention entropy = 0 on any head | Reinitialize head parameters |

### 6.2 Soft Boundaries (Gradual Correction)

When a soft invariant is violated:

| Detection | Action |
|---|---|
| Gradient spikes above threshold | Reduce LR for affected component by damping factor |
| Oscillation detected (metric alternating above/below threshold) | Increase damping, reduce step size |
| Slow drift in representation statistics | Rescale norms to baseline |
| MI between layers dropping below floor | Log warning, increase monitoring cadence |

### 6.3 Control Stability

To prevent oscillation or thrashing:

- **Hysteresis:** An invariant must be violated by a margin (not just touched) to trigger intervention.
- **Cooldown:** No more than one hard intervention per component per K steps.
- **Phase locking:** Thresholds only tighten when the phase advances (never during recovery).
- **Critical damping:** Soft corrections use proportional control, not bang-bang.
- **Intervention budget:** Maximum hard interventions per component per phase. Exceeding the budget triggers phase regression or training halt.

### 6.4 Intervention Regret

Interventions are auditable decisions, not assumed to be infallible. The supervisor may intervene on a component that would have recovered on its own, or fail to intervene on a component that was close to violation. Both cases are informative.

**Regret windows.** Every hard intervention opens a **regret window** of K steps (configurable, default 100). During this window, the supervisor tracks:

- **Recovery speed:** How quickly the affected component returns to invariant compliance.
- **Post-intervention improvement:** Whether downstream metrics (loss, gradient flow, representation diversity) improve relative to pre-intervention baseline.
- **Counterfactual signal:** Whether the pre-intervention metric trajectory was trending toward recovery (suggesting the intervention may have been unnecessary).

If post-intervention metrics show no improvement — or the pre-intervention trajectory was already recovering — the intervention is tagged `low_confidence` in the ledger. This does not trigger auto-reversal in v1; the supervisor's authority is unconditional. But the tag feeds offline analysis, threshold tuning, and eventual supervisor policy learning (§15.1).

**Near-miss tracking.** The supervisor also records **near-misses**: steps where a soft invariant came within the hysteresis margin of a hard threshold but did not cross it. Near-misses are logged with the same metric snapshot as violations, tagged `near_miss`. They are as informative as false positives for threshold calibration — a threshold that is never approached is too loose; a threshold that triggers constant near-misses is too tight. The ledger captures both failure and the shape of almost-failure.

---

## 7. Legal Interventions (Actuator Inventory)

TransXform is allowed to:

| Action | Scope | When Used |
|---|---|---|
| `reinitialize(component)` | Weights of a single submodule | Dead gradients, collapsed representations |
| `freeze(component)` | Stop gradient flow to a submodule | Healthy component, focus learning elsewhere |
| `unfreeze(component)` | Resume gradient flow | Component needed for current phase |
| `rescale(component, factor)` | Multiply activations or weights | Magnitude imbalance between components |
| `inject_noise(component, magnitude)` | Add controlled noise to weights | Break symmetry in collapsed subspace |
| `adjust_lr(component, factor)` | Per-component learning rate | Gradient spikes or slow convergence |
| `abort(reason)` | Halt training entirely | Unsatisfiable spec, persistent degeneration |

TransXform is NOT allowed to:
- Modify model architecture silently
- Change loss definitions without logging
- Commit illegal states
- Override hard invariants (even with human override)

---

## 8. Phase Model

Training progresses through explicit phases. Phases are condition-based, not time-based.

### 8.1 Default Phase Sequence

```
Bootstrap → Representation Formation → Stabilization → Refinement
     ↑                                                      │
     └──────────── Phase Regression (on persistent failure) ─┘
```

**Bootstrap:** Expect high variance, loose thresholds. The supervisor observes but intervenes only on catastrophic failure (NaN, explosion). Purpose: let the optimizer find initial structure.

**Representation Formation:** Enforce diversity. Monitor collapse aggressively. This is where most failures occur — the supervisor is most active here. Transition requires all hard invariants satisfied for T consecutive steps.

**Stabilization:** Tighten variance bounds. Reduce noise tolerance. The model should be learning useful structure, not just surviving.

**Refinement:** Freeze healthy components. Focus learning on remaining weak points. Reduced intervention — the supervisor is mostly auditing.

### 8.2 Phase Transitions

- Advance: all hard invariants satisfied for `transition_guard` consecutive steps.
- Regress: persistent hard invariant violation after `max_hard_interventions` per component.
- Halt: phase regression fails to resolve violation within one full regression cycle.

### 8.3 Phase-Specific Behavior

Different phases may have different thresholds for the same invariant. During bootstrap, a cosine similarity of 0.98 may be acceptable (representations haven't separated yet). During stabilization, 0.95 triggers intervention.

This is why **phases exist**: the same metric means different things at different training stages.

---

## 9. Training as Thermodynamics

TransXform treats representation diversity, signal energy, and gradient flow as conserved or bounded quantities. Collapse is an entropy sink. The supervisor enforces conservation laws and applies controlled energy injections to move the system out of bad attractors.

| Thermodynamic Analog | Training Equivalent | Supervisor Action |
|---|---|---|
| Conservation of energy | Signal magnitude preservation across layers | Rescale if magnitude drifts |
| Entropy increase (2nd law) | Representation diversity must not decrease below floor | Reinitialize if collapsed |
| Phase transition | Training regime shift | Explicit phase change with new thresholds |
| Heat death | All components converged to trivial solution | Abort with diagnostic |
| Controlled energy injection | Noise injection or reinitialization | Break degenerate equilibria |

This framing is physically inspired and operationally precise. The metrics (variance, entropy, mutual information) are well-defined information-theoretic quantities, and the supervisor enforces bounds on them using control-theoretic rules. No claim is made that SGD obeys thermodynamic laws in the strict physical sense — but the analogy reliably guides control design, failure diagnosis, and intuition about what "healthy" training dynamics look like. The value is engineering utility, not physical correspondence.

---

## 10. Boundary Ledger and Proof of Training Correctness

### 10.1 Ledger Structure

Every supervisor action is recorded:

```json
{
  "step": 14231,
  "phase": "representation_formation",
  "component": "emission_head",
  "invariant": "representation_diversity",
  "metric_snapshot": {
    "emission_pairwise_cos": 0.997,
    "emission_grad_norm": 0.0001,
    "contrastive_loss": 6.2
  },
  "action": "reinitialize",
  "justification": "Emission head in zero-gradient subspace. Pairwise cosine > 0.95 for 50 steps. Contrastive loss at log(N) plateau.",
  "outcome": "pending"
}
```

### 10.2 Training Health Certificate

At the end of training, TransXform emits a machine-verifiable certificate:

- **Invariant satisfaction history:** Per-invariant, per-phase compliance record
- **Intervention log:** Every action taken, with justification and outcome
- **Phase transition trace:** When and why each transition occurred
- **Final health summary:** Per-component health metrics at training completion
- **Verdict:** HEALTHY, RECOVERED (with intervention count), or COMPROMISED (with details)

This certificate attests that the model never entered — or was recovered from — degenerate states during training. It enables:
- Reproducibility audits
- Regulatory compliance for safety-critical deployments
- Trust that the model's learned representations are genuine, not artifacts of collapse

---

## 11. Failure Signature Registry

TransXform maintains a catalog of known failure modes with detection signatures and proven fixes. This is institutional knowledge, not autonomous learning.

### 11.1 Documented Failure Signatures

| Signature ID | Failure Mode | Detection Pattern | Proven Fix |
|---|---|---|---|
| `SIG-001` | Stability loss trivial convergence | Stability loss drops, representation variance collapses | Reformulate stability loss |
| `SIG-002` | Compressor attention collapse | comp_cos > 0.7 sustained | Global mean subtraction in compressor |
| `SIG-003` | Multi-task gradient conflict | One loss explodes on step 2 of training | Separate LR per parameter group |
| `SIG-004` | Recurrence erasing signal | H0→Hfinal cosine > 0.5 | Normalize state token magnitude to input scale |
| `SIG-005` | Emission head frozen | emission_cos ~ 1.0, contrastive grad ~ 0 | Reinitialize emission head |
| `SIG-006` | Dead attention heads | Attention entropy = 0, head grad norm = 0 | Reinitialize head, check head count |
| `SIG-007` | Loss explosion | Loss > 3x baseline within 10 steps | Reduce LR, check for NaN in gradients |

### 11.2 Registry as Reusable Infrastructure

Signatures are versioned, architecture-tagged, and composable into **Architecture Profiles** — reusable bundles of invariants, thresholds, phase definitions, and control laws for common architectures.

```yaml
profile: recurrent_transformer_v1
inherits: base_transformer
overrides:
  invariants:
    h0_hfinal_cosine_max: 0.5
    state_token_magnitude_ratio_max: 5.0
  signatures:
    - SIG-004  # recurrence erasing signal
    - SIG-005  # emission head frozen
```

---

## 12. Diagnostic Layer (V2): Advisory Signals

The V1 supervisor is reactive — it detects invariant violations and intervenes. The V2 diagnostic layer is predictive — it observes metric trends and warns before violations occur. The diagnostic layer is strictly **non-authoritative**: it never intervenes, never blocks phase transitions, and never modifies training state. It observes, and it advises.

### 12.1 Design Principles

- **Advisory only.** Diagnostics produce warnings, not actions. The supervisor's control laws remain the sole authority.
- **Calm language.** Warnings use "observed," "consistent with," "suggests" — never "detected" or "error."
- **Deduplication.** Each (signal, component) pair fires once. The user can acknowledge or resolve a warning to allow re-firing.
- **Phase-aware.** All history and active signals are cleared on phase transition to prevent cross-phase trend contamination.
- **Configurable.** Warmup period (default 100 steps), cadence (default every 10 steps), history window (default 50 snapshots), and per-signal thresholds are all configurable via `DiagnosticConfig`.

### 12.2 Signal Catalog

| # | Signal | Detects | Validated By |
|---|--------|---------|--------------|
| 1 | UnusedCapacity | Component with near-zero variance and/or gradient — not participating in forward pass | — |
| 2 | MissingStructuralSignal | Cosine and variance not moving — model is not learning structure | — |
| 3 | LossRepresentationMisalignment | Loss improving but representations stagnating — loss is hiding structural problems | FROG (early) |
| 4 | DynamicallyUnlearnableRegime | Loss plateau with pathological gradients (vanishing or oscillating) | — |
| 5 | ShortcutLearning | Variance explosion or collapse while loss improves — model exploiting a shortcut | FROG |
| 6 | MissingExpectedMetric | Spec declares component metrics that never appear in snapshots | — |
| 7 | LossStagnation | Loss plateau with healthy gradients — model trying but data signal-to-noise too low | MIRE |
| 8 | ThresholdDrift | Metric trending monotonically toward a hard threshold — predictive warning | — |
| 9 | MetricInstability | High coefficient of variation — metric oscillating rather than converging | CRUX |
| 10 | InterventionFutility | Repeated interventions producing only temporary recovery — structural problem | CRUX |
| 11 | GradientDomination | One component's gradients overwhelm others — monopolizing the optimizer | CRUX |
| 12 | MetricAnomaly | NaN or Inf in any metric — numerical corruption sentinel | — |
| 13 | TrainValDivergence | Training loss decreasing while validation loss increasing — overfitting | — |

**A note on Signal #5 (ShortcutLearning).** Variance explosion with simultaneous loss improvement is the canonical signature of shortcut learning — but it is also the signature of legitimate complex representation formation. When a model transitions from smooth, low-variance features to rich, high-variance features, the metric trajectory is indistinguishable from shortcut exploitation at the activation-variance level alone.

The current implementation mitigates this in three ways:

1. **Conservative confidence ceiling**: Signal #5 caps at 0.5 confidence, reflecting genuine ambiguity.
2. **Correlation with Signal #13 (TrainValDivergence)**: True shortcut learning typically degrades validation performance. If Signal #5 fires but validation loss continues improving, the shortcut hypothesis is weakened.
3. **Threshold calibration**: The default `shortcut_variance_explosion` threshold (100% increase) filters out moderate variance growth that is characteristic of healthy learning.

A more principled discrimination would require **rank analysis** of the representation matrix — shortcuts concentrate variance in a low-rank subspace, while legitimate learning distributes it. This is a Tier 2 metric not yet implemented (see §15.7).

### 12.3 Integration with V1

The diagnostic layer runs inside `supervisor.step()`, after the V1 invariant check and control laws. Warnings are recorded in the boundary ledger as `Advisory` entries and summarized in the training certificate and report.

The diagnostic layer receives the same `MetricSnapshot` as the V1 monitor but uses it differently:
- V1 checks each metric against its threshold independently, at each step.
- V2 accumulates a history window and detects *patterns* — trends, ratios, oscillation, divergence — that individual-step checks cannot see.

This separation is deliberate. V1 is the law. V2 is the weather forecast.

---

## 13. Explicit Non-Goals

TransXform does NOT:

- **Invent new loss functions.** The user defines losses. TransXform enforces invariants orthogonal to loss.
- **Guarantee SOTA performance.** It guarantees representational health, not task accuracy.
- **Replace optimizers.** Adam, SGD, LAMB, etc. all work as proposal mechanisms.
- **Automate architecture design.** The user designs the architecture. TransXform enforces the spec.
- **Remove the need for domain knowledge.** The user must know what invariants to declare. TransXform does not remove judgment — it removes ambiguity. Expert knowledge that previously lived in "intuition about when to restart" becomes a versioned, testable spec.
- **Learn autonomously (v1).** The supervisor executes static, declarative policies. Intervention outcomes are logged for offline analysis, but the supervisor does not update its own policies during training.

---

## 14. Negative Capability: Refusal to Train

TransXform must sometimes refuse to proceed. These are not crashes — they are **verdicts**.

| Verdict | Meaning |
|---|---|
| `UNSATISFIABLE_SPEC` | Declared invariants are mutually incompatible |
| `UNSTABLE_ARCHITECTURE` | Persistent violation across all phases despite interventions |
| `INSUFFICIENT_SIGNAL` | Gradients never exceed liveliness floor on any component |
| `DEGENERATE_OBJECTIVE` | Loss provides no usable gradient for any component |

A verdict is accompanied by full diagnostic state: which invariants failed, which interventions were attempted, and what the metric trajectories looked like.

This is philosophically aligned with EDGE and CLIFFORD: **honest refusal beats false success.**

---

## 15. Future Extensions

Some of these extensions have been partially realized in V2 (the diagnostic layer, §12). Others remain future work:

### 15.0 Validation Strategy: LOOM-First

The v1 implementation is validated against documented LOOM failure modes before generalization. LOOM is the reference architecture — TransXform is the generalization. Every failure signature in §11.1 was discovered during LOOM training; the v1 supervisor must detect and correct each one before the system is applied to other architectures. This ensures that TransXform's invariant catalog, control laws, and phase model are proven against real pathology, not theoretical scenarios.

### 15.1 Supervisor Policy Learning

In v1, policies are static and human-authored. All intervention outcomes are logged with pre-state metrics, action taken, and post-state recovery curves. These logs are structured to support offline analysis and eventual supervised policy updates. Authority must be stable before it can be optimized.

### 15.2 Architecture Profiles (Reusable Spec Packs)

Named, versioned bundles of invariants + thresholds + phases + control laws for common architectures. Profiles compose: `recurrent_transformer_v1` inherits `base_transformer` and overrides specific invariants. This turns TransXform into a platform with institutional memory.

### 15.3 Cross-Run Memory

A failure signature registry that accumulates across runs. Signatures are pattern matches over metric trajectories, advisory (not authoritative). The system gets better without online learning — it accumulates institutional knowledge.

### 15.4 Proof of Training Correctness (Cryptographic)

Given full audit logs, emit a cryptographically verifiable certificate attesting that the model never entered illegal states during training. This enables deployment in regulated industries.

### 15.5 Distributed / Multi-Node Training

TransXform runs at the trainer level, not inside the model. Metrics can be reduced across workers; authority decisions are centralized. Simpler than synchronizing optimizers.

### 15.6 Selective Shadow-Stepping

Post-commit authority (§2.1) has a known gap: adaptive optimizers retain momentum from pre-intervention states. **Selective shadow-stepping** runs a trial optimizer step, evaluates the resulting state against invariants, and only commits if it passes. This is expensive — it requires checkpointing and potentially rolling back the optimizer state — and is only justified for high-risk phases (e.g., phase transitions, post-reinit recovery). A hybrid approach applies shadow-stepping only when regret scores (§6.4) indicate recent interventions are failing.

### 15.7 Rank-Based Shortcut Discrimination

Signal #5 (ShortcutLearning) currently detects variance anomalies, which are necessary but not sufficient for shortcut identification (see §12.2 note). **Effective rank** of the representation matrix (computed via singular value entropy) can distinguish shortcut learning (low rank, variance concentrated in few dimensions) from legitimate complex features (high rank, variance distributed). This is a Tier 2 metric requiring SVD computation, feasible at cadence intervals but not every step.

### 15.8 Bootstrap Threshold Discovery

For novel architectures with no prior training history, the bootstrap phase can run in **observation-only mode** (§4.7), collecting metric distributions without enforcement. Post-observation, the system proposes thresholds from empirical percentiles. This converts the bootstrap from a "settling period" into a structured discovery phase, reducing the threshold authorship burden for new architectures from "know the answer" to "review a proposal."

---

## 16. Why This Is Not Overkill

### "Loss already tells us what's going on."

Loss is a scalar proxy for task performance. It contains no information about representational health. Collapse can reduce loss. Dead subspaces can reduce loss. Shortcut learning can reduce loss. Loss is insufficient by construction.

### "Just restart the run if it goes bad."

Restarts discard information. If only one component is broken, restarting wastes compute, destroys learned structure, and hides failure modes. Component-local intervention is cheaper and more reliable.

### "Isn't this too complex?"

Uncontrolled systems appear simple until they fail. Modern transformer training already has dozens of losses, hundreds of hyperparameters, and thousands of implicit assumptions. That complexity already exists. TransXform makes it legible.

### "What if the supervisor makes things worse?"

Then it does so loudly, reversibly, and auditably. Every intervention is logged with justification. Every hard intervention opens a regret window (§6.4) that tracks whether the intervention actually helped. Interventions can be replayed, disabled, or rolled back. Near-misses are recorded alongside actual violations. The system does not claim infallibility — it claims visibility. And visibility beats false stability.

### "Why hasn't this been done already?"

Most ML culture optimizes for single-run success, benchmark reporting, and cheap restarts. TransXform optimizes for robustness, repeatability, and engineering discipline. Those incentives rarely aligned — until training custom architectures became common enough that wasted compute matters.

### "Isn't this just control theory?"

Yes. That's the point. Transformer training is a high-dimensional dynamical system. Treating it as one *without* feedback control is negligent, not elegant.

---

## 17. Comparison to Existing Approaches

| Approach | What It Does | What TransXform Adds |
|---|---|---|
| **Early stopping** | Halts on validation loss plateau | TransXform detects *why* progress stopped, not just *that* it stopped |
| **Learning rate scheduling** | Reduces LR on plateau | TransXform adjusts LR per-component based on health, not schedule |
| **Gradient clipping** | Caps gradient magnitude | TransXform diagnoses the *cause* of gradient spikes |
| **Regularization** (dropout, weight decay) | Modifies gradients | TransXform enforces invariants orthogonal to gradient modification |
| **AutoML / HPO** | Searches hyperparameters offline | TransXform intervenes online during a single run |
| **Human dashboard monitoring** | Practitioner watches metrics | TransXform automates detection and intervention with audit trail |
| **Weights & Biases / TensorBoard** | Visualization | TransXform acts on metrics, not just displays them |

TransXform is not a replacement for any of these. It is the **authority layer** that sits above all of them, ensuring that no combination of optimizer + scheduler + regularization + loss can produce a silently degenerate model.

---

## 18. Implementation Skeleton

### 17.1 Core Objects

```rust
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Static, user-authored contract. Parsed once at startup.
#[derive(Debug, Deserialize)]
pub struct TrainingSpec {
    pub model_components: Vec<ComponentDecl>,
    pub invariants: HashMap<String, InvariantDecl>,  // hard and soft
    pub phases: Vec<PhaseDecl>,
    pub control: ControlConfig,
    pub metric_cadence: HashMap<String, u64>,
}

impl TrainingSpec {
    pub fn from_yaml(path: &str) -> Result<Self, TransXformError> {
        let contents = std::fs::read_to_string(path)?;
        Ok(serde_yaml::from_str(&contents)?)
    }
}

/// Authoritative controller. Owns the feedback loop.
pub struct Supervisor<M: Model> {
    spec: TrainingSpec,
    monitor: InvariantMonitor,
    phase_ctrl: PhaseController,
    executor: InterventionExecutor<M>,
    ledger: BoundaryLedger,
}

impl<M: Model> Supervisor<M> {
    pub fn new(spec: TrainingSpec, model: Rc<RefCell<M>>) -> Self {
        let monitor = InvariantMonitor::new(&spec.invariants);
        let phase_ctrl = PhaseController::new(&spec.phases);
        let executor = InterventionExecutor::new(model);
        let ledger = BoundaryLedger::new();
        Self { spec, monitor, phase_ctrl, executor, ledger }
    }

    pub fn step(&mut self, metrics: &HashMap<String, f64>, step: u64) -> Phase {
        let violations = self.monitor.check(metrics, self.phase_ctrl.current_phase());

        for v in &violations {
            match v.severity {
                Severity::Hard => {
                    let action = self.control_law(v);
                    self.executor.execute(&action);
                    self.ledger.record(step, v, &action);
                }
                Severity::Soft => {
                    let correction = self.soft_correction(v);
                    self.executor.execute(&correction);
                    self.ledger.record(step, v, &correction);
                }
            }
        }

        self.phase_ctrl.update(&violations, step);
        self.phase_ctrl.current_phase()
    }

    fn control_law(&self, violation: &Violation) -> Action { todo!() }
    fn soft_correction(&self, violation: &Violation) -> Action { todo!() }
}

/// Pure observability. Hooks into model, computes metrics.
pub struct MetricCollector;

impl MetricCollector {
    pub fn collect<M: Model>(
        &self,
        model: &M,
        loss_dict: &HashMap<String, f64>,
    ) -> HashMap<String, f64> {
        todo!()
    }
}

/// Finite state machine over training regimes.
pub struct PhaseController {
    phases: Vec<PhaseDecl>,
    current: Phase,
}

impl PhaseController {
    pub fn new(phases: &[PhaseDecl]) -> Self { todo!() }
    pub fn current_phase(&self) -> Phase { self.current }
    pub fn update(&mut self, violations: &[Violation], step: u64) { todo!() }
    pub fn can_advance(&self) -> bool { todo!() }
    pub fn should_regress(&self) -> bool { todo!() }
}

/// Actuator. Receives commands, executes against model/optimizer.
/// Uses Rc<RefCell<M>> for interior mutability — borrows are explicit and scoped,
/// consistent with TransXform's philosophy that illegal states cannot hide.
pub struct InterventionExecutor<M: Model> {
    model: Rc<RefCell<M>>,
}

impl<M: Model> InterventionExecutor<M> {
    pub fn new(model: Rc<RefCell<M>>) -> Self {
        Self { model }
    }
    pub fn execute(&mut self, action: &Action) {
        match action {
            Action::Reinitialize { component } => self.reinitialize(component),
            Action::Freeze { component }       => self.freeze(component),
            Action::Unfreeze { component }     => self.unfreeze(component),
            Action::Rescale { component, factor }     => self.rescale(component, *factor),
            Action::InjectNoise { component, magnitude } => self.inject_noise(component, *magnitude),
            Action::AdjustLr { component, factor }    => self.adjust_lr(component, *factor),
            Action::Abort { reason }           => self.abort(reason),
        }
    }

    fn reinitialize(&mut self, component: &str) { todo!() }
    fn freeze(&mut self, component: &str) { todo!() }
    fn unfreeze(&mut self, component: &str) { todo!() }
    fn rescale(&mut self, component: &str, factor: f64) { todo!() }
    fn inject_noise(&mut self, component: &str, magnitude: f64) { todo!() }
    fn adjust_lr(&mut self, component: &str, factor: f64) { todo!() }
    fn abort(&self, reason: &str) { todo!() }
}

/// Append-only audit log.
pub struct BoundaryLedger {
    entries: Vec<LedgerEntry>,
}

impl BoundaryLedger {
    pub fn new() -> Self { Self { entries: Vec::new() } }
    pub fn record(&mut self, step: u64, violation: &Violation, action: &Action) { todo!() }
    pub fn emit_certificate(&self) -> TrainingCertificate { todo!() }
    pub fn last_entry(&self) -> Option<&LedgerEntry> { self.entries.last() }
}

// --- Supporting types ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity { Hard, Soft }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase { Warmup, Active, Cooldown, Aborted }

pub struct Violation {
    pub invariant_name: String,
    pub severity: Severity,
    pub observed: f64,
    pub threshold: f64,
}

pub enum Action {
    Reinitialize { component: String },
    Freeze { component: String },
    Unfreeze { component: String },
    Rescale { component: String, factor: f64 },
    InjectNoise { component: String, magnitude: f64 },
    AdjustLr { component: String, factor: f64 },
    Abort { reason: String },
}

pub struct LedgerEntry {
    pub step: u64,
    pub violation: Violation,
    pub action: Action,
    pub timestamp: std::time::Instant,
}

pub struct TrainingCertificate { /* ... */ }

pub trait Model {
    fn parameters(&self) -> &[tch::Tensor];
    fn component(&self, name: &str) -> Option<&dyn std::any::Any>;
    fn component_mut(&mut self, name: &str) -> Option<&mut dyn std::any::Any>;
}
```

### 17.2 Training Loop Integration

```rust
fn main() -> Result<(), TransXformError> {
    let spec = TrainingSpec::from_yaml("training_spec.yaml")?;
    let model = Rc::new(RefCell::new(build_model(&spec.model_components)?));
    let mut optimizer = AdamW::new(model.borrow().parameters(), spec.control.base_lr);
    let mut supervisor = Supervisor::new(spec, model.clone());
    let collector = MetricCollector;

    for step in 0..max_steps {
        // 1. Optimizer commits update (non-authoritative)
        let loss = forward_backward(&mut model.borrow_mut(), &batch);
        optimizer.step();

        // 2. Supervisor observes, evaluates, corrects (authoritative)
        let metrics = collector.collect(&model.borrow(), &loss);
        let phase = supervisor.step(&metrics, step);

        // Check for abort
        if phase == Phase::Aborted {
            if let Some(entry) = supervisor.ledger.last_entry() {
                eprintln!("Training aborted at step {}: {:?}", step, entry);
            }
            break;
        }
    }

    // Emit certificate
    let cert = supervisor.ledger.emit_certificate();
    cert.save("training_certificate.json")?;
    Ok(())
}
```

---

## 19. Training Integrity: Why Correction Is Not Prevention

### 19.1 Late Correction Is Structural Injury

Every intervention — every reinit, every LR halving, every forced rescale — changes the internal geometry of the model. The optimizer may recover loss, but the optimization trajectory has been permanently altered. The model lands in a different basin. It generalizes differently. It behaves differently under distribution shift.

This is not speculation. Recent work on training dynamics establishes that:

- **Training stability is a system property, not a loss artifact.** Small perturbations to optimization state can induce abrupt, irreversible collapse, and stability is a distinct dimension from final performance (Zhang et al., 2026).
- **Gradient and loss spikes disrupt the learning process.** Large spikes destabilize optimization and commonly force manual restarts (Huang et al., 2025 — SPAM).
- **Reinitialization changes the solution geometry.** Reinitializing subsets of parameters during training biases convergence toward different minima with different generalization properties (Alabdulmohsin et al., 2021; Zaidi et al., 2023).
- **The training path matters, not just the endpoint.** Techniques like Stochastic Weight Averaging demonstrate that trajectory shape affects the width and quality of the final optimum (Foret et al., 2020 — SAM).

The implication is clear: **a model that required intervention is not the same as a model that never needed it.** Even when the intervention "works" — loss recovers, metrics return to healthy ranges — the model carries a scar. The optimizer remembers the trauma even if the dashboard forgets it.

### 19.2 The Trichotomy

TransXform produces three verdicts, and they are not equal:

| Verdict | Meaning | What It Says About the Model |
|---------|---------|------|
| **CLEAN** | Zero interventions throughout training | The model was never injured. The governance prevented illegal states through timing alone. |
| **RECOVERED** | Interventions were required but succeeded | The model was injured and healed. The scar is acknowledged in the boundary ledger. |
| **ABORTED** | Interventions exhausted without recovery | The architecture is structurally broken. Honest refusal beats false success. |

This trichotomy matters enormously once you care about long-horizon behavior, stability under distribution shift, interpretability, downstream fine-tuning, or safety guarantees that aren't vibes-based.

A CLEAN verdict means the readiness gate held the model until it was ready, the phase transitions happened at the right time, and the thresholds were never crossed. No reinit. No LR cascade. No forced correction. The model learned on its own schedule, within its declared boundaries.

That distinction — between RECOVERED and CLEAN — is the reason TransXform becomes a precondition rather than an add-on. Like memory safety, determinism, reproducibility, and audit logs: not optional, not a feature, a requirement.

### 19.3 From Supervision to Integrity

**TransXform is not a system for fixing broken training runs. It is a system for ensuring that training never breaks in the first place.**

The diagnostic layer (§12) predicts problems before they become violations. The readiness gate prevents premature phase transitions. The phase model applies the right thresholds at the right time. The V1 control laws are the last line of defense — and the goal is for them to never fire.

This reframes what TransXform is. It is not training supervision. It is **training integrity** — the structural guarantee that the optimization trajectory was never perturbed, the representations were never corrupted, and the model was never forced to recover from something that should not have happened.

---

## 20. Conclusion

TransXform doesn't make training smarter. It makes failure illegal — and prevention preferable to cure.

By enforcing invariants, modeling phases, predicting violations, and granting authority to a supervisor instead of loss curves, it becomes structurally impossible for a model to fail silently. Training stops being an art practiced by the lucky few and becomes ordinary engineering practiced by anyone who can write a spec.

The core insight is simple: **training is a dynamical system, and dynamical systems without feedback control are not engineered systems — they are experiments.** TransXform adds the feedback control. The diagnostic layer adds the weather forecast. The readiness gate adds the timing. Everything else follows.

A model trained under TransXform with a CLEAN verdict isn't just successful. It's unscarred.

---

## Appendix A: Glossary

| Term | Definition |
|---|---|
| **Invariant** | A declared constraint that must hold during training. Violations trigger intervention. |
| **Hard boundary** | An invariant whose violation triggers immediate intervention. |
| **Soft boundary** | An invariant whose violation triggers gradual correction. |
| **Phase** | An explicit training regime with phase-specific thresholds and allowed interventions. |
| **Intervention** | A component-local action taken by the supervisor to restore invariant compliance. |
| **Boundary ledger** | Append-only audit log of all invariant evaluations and interventions. |
| **Training spec** | User-authored declarative contract defining model components, invariants, phases, and control laws. |
| **Training certificate** | Machine-verifiable attestation that all invariants held (or were recovered) during training. |
| **Provisional update** | A gradient-driven state change committed by the optimizer. Non-authoritative; subject to post-commit correction. |
| **Authority** | The supervisor's power to accept, correct, or reverse committed updates. |
| **Architecture profile** | A reusable bundle of invariants, thresholds, phases, and control laws for a common architecture. |
| **Failure signature** | A documented pattern of metric trajectories that indicates a specific failure mode. |
| **Intervention regret** | Post-hoc assessment of whether an intervention was necessary, based on recovery speed and counterfactual trajectory analysis. |
| **Near-miss** | A step where a soft invariant approached but did not cross a hard threshold. Logged for threshold calibration. |
| **Threshold provenance** | The documented origin of a threshold value: prior runs, architecture profiles, failure signatures, or theoretical bounds. |
| **Diagnostic signal** | An advisory observation about metric trends that may indicate a developing problem. Non-authoritative — never triggers intervention. |
| **Diagnostic layer** | The V2 subsystem that observes metric history and emits advisory warnings. Runs after V1 invariant checks. |
| **Readiness gate** | V1.3 mechanism that blocks phase transitions until the model can satisfy the next phase's thresholds. Prevents cliff transitions. |
| **Checkpoint** | Serialized snapshot of all supervisor runtime state. Enables training resumption with identical governance behavior. |
| **Training integrity** | The structural guarantee that the optimization trajectory was never perturbed by forced corrections. A CLEAN verdict attests to training integrity. |
| **Structural injury** | The lasting geometric change to a model's parameter space caused by a forced intervention (reinit, LR cascade, rescale). Recoverable in loss, not in trajectory. |

## Appendix B: Relationship to EDGE

TransXform borrows its authority model directly from EDGE (Explicit Delimitation of Generative Epistemics):

| EDGE Concept | TransXform Equivalent |
|---|---|
| Boundary Ledger | Training Boundary Ledger |
| Hard boundary | Hard training invariant |
| Soft boundary | Soft training invariant |
| Probe | Training step |
| Generator (non-authoritative) | Optimizer/gradients (non-authoritative) |
| E0 kernel (authoritative) | Supervisor (authoritative) |
| Witness DAG | Intervention audit record |
| Licensing (temporary relaxation) | Phase-specific threshold relaxation |
| Boundary hardening (learning) | Failure signature accumulation |

The philosophical alignment is exact: EDGE says "no reasoning step may commit without boundary approval." TransXform says "no gradient step may persist without invariant approval." Same principle — authority over state transitions — applied at different phases of the system lifecycle.

## Appendix C: References

The following works support the claim in §19 that mid-training corrections alter model geometry in lasting, non-neutral ways:

1. Zhang, Z. et al., "Training instability in deep learning follows low-dimensional dynamical principles," 2026. Establishes that training stability is a system property distinct from final performance, and that small perturbations to optimization state can induce abrupt, irreversible collapse.

2. Huang, T. et al., "SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training," 2025. Documents that gradient and loss spikes during large model training destabilize optimization and commonly force manual restarts, motivating spike-aware optimizers.

3. Alabdulmohsin, I. et al., "The Impact of Reinitialization on Generalization in Convolutional Neural Networks," 2021. Evidence that reinitializing subsets of parameters during training biases convergence toward different minima with different generalization properties.

4. Zaidi, S. et al., "When Does Re-initialization Work?," 2023. Empirical study showing that periodic reinitialization interacts with hyperparameters in complex ways and is not a neutral operation — corrective resets materially alter training behavior.

5. Plusch, G. et al., "The Weights Reset Technique for Deep Neural Networks," 2023. Broader framework for understanding how weight modifications during training change generalization behavior.

6. Foret, P. et al., "Sharpness-Aware Minimization (SAM) for Efficiently Improving Generalization," 2020. Demonstrates that trajectory shape affects the width and quality of the final optimum, implying that perturbations along the path change the basin of convergence.
