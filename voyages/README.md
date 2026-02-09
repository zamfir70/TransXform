# TransXform Voyage Logs

Evidence of TransXform supervising real transformer training runs.

## Runs

| Run | Model | Steps | Outcome | Key Feature Tested |
|-----|-------|-------|---------|--------------------|
| 7 | SemanticNormalizer 350M | ~300 (died) | Death spiral: cliff transition → reinit → LR cascade | V1.2 (no readiness gate) |
| **8** | SemanticNormalizer 350M | 910+ (running) | **Zero violations, zero interventions** | V1.3 readiness gate + V1.4 checkpointing |
| **ICFU** | ICFU 1.6M on 30K AEONCORE traces | 76,647 (complete) | **HEALTHY — 7 violations caught, 3 LR interventions, full recovery** | V1.0–V1.4 + V2 diagnostics |
| **CRUX** | CRUX 2.1M bottleneck (192→32) | 221 (aborted) | **ABORTED — 29 hard violations, 12 interventions exhausted, correct abort** | V1.0–V1.4 negative capabilities, V2.3 validation (Signals 9+10) |
| **FROG** | FROG 525K slow poison | 27,024 (complete) | **HEALTHY — 0 hard violations, 3 diagnostic advisories, shortcut detected at step 7,900** | V2.1 diagnostic layer (variance explosion + metric key fix) |
| **MIRE** | MIRE 525K label noise | 33,780 (complete) | **HEALTHY — 0 hard violations, 1 diagnostic advisory, stagnation detected at step 2,400** | V2.2 LossStagnation signal |

## The Stories

### Runs 7 & 8: Same Model, Different Governance

Run 7 and Run 8 used the **same model**, the **same data**, and the **same learning rate**. The only difference was TransXform's governance timing.

Run 7 allowed a phase transition at step 199 when cosine similarity was 0.9975 — above the next phase's 0.98 threshold. Immediate hard violation. Reinit. Loss explosion. LR halving cascade. Death.

Run 8 held the model in bootstrap for 300 extra steps via the readiness gate. When patience expired, it adaptively relaxed the threshold from 0.98 to 0.9996. The model transitioned safely at step 500 with zero violations. By step 700, cosine had naturally dropped below 0.98 — the model did the work on its own schedule.

> "You didn't change the model. We changed the timing of pressure."

### ICFU: Epoch-Boundary Instability Discovery

The ICFU run on 30,000 real AEONCORE reasoning traces (817K samples) uncovered a previously undocumented training pathology: **epoch-boundary representation collapse**.

At the exact step where epochs end and data reshuffles, three components simultaneously spiked to near-identical representations (cosine 0.976–0.999). This happened at both epoch boundaries — steps 25,548 and 51,097 — with the same pattern: intent_head collapses first, encoder follows, then a third component (affect or focus). The model recovered within ~100 steps each time.

TransXform caught every instance without being programmed to look for it. The same model trained on synthetic data produced zero violations — confirming TransXform only fires when real pathology exists.

> The synthetic run proves no false positives. The AEONCORE run proves it catches real problems.

TransXform's value isn't in making dramatic interventions. It's in quietly removing entire classes of failure before they become events.

### CRUX: Proving the Abort Path

CRUX was designed to be *unsalvageable* — a 192→32 bottleneck with a 400:1 gradient magnitude ratio between BoW reconstruction (2048 outputs) and category classification (5 outputs), plus contrastive collapse pressure. The question wasn't "will TransXform catch violations?" but "will it correctly decide to give up?"

TransXform fired 12 hard interventions across 221 steps — reinitializing the encoder, compressor, contrastive head, and category head multiple times. None worked. The contrastive head sat at cosine 0.9994+ from step 1 to abort. The category head hit **1.000000** by step 144. The 32-dim bottleneck couldn't carry enough information for three competing objectives, and the BoW head's massive gradient magnitude drowned the smaller heads.

At step 221, TransXform exhausted `max_hard_interventions: 12` and transitioned to `Aborted`. This is the correct behavior — the architecture is genuinely broken, and continuing to reinitialize would waste compute without recovery.

**V2.3 validation re-run** added 3 new signals to this story: MetricInstability (Signal 9) caught compressor oscillation (CV=4.66) and contrastive gradient instability (CV=0.68) at step 50. InterventionFutility (Signal 10) caught the chronic reinit-collapse cycle at step 145 — 76 steps before V1's abort, flagging that "even successful interventions are not producing lasting improvement." The original V1 run took 221 steps to reach the same conclusion. Signal 10 now gives early warning that the architecture is structurally broken, not just transiently unstable.

> Three stress tests. Three outcomes. HEALTHY recovery from real pathology (ICFU). Zero-violation governance via timing (SemanticNormalizer). Correct abort of an unsalvageable architecture (CRUX). TransXform handles all three.

### FROG: The Diagnostic Layer Earns Its Keep

FROG was designed to be invisible to V1. A class-specific unit-normalized vector injected at position 0 of every embedding, growing linearly from strength 0 to 5.0 over 15,000 steps. The model learns normally at first, then gradually discovers it can exploit the shortcut instead of learning real content features. Loss crashes to zero. Encoder variance explodes to 57x baseline. Cosine similarity actually *improves* (drops further from threshold). Every hard invariant stays satisfied.

V1 saw nothing wrong. Zero violations. Zero interventions. A clean bill of health for a model that had memorized a trivial feature.

V2 caught it. At step 7,900, with leak strength 2.63, the diagnostic layer flagged ShortcutLearning: "Activation variance in encoder increased 63% while loss improved 37.7%. This rapid variance growth while loss improves suggests the model is amplifying a low-dimensional shortcut feature rather than learning distributed representations."

Getting there took three runs. Run 1 found that shortcut learning can manifest as variance *explosion*, not just collapse — the original detection only looked down, not up. Run 2 found a silent metric key mismatch (`activation_variance` vs `activation_variance_min`) that disabled variance analysis in three signals without any error or warning. Run 3, with both fixes, detected the poison 2,600 steps earlier than Run 2.

> Four stress tests. Four outcomes. ICFU: catch real pathology. SemanticNormalizer: prevent it with timing. CRUX: abort the unsalvageable. FROG: detect what hard invariants cannot. The diagnostic layer isn't optional — it's the only thing that caught the slow poison.

### MIRE: The Silent Plateau

MIRE is a clean model on noisy data. Same 525K architecture as FROG — no poison, no structural sabotage, no bottleneck. The only corruption is progressive label noise: starting at 0%, ramping to 90% over 15,000 steps. After step 15,000, 9 out of 10 labels are random.

The model learned fast — loss dropped to 0.58 by step 311 with only 2% noise. Then the noise took over. Loss climbed to ~1.60 (random chance for 5 classes) and stayed there for 20,000 more steps. Cosine similarity stayed healthy (0.07–0.42). Variance decreased gradually (6.68 to 1.25). Gradients flowed normally (0.04–0.98). Every structural invariant was satisfied at every step. V1 saw a perfectly healthy run.

Signals 1-6 had nothing to report. Signal 4 (DynamicallyUnlearnableRegime) specifically didn't fire because it requires pathological gradients — vanishing or oscillating. MIRE's gradients were healthy. The model was actively trying to learn. The labels were just noise.

Signal 7 (LossStagnation) fired at step 2,400: "Loss has stagnated at 0.5796 for 2,089 steps despite healthy gradient flow. The data signal-to-noise ratio may be too low." It correctly identified both the symptom (loss plateau with active gradients) and the likely cause (data quality).

> Five stress tests. Five outcomes. Thirteen diagnostic signals. ICFU: catch real pathology. SemanticNormalizer: prevent it with timing. CRUX: abort the unsalvageable — and predict the abort 76 steps early. FROG: detect shortcut learning. MIRE: detect stagnation. V2.4 closes the remaining gaps: gradient domination (the root cause behind CRUX's 400:1 ratio), NaN/Inf sentinels (the silent corruptor), and train/val divergence (the most common real-world failure). The bowling alley bumpers are up — if you don't throw the ball over them, it gets to the end.

## Voyage Logs

- [run8_semantic_normalizer.md](run8_semantic_normalizer.md) — SemanticNormalizer 350M: readiness gate prevents cliff transition death spiral
- [icfu_aeoncore_30k.md](icfu_aeoncore_30k.md) — ICFU 1.6M on 30K AEONCORE traces: epoch-boundary collapse detection, graduated LR response
- [crux_bottleneck_abort.md](crux_bottleneck_abort.md) — CRUX 2.1M bottleneck: correct abort of unsalvageable architecture
- [frog_slow_poison.md](frog_slow_poison.md) — FROG 525K slow poison: V2 diagnostic catches shortcut learning invisible to V1
- [mire_label_noise.md](mire_label_noise.md) — MIRE 525K label noise: V2.2 LossStagnation catches loss plateau invisible to V1 and Signals 1-6

## All Project Documentation

**In this repo (`D:\TransXform\`):**

| File | What |
|------|------|
| `TransXform_Whitepaper.md` | Theory, architecture, spec format (v1.1) |
| `voyages/` (this folder) | Evidence of real training runs — results + narratives |
| `src/` | 24 source files (lib.rs is the entry point) |
| `tests/integration.rs` | 15 integration tests |
| `examples/basic_training.rs` | Full supervisor loop + checkpoint hints demo |

**Stress test projects (separate repos, each depends on transxform via path dep):**

| Repo | Test | Params | Spec |
|------|------|--------|------|
| `D:\Hydra\` | #1: Multi-head classifier | ~2M | `specs/hydra_spec.yaml` |
| `D:\ICFU\` | #2: Set transformer on AEONCORE data | ~1.6M | `specs/icfu_spec.yaml` |
| `D:\CRUX\` | #3: Bottleneck + competing heads | ~2.12M | `specs/crux_spec.yaml` |
| `D:\FROG\` | #4: Slow poison via label leak | ~525K | `specs/frog_spec.yaml` |
| `D:\MIRE\` | #5: Progressive label noise | ~525K | `specs/mire_spec.yaml` |

**External:**

| File | What |
|------|------|
| `D:\LOOM failures so far.txt` | 13 real-world training failures (V1.2 audit source) |
