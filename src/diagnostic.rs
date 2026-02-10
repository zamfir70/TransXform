//! TransXform V2 — Epistemic Early Warning Layer
//!
//! Advisory-only diagnostics that surface evidence of potential problems
//! without intervening. The supervisor (V1) enforces structural health;
//! the diagnostic layer (V2) asks: "Are you training the right thing?"
//!
//! # Design Principles
//!
//! - **Non-authoritative.** No interventions. No auto-fixes.
//! - **Evidence-based.** Every warning cites specific metric patterns.
//! - **Calm, precise language.** "Observed," "consistent with," "suggests."
//! - **Opinionated but honest.** Speaks clearly about what the evidence
//!   indicates, but never claims certainty.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::types::*;

fn default_shortcut_variance_explosion() -> f64 {
    1.0
}
fn default_stagnation_patience_steps() -> u64 {
    2000
}
fn default_stagnation_improvement_threshold() -> f64 {
    0.01
}
fn default_stagnation_grad_floor() -> f64 {
    1e-5
}
fn default_drift_window_steps() -> usize {
    1000
}
fn default_drift_crossing_horizon() -> u64 {
    2000
}
fn default_drift_monotonic_pct() -> f64 {
    0.8
}
fn default_instability_cv_threshold() -> f64 {
    0.3
}
fn default_futility_lookback_interventions() -> usize {
    3
}
fn default_futility_min_recovery_steps() -> Option<u64> {
    None
}
fn default_gradient_domination_ratio() -> f64 {
    100.0
}
fn default_overfit_min_divergence() -> f64 {
    0.05
}
fn default_shortcut_rank_threshold() -> f64 {
    0.3
}

/// Configuration for the diagnostic layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticConfig {
    /// Minimum steps before diagnostics activate (let training warm up).
    pub warmup_steps: u64,
    /// How often to run full diagnostics (every N steps).
    pub cadence: u64,
    /// How many recent metric snapshots to retain for trend analysis.
    pub history_window: usize,
    /// Minimum confidence threshold to emit a warning.
    pub min_confidence: f64,

    // --- Signal-specific thresholds ---

    /// Activation variance below this is considered "near-zero" for unused capacity.
    pub unused_capacity_variance_floor: f64,
    /// Grad norm below this is considered "near-zero" for unused capacity.
    pub unused_capacity_grad_floor: f64,
    /// Fraction of window steps that must show near-zero to trigger.
    pub unused_capacity_persistence: f64,

    /// Entropy below this suggests collapsed/uniform attention.
    pub structural_signal_entropy_floor: f64,

    /// Loss must improve by at least this fraction to count as "decreasing."
    pub alignment_loss_improvement_threshold: f64,
    /// Representation metric change below this is "stagnant."
    pub alignment_repr_stagnation_threshold: f64,

    /// Grad norm below this relative to param scale suggests unlearnable regime.
    pub unlearnable_grad_ratio_floor: f64,
    /// Grad norm above this suggests instability.
    pub unlearnable_grad_ratio_ceiling: f64,

    /// Cosine similarity increase over window that suggests shortcut learning.
    pub shortcut_cosine_drift: f64,
    /// Variance decrease over window that suggests shortcut learning.
    pub shortcut_variance_drift: f64,
    /// Variance increase (explosion) over window that suggests shortcut
    /// learning via feature amplification. When a model discovers a shortcut
    /// signal, it may amplify that feature, causing variance to explode
    /// rather than collapse. Default 1.0 = 100% increase.
    #[serde(default = "default_shortcut_variance_explosion")]
    pub shortcut_variance_explosion: f64,

    /// How many steps without loss improvement before flagging stagnation.
    #[serde(default = "default_stagnation_patience_steps")]
    pub stagnation_patience_steps: u64,
    /// Minimum fractional improvement to count as "learning" (default 0.01 = 1%).
    #[serde(default = "default_stagnation_improvement_threshold")]
    pub stagnation_improvement_threshold: f64,
    /// Grad norm above this floor indicates the model is still trying
    /// (not converged). Below this, the plateau is likely convergence,
    /// not stagnation.
    #[serde(default = "default_stagnation_grad_floor")]
    pub stagnation_grad_floor: f64,

    /// Minimum number of history data points for drift trend analysis.
    #[serde(default = "default_drift_window_steps")]
    pub drift_window_steps: usize,
    /// Fire if linear extrapolation predicts threshold crossing within this many steps.
    #[serde(default = "default_drift_crossing_horizon")]
    pub drift_crossing_horizon: u64,
    /// Fraction of consecutive pairs that must trend toward threshold to qualify as monotonic.
    #[serde(default = "default_drift_monotonic_pct")]
    pub drift_monotonic_pct: f64,

    /// Coefficient of variation above this triggers MetricInstability.
    #[serde(default = "default_instability_cv_threshold")]
    pub instability_cv_threshold: f64,

    /// Number of recent interventions per component to check for futility.
    #[serde(default = "default_futility_lookback_interventions")]
    pub futility_lookback_interventions: usize,
    /// If set, interventions that don't recover within this many steps are considered failed.
    #[serde(default = "default_futility_min_recovery_steps")]
    pub futility_min_recovery_steps: Option<u64>,

    /// Max/min gradient norm ratio across components that triggers domination warning.
    #[serde(default = "default_gradient_domination_ratio")]
    pub gradient_domination_ratio: f64,
    /// Minimum relative divergence between train and val loss to trigger overfitting warning.
    #[serde(default = "default_overfit_min_divergence")]
    pub overfit_min_divergence: f64,

    /// Rank threshold for shortcut discrimination. When `{component}.effective_rank`
    /// drops below this fraction of its initial value while variance explodes,
    /// confidence in shortcut learning is boosted. When rank is stable or growing,
    /// the signal is suppressed. Default: 0.3 (rank below 30% of initial).
    #[serde(default = "default_shortcut_rank_threshold")]
    pub shortcut_rank_threshold: f64,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        Self {
            warmup_steps: 100,
            cadence: 10,
            history_window: 50,
            min_confidence: 0.3,

            unused_capacity_variance_floor: 1e-5,
            unused_capacity_grad_floor: 1e-6,
            unused_capacity_persistence: 0.8,

            structural_signal_entropy_floor: 0.2,

            alignment_loss_improvement_threshold: 0.02,
            alignment_repr_stagnation_threshold: 0.005,

            unlearnable_grad_ratio_floor: 1e-7,
            unlearnable_grad_ratio_ceiling: 1e3,

            shortcut_cosine_drift: 0.05,
            shortcut_variance_drift: 0.3,
            shortcut_variance_explosion: 1.0,

            stagnation_patience_steps: 2000,
            stagnation_improvement_threshold: 0.01,
            stagnation_grad_floor: 1e-5,

            drift_window_steps: 1000,
            drift_crossing_horizon: 2000,
            drift_monotonic_pct: 0.8,

            instability_cv_threshold: 0.3,

            futility_lookback_interventions: 3,
            futility_min_recovery_steps: None,

            gradient_domination_ratio: 100.0,
            overfit_min_divergence: 0.05,
            shortcut_rank_threshold: 0.3,
        }
    }
}

/// Record of an intervention's outcome — fed to the diagnostic layer by the
/// supervisor after regret assessment. Used for InterventionFutility detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionOutcomeRecord {
    pub step: u64,
    pub component: String,
    pub action: Action,
    pub recovered: bool,
    pub recovery_steps: Option<u64>,
}

/// The V2 diagnostic layer. Observes training dynamics and surfaces
/// advisory warnings when patterns are inconsistent with stated intent.
///
/// Does not intervene. Does not modify model state. Does not claim certainty.
pub struct DiagnosticLayer {
    /// Public config — users can override via `supervisor.diagnostic_mut().config = ...`.
    pub config: DiagnosticConfig,
    /// Rolling window of recent metric snapshots.
    history: VecDeque<MetricSnapshot>,
    /// Component names from the training spec.
    components: Vec<String>,
    /// All warnings emitted during the run (including resolved ones).
    warnings: Vec<DiagnosticWarning>,
    /// Active (unresolved) warning signals — used for deduplication.
    active_signals: Vec<(DiagnosticSignal, String)>, // (signal, component_or_global)
    /// Metric keys that invariants depend on — if these never appear in
    /// snapshots, the corresponding invariants are silently inactive.
    expected_metrics: Vec<String>,
    /// Best loss seen in the current phase (for stagnation detection).
    best_loss: Option<f64>,
    /// Step at which best_loss was achieved.
    best_loss_step: u64,
    /// Invariant thresholds + directions — needed for drift prediction.
    invariant_thresholds: HashMap<String, (f64, ThresholdDirection)>,
    /// Recent intervention outcomes — needed for futility detection.
    intervention_outcomes: Vec<InterventionOutcomeRecord>,
}

impl DiagnosticLayer {
    /// Create a new diagnostic layer with the given config and component list.
    pub fn new(config: DiagnosticConfig, components: Vec<String>) -> Self {
        Self {
            config,
            history: VecDeque::new(),
            components,
            warnings: Vec::new(),
            active_signals: Vec::new(),
            expected_metrics: Vec::new(),
            best_loss: None,
            best_loss_step: 0,
            invariant_thresholds: HashMap::new(),
            intervention_outcomes: Vec::new(),
        }
    }

    /// Set the metric keys that invariants depend on. When these keys are
    /// absent from metric snapshots for the full history window, the
    /// diagnostic layer emits a MissingExpectedMetric warning.
    pub fn set_expected_metrics(&mut self, keys: Vec<String>) {
        self.expected_metrics = keys;
    }

    /// Set invariant thresholds for drift detection. Called from the supervisor
    /// at init time, same pattern as `set_expected_metrics()`.
    pub fn set_invariant_thresholds(
        &mut self,
        thresholds: HashMap<String, (f64, ThresholdDirection)>,
    ) {
        self.invariant_thresholds = thresholds;
    }

    /// Record an intervention outcome from the regret tracker. Called by the
    /// supervisor after regret assessment completes (Confident or LowConfidence).
    pub fn record_intervention_outcome(&mut self, record: InterventionOutcomeRecord) {
        self.intervention_outcomes.push(record);
    }

    /// Extract state for checkpointing.
    pub fn save_state(&self) -> crate::checkpoint::DiagnosticState {
        crate::checkpoint::DiagnosticState {
            config: self.config.clone(),
            history: self.history.clone(),
            components: self.components.clone(),
            warnings: self.warnings.clone(),
            active_signals: self.active_signals.clone(),
            expected_metrics: self.expected_metrics.clone(),
            best_loss: self.best_loss,
            best_loss_step: self.best_loss_step,
            invariant_thresholds: self.invariant_thresholds.clone(),
            intervention_outcomes: self.intervention_outcomes.clone(),
        }
    }

    /// Restore state from a checkpoint.
    pub fn restore_state(&mut self, state: crate::checkpoint::DiagnosticState) {
        self.config = state.config;
        self.history = state.history;
        self.components = state.components;
        self.warnings = state.warnings;
        self.active_signals = state.active_signals;
        self.expected_metrics = state.expected_metrics;
        self.best_loss = state.best_loss;
        self.best_loss_step = state.best_loss_step;
        self.invariant_thresholds = state.invariant_thresholds;
        self.intervention_outcomes = state.intervention_outcomes;
    }

    /// Run diagnostics for a single step. Returns new warnings (if any).
    ///
    /// This is the main entry point, called from the supervisor after metric
    /// collection. It stores the metrics, then — if past warmup and on cadence —
    /// runs all five signal detectors.
    pub fn diagnose(&mut self, step: u64, metrics: &MetricSnapshot) -> Vec<DiagnosticWarning> {
        // Store in history
        self.history.push_back(metrics.clone());
        if self.history.len() > self.config.history_window {
            self.history.pop_front();
        }

        // Track best loss for stagnation detection (every step, not just cadence)
        if let Some(&loss) = metrics.get("loss") {
            let threshold = self.config.stagnation_improvement_threshold;
            match self.best_loss {
                None => {
                    self.best_loss = Some(loss);
                    self.best_loss_step = step;
                }
                Some(best) if loss < best * (1.0 - threshold) => {
                    self.best_loss = Some(loss);
                    self.best_loss_step = step;
                }
                _ => {}
            }
        }

        // Don't diagnose during warmup or off-cadence
        if step < self.config.warmup_steps {
            return Vec::new();
        }
        if step % self.config.cadence != 0 {
            return Vec::new();
        }
        // Need enough history for trend analysis
        if self.history.len() < 10 {
            return Vec::new();
        }

        let mut new_warnings = Vec::new();

        // Run all thirteen detectors
        self.detect_unused_capacity(step, metrics, &mut new_warnings);
        self.detect_missing_structural_signal(step, metrics, &mut new_warnings);
        self.detect_loss_representation_misalignment(step, &mut new_warnings);
        self.detect_unlearnable_regime(step, metrics, &mut new_warnings);
        self.detect_shortcut_learning(step, &mut new_warnings);
        self.detect_loss_stagnation(step, metrics, &mut new_warnings);
        self.detect_missing_expected_metrics(step, &mut new_warnings);
        self.detect_threshold_drift(step, metrics, &mut new_warnings);
        self.detect_metric_instability(step, &mut new_warnings);
        self.detect_intervention_futility(step, &mut new_warnings);
        self.detect_gradient_domination(step, metrics, &mut new_warnings);
        self.detect_metric_anomaly(step, metrics, &mut new_warnings);
        self.detect_train_val_divergence(step, &mut new_warnings);

        // Filter by confidence threshold and dedup against active signals
        let new_warnings: Vec<DiagnosticWarning> = new_warnings
            .into_iter()
            .filter(|w| w.confidence >= self.config.min_confidence)
            .filter(|w| {
                let key = (w.signal, self.warning_key(w));
                !self.active_signals.iter().any(|k| k.0 == key.0 && k.1 == key.1)
            })
            .collect();

        // Register new active signals and store warnings
        for w in &new_warnings {
            self.active_signals.push((w.signal, self.warning_key(w)));
            self.warnings.push(w.clone());
        }

        new_warnings
    }

    /// Get all warnings emitted during the run.
    pub fn warnings(&self) -> &[DiagnosticWarning] {
        &self.warnings
    }

    /// Get the count of unacknowledged warnings.
    pub fn unacknowledged_count(&self) -> usize {
        self.warnings.iter().filter(|w| !w.acknowledged).count()
    }

    /// Acknowledge a warning by index.
    pub fn acknowledge(&mut self, index: usize) {
        if let Some(w) = self.warnings.get_mut(index) {
            w.acknowledged = true;
        }
    }

    /// Clear a resolved signal so it can fire again if conditions recur.
    pub fn resolve_signal(&mut self, signal: DiagnosticSignal, component: &str) {
        self.active_signals
            .retain(|k| !(k.0 == signal && k.1 == component));
    }

    /// Get the diagnostic config.
    pub fn config(&self) -> &DiagnosticConfig {
        &self.config
    }

    /// Notify the diagnostic layer of a phase transition.
    ///
    /// Clears the metric history and active signal deduplication so that:
    /// - Trend analysis starts fresh for the new phase (bootstrap metrics
    ///   don't contaminate representation_formation analysis)
    /// - Signals can re-fire if conditions recur under the new phase's rules
    pub fn on_phase_transition(&mut self, _from: Phase, _to: Phase) {
        self.history.clear();
        self.active_signals.clear();
        self.best_loss = None;
        self.best_loss_step = 0;
        self.intervention_outcomes.clear();
    }

    // -----------------------------------------------------------------------
    // Signal 1: Unused Capacity
    // -----------------------------------------------------------------------
    //
    // Detects attention heads with near-zero variance, layers with
    // pass-through behavior, parameters that never update meaningfully.

    fn detect_unused_capacity(
        &self,
        step: u64,
        metrics: &MetricSnapshot,
        out: &mut Vec<DiagnosticWarning>,
    ) {
        for component in &self.components {
            // Count how many recent steps had near-zero variance
            // Try both activation_variance_min and activation_variance keys
            let near_zero_var_count = self
                .history
                .iter()
                .filter(|m| {
                    Self::resolve_var_key(m, component)
                        .and_then(|k| m.get(&k).copied())
                        .map_or(false, |v| v < self.config.unused_capacity_variance_floor)
                })
                .count();

            let near_zero_grad_count = self
                .history
                .iter()
                .filter(|m| {
                    Self::resolve_grad_key(m, component)
                        .and_then(|k| m.get(&k).copied())
                        .map_or(false, |v| v < self.config.unused_capacity_grad_floor)
                })
                .count();

            let window = self.history.len();
            let var_ratio = near_zero_var_count as f64 / window as f64;
            let grad_ratio = near_zero_grad_count as f64 / window as f64;

            if var_ratio >= self.config.unused_capacity_persistence
                || grad_ratio >= self.config.unused_capacity_persistence
            {
                let current_var = Self::resolve_var_key(metrics, component)
                    .and_then(|k| metrics.get(&k).copied());
                let current_grad = Self::resolve_grad_key(metrics, component)
                    .and_then(|k| metrics.get(&k).copied());

                let mut evidence = Vec::new();
                if var_ratio >= self.config.unused_capacity_persistence {
                    evidence.push(format!(
                        "Activation variance in {} has been below {:.1e} for {:.0}% of the last {} steps{}",
                        component,
                        self.config.unused_capacity_variance_floor,
                        var_ratio * 100.0,
                        window,
                        current_var.map_or(String::new(), |v| format!(" (current: {:.2e})", v)),
                    ));
                }
                if grad_ratio >= self.config.unused_capacity_persistence {
                    evidence.push(format!(
                        "Gradient norm in {} has been below {:.1e} for {:.0}% of the last {} steps{}",
                        component,
                        self.config.unused_capacity_grad_floor,
                        grad_ratio * 100.0,
                        window,
                        current_grad.map_or(String::new(), |v| format!(" (current: {:.2e})", v)),
                    ));
                }

                let confidence = (var_ratio.max(grad_ratio) - self.config.unused_capacity_persistence)
                    / (1.0 - self.config.unused_capacity_persistence)
                    * 0.5
                    + 0.5;

                out.push(DiagnosticWarning {
                    signal: DiagnosticSignal::UnusedCapacity,
                    step,
                    summary: format!(
                        "Observed persistently low activity in {}. This is consistent with \
                         unused model capacity — the component appears to contribute minimally \
                         to the forward pass.",
                        component,
                    ),
                    evidence,
                    confidence: confidence.min(1.0),
                    acknowledged: false,
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // Signal 2: Missing Structural Signal
    // -----------------------------------------------------------------------
    //
    // Higher-order pathways (multi-hop reasoning, compositional structure)
    // remain dormant despite capacity. Detected via persistently low
    // attention entropy.

    fn detect_missing_structural_signal(
        &self,
        step: u64,
        metrics: &MetricSnapshot,
        out: &mut Vec<DiagnosticWarning>,
    ) {
        for component in &self.components {
            let entropy_key = format!("{}.attention_entropy", component);

            let low_entropy_count = self
                .history
                .iter()
                .filter(|m| {
                    m.get(&entropy_key)
                        .map_or(false, |&v| v < self.config.structural_signal_entropy_floor)
                })
                .count();

            let window = self.history.len();
            // Only proceed if we've actually seen this metric
            let has_metric = self.history.iter().any(|m| m.contains_key(&entropy_key));
            if !has_metric {
                continue;
            }

            let low_ratio = low_entropy_count as f64 / window as f64;

            if low_ratio >= 0.7 {
                let current = metrics.get(&entropy_key).copied();

                let evidence = vec![
                    format!(
                        "Attention entropy in {} has been below {:.2} for {:.0}% of the last {} steps{}",
                        component,
                        self.config.structural_signal_entropy_floor,
                        low_ratio * 100.0,
                        window,
                        current.map_or(String::new(), |v| format!(" (current: {:.4})", v)),
                    ),
                    format!(
                        "This suggests uniform or collapsed attention patterns, consistent with \
                         the training corpus lacking structural signal for higher-order pathways \
                         in this component."
                    ),
                ];

                let confidence = (low_ratio - 0.7) / 0.3 * 0.4 + 0.4;

                out.push(DiagnosticWarning {
                    signal: DiagnosticSignal::MissingStructuralSignal,
                    step,
                    summary: format!(
                        "Observed persistently low attention entropy in {}. This is consistent \
                         with missing structural signal in the training corpus for higher-order \
                         pathways.",
                        component,
                    ),
                    evidence,
                    confidence: confidence.min(1.0),
                    acknowledged: false,
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // Signal 3: Loss-Representation Misalignment
    // -----------------------------------------------------------------------
    //
    // Loss decreases but declared objective metrics don't move. Learning
    // happens in subspaces unrelated to task.

    fn detect_loss_representation_misalignment(
        &self,
        step: u64,
        out: &mut Vec<DiagnosticWarning>,
    ) {
        if self.history.len() < 20 {
            return;
        }

        // Split history into first half and second half for trend comparison
        let mid = self.history.len() / 2;
        let all: Vec<&MetricSnapshot> = self.history.iter().collect();
        let first = &all[..mid];
        let second = &all[mid..];

        // Compute loss trend
        let first_loss = Self::mean_metric(first, "loss");
        let second_loss = Self::mean_metric(second, "loss");

        let (Some(fl), Some(sl)) = (first_loss, second_loss) else {
            return;
        };

        // Is loss improving?
        let loss_improvement = (fl - sl) / fl.abs().max(1e-10);
        if loss_improvement < self.config.alignment_loss_improvement_threshold {
            return; // Loss isn't improving, so no misalignment concern
        }

        // Check if representation metrics are stagnant while loss improves
        let mut stagnant_components = Vec::new();
        let mut evidence = Vec::new();

        for component in &self.components {
            let cos_key = format!("{}.pairwise_cosine", component);

            let mut stagnant = true;

            // Check cosine stagnation
            let first_cos = Self::mean_metric(first, &cos_key);
            let second_cos = Self::mean_metric(second, &cos_key);
            if let (Some(fv), Some(sv)) = (first_cos, second_cos) {
                let change = (sv - fv).abs() / fv.abs().max(1e-10);
                if change > self.config.alignment_repr_stagnation_threshold {
                    stagnant = false;
                }
            }

            // Check variance stagnation (try _min suffix first)
            let first_var = Self::mean_var_metric(first, component);
            let second_var = Self::mean_var_metric(second, component);
            if let (Some(fv), Some(sv)) = (first_var, second_var) {
                let change = (sv - fv).abs() / fv.abs().max(1e-10);
                if change > self.config.alignment_repr_stagnation_threshold {
                    stagnant = false;
                }
            }

            if stagnant {
                // Check we actually had metrics for this component
                let has_var = first.iter().any(|m| Self::resolve_var_key(m, component).is_some());
                let has_data = has_var || first.iter().any(|m| m.contains_key(&cos_key));
                if has_data {
                    stagnant_components.push(component.clone());
                    evidence.push(format!(
                        "Representation metrics in {} have not changed meaningfully \
                         over the observation window while loss improved {:.1}%.",
                        component,
                        loss_improvement * 100.0,
                    ));
                }
            }
        }

        if !stagnant_components.is_empty() {
            evidence.insert(0, format!(
                "Loss improved {:.1}% over the last {} steps (from {:.4} to {:.4}).",
                loss_improvement * 100.0,
                self.history.len(),
                fl,
                sl,
            ));
            evidence.push(
                "Learning appears to occur in subspaces orthogonal to the declared \
                 structural objectives. This may indicate the model is optimizing \
                 something other than the intended task."
                    .into(),
            );

            let confidence = (loss_improvement * 5.0).min(0.9).max(0.3);

            out.push(DiagnosticWarning {
                signal: DiagnosticSignal::LossRepresentationMisalignment,
                step,
                summary: format!(
                    "Loss is decreasing while representation metrics in [{}] remain unchanged. \
                     This suggests learning in subspaces unrelated to declared objectives.",
                    stagnant_components.join(", "),
                ),
                evidence,
                confidence,
                acknowledged: false,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Signal 4: Dynamically Unlearnable Regime
    // -----------------------------------------------------------------------
    //
    // Gradients exist but update scale prevents escape from basin.
    // LR too low to climb out, too high to settle.

    fn detect_unlearnable_regime(
        &self,
        step: u64,
        _metrics: &MetricSnapshot,
        out: &mut Vec<DiagnosticWarning>,
    ) {
        if self.history.len() < 20 {
            return;
        }

        // Check for loss plateau despite non-zero gradients
        let all: Vec<&MetricSnapshot> = self.history.iter().collect();
        let mid = all.len() / 2;
        let first = &all[..mid];
        let second = &all[mid..];

        let first_loss = Self::mean_metric(first, "loss");
        let second_loss = Self::mean_metric(second, "loss");

        let (Some(fl), Some(sl)) = (first_loss, second_loss) else {
            return;
        };

        // Loss is plateau'd (not improving meaningfully)
        let loss_change = (fl - sl).abs() / fl.abs().max(1e-10);
        if loss_change > 0.01 {
            return; // Loss is still moving, not stuck
        }

        // Check if gradients exist but are at problematic scales
        let mut problematic_components = Vec::new();
        let mut evidence = Vec::new();

        for component in &self.components {
            // Get recent grad norms — try grad_norm_min first, then grad_norm
            let recent_grads: Vec<f64> = {
                let min_key = format!("{}.grad_norm_min", component);
                let plain_key = format!("{}.grad_norm", component);
                let grads: Vec<f64> = self
                    .history
                    .iter()
                    .rev()
                    .take(10)
                    .filter_map(|m| m.get(&min_key).copied())
                    .collect();
                if grads.is_empty() {
                    self.history
                        .iter()
                        .rev()
                        .take(10)
                        .filter_map(|m| m.get(&plain_key).copied())
                        .collect()
                } else {
                    grads
                }
            };

            if recent_grads.is_empty() {
                continue;
            }

            let mean_grad = recent_grads.iter().sum::<f64>() / recent_grads.len() as f64;

            // Check if gradients are too small (LR too low) or oscillating (LR too high)
            if mean_grad > 0.0 && mean_grad < self.config.unlearnable_grad_ratio_floor {
                problematic_components.push(component.clone());
                evidence.push(format!(
                    "Gradient norm in {} averages {:.2e} over recent steps — \
                     this is unlikely to produce meaningful parameter updates \
                     at typical learning rates.",
                    component, mean_grad,
                ));
            }

            // Check for oscillating gradients (high variance relative to mean)
            if recent_grads.len() >= 5 {
                let grad_variance = Self::variance(&recent_grads);
                let cv = grad_variance.sqrt() / mean_grad.max(1e-10);
                if cv > 2.0 && mean_grad > self.config.unused_capacity_grad_floor {
                    problematic_components.push(component.clone());
                    evidence.push(format!(
                        "Gradient norm in {} shows high variability (CV={:.1}) \
                         suggesting oscillatory dynamics inconsistent with stable learning.",
                        component, cv,
                    ));
                }
            }
        }

        if !problematic_components.is_empty() {
            evidence.insert(0, format!(
                "Loss has plateaued at {:.4} (change < 1% over {} steps) while \
                 gradients remain non-zero.",
                sl,
                self.history.len(),
            ));
            evidence.push(
                "This suggests a dynamically unlearnable regime where the current \
                 hyperparameters prevent the model from making effective progress."
                    .into(),
            );

            // Confidence scales with how long the plateau has lasted
            let confidence = (self.history.len() as f64 / self.config.history_window as f64)
                .min(0.8)
                .max(0.3);

            out.push(DiagnosticWarning {
                signal: DiagnosticSignal::DynamicallyUnlearnableRegime,
                step,
                summary: format!(
                    "Loss has plateaued while gradients persist in [{}]. This suggests \
                     the current learning rate prevents effective recovery from the \
                     current loss basin.",
                    problematic_components.join(", "),
                ),
                evidence,
                confidence,
                acknowledged: false,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Signal 5: Shortcut Learning
    // -----------------------------------------------------------------------
    //
    // High task performance but representation rank collapses. Model ignores
    // input structure, exploits surface statistics.

    fn detect_shortcut_learning(
        &self,
        step: u64,
        out: &mut Vec<DiagnosticWarning>,
    ) {
        if self.history.len() < 20 {
            return;
        }

        let all: Vec<&MetricSnapshot> = self.history.iter().collect();
        let mid = all.len() / 2;
        let first = &all[..mid];
        let second = &all[mid..];

        // Is loss improving (task performance getting better)?
        let first_loss = Self::mean_metric(first, "loss");
        let second_loss = Self::mean_metric(second, "loss");

        let (Some(fl), Some(sl)) = (first_loss, second_loss) else {
            return;
        };

        let loss_improving = sl < fl * 0.99; // at least 1% improvement
        if !loss_improving {
            return;
        }

        // Check for representation collapse while loss improves
        let mut collapsing_components = Vec::new();
        let mut evidence = Vec::new();
        let mut rank_supported_components = 0usize;

        for component in &self.components {
            let cos_key = format!("{}.pairwise_cosine", component);

            let first_cos = Self::mean_metric(first, &cos_key);
            let second_cos = Self::mean_metric(second, &cos_key);
            let first_var = Self::mean_var_metric(first, component);
            let second_var = Self::mean_var_metric(second, component);

            let mut signals = 0;

            // Cosine similarity increasing (representations becoming more similar)
            if let (Some(fc), Some(sc)) = (first_cos, second_cos) {
                let cosine_drift = sc - fc;
                if cosine_drift > self.config.shortcut_cosine_drift {
                    signals += 1;
                    evidence.push(format!(
                        "Pairwise cosine similarity in {} increased from {:.4} to {:.4}. \
                         Representations are becoming less diverse.",
                        component, fc, sc,
                    ));
                }
            }

            // Variance decreasing (representation rank collapsing)
            if let (Some(fv), Some(sv)) = (first_var, second_var) {
                if fv > 1e-10 {
                    let var_decrease = (fv - sv) / fv;
                    if var_decrease > self.config.shortcut_variance_drift {
                        signals += 1;
                        evidence.push(format!(
                            "Activation variance in {} decreased {:.1}% (from {:.2e} to {:.2e}). \
                             Representation dimensionality appears to be collapsing.",
                            component,
                            var_decrease * 100.0,
                            fv,
                            sv,
                        ));
                    }
                }
            }

            // Variance exploding (shortcut feature amplification).
            // When a model discovers a trivial shortcut, it may amplify that
            // feature rather than collapse. The result is rapidly increasing
            // variance concentrated in a low-dimensional subspace — the model
            // looks "active" but is exploiting a single signal.
            if let (Some(fv), Some(sv)) = (first_var, second_var) {
                if fv > 1e-10 {
                    let var_increase = (sv - fv) / fv;
                    if var_increase > self.config.shortcut_variance_explosion {
                        signals += 1;
                        evidence.push(format!(
                            "Activation variance in {} increased {:.0}% (from {:.2e} to {:.2e}). \
                             This rapid variance growth while loss improves suggests the model \
                             is amplifying a low-dimensional shortcut feature rather than learning \
                             distributed representations.",
                            component,
                            var_increase * 100.0,
                            fv,
                            sv,
                        ));
                    }
                }
            }

            // Rank-based discrimination (§15.7): if {component}.effective_rank
            // is reported, use it to distinguish shortcut amplification from
            // legitimate complex feature learning. Zero-config: if the metric
            // never appears, this block is a no-op.
            let rank_key = format!("{}.effective_rank", component);
            let first_rank = Self::mean_metric(first, &rank_key);
            let second_rank = Self::mean_metric(second, &rank_key);
            let mut has_rank_suppression = false;

            if let (Some(fr), Some(sr)) = (first_rank, second_rank) {
                if fr > 1e-10 {
                    let rank_ratio = sr / fr;
                    if rank_ratio < self.config.shortcut_rank_threshold {
                        // Rank collapsed — strong shortcut indicator. Boost signal.
                        if signals == 0 {
                            signals = 1; // Trigger even without other indicators
                        }
                        evidence.push(format!(
                            "Effective rank in {} dropped to {:.1}% of initial ({:.4} -> {:.4}). \
                             Variance is concentrating in few dimensions, consistent with \
                             shortcut feature amplification.",
                            component,
                            rank_ratio * 100.0,
                            fr, sr,
                        ));
                    } else if rank_ratio >= 1.0 && signals > 0 {
                        // Rank stable or growing — likely legitimate learning.
                        // Suppress the shortcut signal for this component.
                        signals = 0;
                        has_rank_suppression = true;
                        evidence.retain(|e| !e.contains(component.as_str()));
                    }
                }
            }
            let _ = has_rank_suppression; // used for confidence adjustment below

            if signals > 0 {
                collapsing_components.push(component.clone());
                if first_rank.is_some() && second_rank.is_some() {
                    rank_supported_components += 1;
                }
            }
        }

        if !collapsing_components.is_empty() {
            evidence.insert(0, format!(
                "Loss improved from {:.4} to {:.4} ({:.1}% decrease) over the last {} steps.",
                fl,
                sl,
                (fl - sl) / fl * 100.0,
                self.history.len(),
            ));
            evidence.push(
                "This pattern is consistent with shortcut learning — the model may \
                 be exploiting low-dimensional, input-insensitive features that satisfy \
                 the loss without learning the intended structure."
                    .into(),
            );

            // When rank data supports the signal, allow higher confidence (up to 0.9).
            // Without rank data, cap at 0.5 — variance alone is ambiguous (§12.2 note).
            let max_confidence = if rank_supported_components > 0 { 0.9 } else { 0.5 };
            let confidence = (collapsing_components.len() as f64 / self.components.len() as f64)
                .min(max_confidence)
                .max(0.3);

            out.push(DiagnosticWarning {
                signal: DiagnosticSignal::ShortcutLearning,
                step,
                summary: format!(
                    "Loss is improving while representation diversity in [{}] is decreasing. \
                     This pattern is consistent with shortcut learning.",
                    collapsing_components.join(", "),
                ),
                evidence,
                confidence,
                acknowledged: false,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Signal 6: Missing Expected Metrics
    // -----------------------------------------------------------------------
    //
    // An invariant's metric key has never appeared in any metric snapshot.
    // The invariant is silently inactive — a configuration or integration
    // error between the spec and the training loop.

    fn detect_missing_expected_metrics(
        &self,
        step: u64,
        out: &mut Vec<DiagnosticWarning>,
    ) {
        if self.expected_metrics.is_empty() {
            return;
        }

        // Only check after we've accumulated a reasonable history
        if self.history.len() < 10 {
            return;
        }

        let mut missing = Vec::new();
        for key in &self.expected_metrics {
            // Has this metric EVER appeared in any snapshot in the history window?
            let ever_seen = self.history.iter().any(|m| m.contains_key(key));
            if !ever_seen {
                missing.push(key.clone());
            }
        }

        if !missing.is_empty() {
            let evidence: Vec<String> = missing
                .iter()
                .map(|k| {
                    format!(
                        "Metric '{}' is declared in the training spec but has not appeared in \
                         any metric snapshot over the last {} steps. The invariant that depends \
                         on this metric is not being evaluated.",
                        k,
                        self.history.len(),
                    )
                })
                .collect();

            out.push(DiagnosticWarning {
                signal: DiagnosticSignal::MissingExpectedMetric,
                step,
                summary: format!(
                    "Invariant metric{} [{}] {} never been reported by the training loop. \
                     Corresponding invariants are silently inactive.",
                    if missing.len() > 1 { "s" } else { "" },
                    missing.join(", "),
                    if missing.len() > 1 { "have" } else { "has" },
                ),
                evidence,
                confidence: 0.95, // High confidence — factual, not heuristic
                acknowledged: false,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Signal 7: Loss Stagnation
    // -----------------------------------------------------------------------
    //
    // Loss has not improved meaningfully for an extended period despite
    // healthy gradient flow. Distinct from Signal 4 (DynamicallyUnlearnable),
    // which requires gradients to be pathological. This fires when gradients
    // are normal but loss is stuck — the most common real-world failure mode.

    fn detect_loss_stagnation(
        &self,
        step: u64,
        _metrics: &MetricSnapshot,
        out: &mut Vec<DiagnosticWarning>,
    ) {
        let best = match self.best_loss {
            Some(b) => b,
            None => return,
        };

        let steps_since = step.saturating_sub(self.best_loss_step);
        if steps_since < self.config.stagnation_patience_steps {
            return;
        }

        // Check that at least one component has healthy gradients.
        // If all gradients are vanishing, this is convergence (not stagnation)
        // and Signal 4 or natural completion applies instead.
        let mut has_healthy_grads = false;
        let mut grad_evidence = Vec::new();

        for component in &self.components {
            // Try grad_norm_min first, then grad_norm
            let recent_grads: Vec<f64> = {
                let min_key = format!("{}.grad_norm_min", component);
                let plain_key = format!("{}.grad_norm", component);
                let grads: Vec<f64> = self
                    .history
                    .iter()
                    .rev()
                    .take(10)
                    .filter_map(|m| m.get(&min_key).copied())
                    .collect();
                if grads.is_empty() {
                    self.history
                        .iter()
                        .rev()
                        .take(10)
                        .filter_map(|m| m.get(&plain_key).copied())
                        .collect()
                } else {
                    grads
                }
            };

            if recent_grads.is_empty() {
                continue;
            }

            let mean_grad = recent_grads.iter().sum::<f64>() / recent_grads.len() as f64;
            if mean_grad >= self.config.stagnation_grad_floor {
                has_healthy_grads = true;
                grad_evidence.push(format!(
                    "Gradient norm in {} averages {:.2e} — healthy flow, \
                     suggesting the model is actively trying to learn.",
                    component, mean_grad,
                ));
            }
        }

        if !has_healthy_grads {
            return; // Gradients vanishing — convergence, not stagnation
        }

        let mut evidence = Vec::new();
        evidence.push(format!(
            "Loss has not improved beyond {:.4} for {} steps (best seen at step {}). \
             The improvement threshold is {:.1}%.",
            best,
            steps_since,
            self.best_loss_step,
            self.config.stagnation_improvement_threshold * 100.0,
        ));
        evidence.extend(grad_evidence);
        evidence.push(
            "Healthy gradients with stagnant loss suggests the model may be stuck \
             in a flat loss basin, the data signal-to-noise ratio may be too low, \
             or the architecture may lack capacity for the task."
                .into(),
        );

        // Confidence scales with plateau duration: 0.3 at patience, up to 0.8 at 2x patience
        let patience = self.config.stagnation_patience_steps as f64;
        let confidence = ((steps_since as f64 - patience) / patience * 0.5 + 0.3).min(0.8);

        out.push(DiagnosticWarning {
            signal: DiagnosticSignal::LossStagnation,
            step,
            summary: format!(
                "Loss has stagnated at {:.4} for {} steps despite healthy gradient \
                 flow. Training may not be making meaningful progress.",
                best, steps_since,
            ),
            evidence,
            confidence,
            acknowledged: false,
        });
    }

    // -----------------------------------------------------------------------
    // Signal 8 — ThresholdDrift
    // -----------------------------------------------------------------------

    fn detect_threshold_drift(
        &self,
        step: u64,
        _metrics: &MetricSnapshot,
        out: &mut Vec<DiagnosticWarning>,
    ) {
        if self.invariant_thresholds.is_empty() {
            return;
        }

        // We need enough history points for trend analysis
        let min_points = (self.config.drift_window_steps / self.config.cadence.max(1) as usize)
            .min(self.history.len())
            .max(1);
        if self.history.len() < 10.min(min_points) {
            return;
        }

        for (metric_key, &(threshold, direction)) in &self.invariant_thresholds {
            // Extract values from history for this metric
            let values: Vec<f64> = self
                .history
                .iter()
                .filter_map(|m| m.get(metric_key).copied())
                .collect();

            if values.len() < 10 {
                continue;
            }

            // Use the tail of the values for trend analysis
            let window_len = min_points.min(values.len());
            let window = &values[values.len() - window_len..];

            // Count pairs that trend toward threshold
            let mut toward_count = 0usize;
            let total_pairs = window.len() - 1;
            if total_pairs == 0 {
                continue;
            }

            for i in 0..total_pairs {
                let moves_toward = match direction {
                    ThresholdDirection::Max => window[i + 1] > window[i],
                    ThresholdDirection::Min => window[i + 1] < window[i],
                };
                if moves_toward {
                    toward_count += 1;
                }
            }

            let monotonic_frac = toward_count as f64 / total_pairs as f64;
            if monotonic_frac < self.config.drift_monotonic_pct {
                continue;
            }

            // Linear extrapolation: estimate steps to crossing
            let current = *window.last().unwrap();
            let first = window[0];
            let slope_per_point = (current - first) / total_pairs as f64;
            if slope_per_point.abs() < 1e-12 {
                continue; // No meaningful slope
            }

            // Convert slope from "per history point" to "per step"
            let slope_per_step = slope_per_point / self.config.cadence.max(1) as f64;

            let distance_to_threshold = match direction {
                ThresholdDirection::Max => threshold - current,
                ThresholdDirection::Min => current - threshold,
            };

            // If already past threshold or moving away, skip
            if distance_to_threshold <= 0.0 {
                continue;
            }

            let steps_to_crossing = (distance_to_threshold / slope_per_step.abs()) as u64;
            if steps_to_crossing > self.config.drift_crossing_horizon {
                continue;
            }

            // Extract component from metric_key (everything before first '.')
            let component = metric_key
                .split('.')
                .next()
                .unwrap_or("global")
                .to_string();

            let confidence = (0.3
                + 0.4 * (monotonic_frac - self.config.drift_monotonic_pct)
                    / (1.0 - self.config.drift_monotonic_pct).max(0.01))
            .min(0.8);

            out.push(DiagnosticWarning {
                signal: DiagnosticSignal::ThresholdDrift,
                step,
                summary: format!(
                    "Metric {} in {} is trending toward its invariant threshold ({:.4}). \
                     At current rate, crossing is estimated in ~{} steps.",
                    metric_key, component, threshold, steps_to_crossing,
                ),
                evidence: vec![
                    format!(
                        "Over the last {} data points, {:.0}% of consecutive pairs move \
                         toward the {} threshold of {:.4}.",
                        window_len,
                        monotonic_frac * 100.0,
                        direction,
                        threshold,
                    ),
                    format!(
                        "Current value: {:.6}, slope: {:.2e} per step, \
                         estimated crossing in {} steps.",
                        current, slope_per_step, steps_to_crossing,
                    ),
                ],
                confidence,
                acknowledged: false,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Signal 9 — MetricInstability
    // -----------------------------------------------------------------------

    fn detect_metric_instability(
        &self,
        step: u64,
        out: &mut Vec<DiagnosticWarning>,
    ) {
        if self.invariant_thresholds.is_empty() || self.history.len() < 10 {
            return;
        }

        for (metric_key, _) in &self.invariant_thresholds {
            let values: Vec<f64> = self
                .history
                .iter()
                .filter_map(|m| m.get(metric_key).copied())
                .collect();

            if values.len() < 10 {
                continue;
            }

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            if mean.abs() < 1e-12 {
                continue; // Avoid division by near-zero mean
            }

            let variance = Self::variance(&values);
            let std_dev = variance.sqrt();
            let cv = std_dev / mean.abs();

            if cv <= self.config.instability_cv_threshold {
                continue;
            }

            let component = metric_key
                .split('.')
                .next()
                .unwrap_or("global")
                .to_string();

            let excess = (cv - self.config.instability_cv_threshold)
                / self.config.instability_cv_threshold;
            let confidence = (0.3 + 0.4 * excess.min(1.0)).min(0.8);

            out.push(DiagnosticWarning {
                signal: DiagnosticSignal::MetricInstability,
                step,
                summary: format!(
                    "Metric {} in {} exhibits high-frequency oscillation \
                     (CV={:.3}, threshold {:.3}). Training may be unstable.",
                    metric_key, component, cv, self.config.instability_cv_threshold,
                ),
                evidence: vec![
                    format!(
                        "Coefficient of variation for {} over {} data points: {:.4} \
                         (mean={:.6}, std={:.6}).",
                        metric_key, values.len(), cv, mean, std_dev,
                    ),
                    format!(
                        "A CV above {:.2} suggests the metric is oscillating rather than \
                         converging, which may indicate learning rate instability or \
                         conflicting gradient signals.",
                        self.config.instability_cv_threshold,
                    ),
                ],
                confidence,
                acknowledged: false,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Signal 10 — InterventionFutility
    // -----------------------------------------------------------------------

    fn detect_intervention_futility(
        &self,
        step: u64,
        out: &mut Vec<DiagnosticWarning>,
    ) {
        if self.intervention_outcomes.is_empty() {
            return;
        }

        // Group outcomes by component
        let mut by_component: HashMap<&str, Vec<&InterventionOutcomeRecord>> = HashMap::new();
        for record in &self.intervention_outcomes {
            by_component
                .entry(&record.component)
                .or_default()
                .push(record);
        }

        let lookback = self.config.futility_lookback_interventions;

        for (component, records) in &by_component {
            if records.len() < lookback {
                continue;
            }

            // Take the last N outcomes
            let recent = &records[records.len() - lookback..];
            let failed_count = recent.iter().filter(|r| !r.recovered).count();
            let recovered_count = recent.len() - failed_count;
            let total = recent.len();

            // Two modes of futility:
            // 1. All failed → interventions don't work at all
            // 2. All/most recovered → interventions "work" but component keeps needing them
            //    (repeated successful-but-temporary recovery is still futile)
            let all_failed = failed_count == total;

            let mut evidence = Vec::new();
            for r in recent {
                let status = if r.recovered { "recovered" } else { "not recovered" };
                evidence.push(format!(
                    "Step {}: {} — {}{}.",
                    r.step,
                    r.action,
                    status,
                    r.recovery_steps
                        .map(|s| format!(" ({} steps)", s))
                        .unwrap_or_default(),
                ));
            }

            if all_failed {
                evidence.push(
                    "Repeated failed interventions suggest the root cause may be architectural \
                     or data-related rather than transient training state."
                        .into(),
                );

                let confidence = match total {
                    0..=2 => 0.3,
                    3 => 0.5,
                    4 => 0.7,
                    _ => 0.8,
                };

                out.push(DiagnosticWarning {
                    signal: DiagnosticSignal::InterventionFutility,
                    step,
                    summary: format!(
                        "The last {} interventions on {} have all failed to produce recovery. \
                         The supervisor's corrective actions may be ineffective for this component.",
                        total, component,
                    ),
                    evidence,
                    confidence,
                    acknowledged: false,
                });
            } else {
                // Repeated interventions (some/all recovered) — chronic problem
                evidence.push(format!(
                    "{} requires repeated intervention ({} in this phase, {} recovered). \
                     Even successful interventions are not producing lasting improvement.",
                    component, total, recovered_count,
                ));

                // Lower confidence for recovered interventions (less severe than total failure)
                let confidence = match total {
                    0..=2 => 0.2,
                    3 => 0.4,
                    4 => 0.5,
                    _ => 0.6,
                };

                out.push(DiagnosticWarning {
                    signal: DiagnosticSignal::InterventionFutility,
                    step,
                    summary: format!(
                        "{} has required {} interventions in this phase ({} recovered temporarily). \
                         Repeated intervention suggests a chronic structural problem.",
                        component, total, recovered_count,
                    ),
                    evidence,
                    confidence,
                    acknowledged: false,
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // Signal 11: GradientDomination
    // -----------------------------------------------------------------------
    //
    // One component's gradient norms overwhelm all others, monopolizing
    // optimizer updates. Detected by comparing mean grad_norm across
    // components over the history window.

    fn detect_gradient_domination(
        &self,
        step: u64,
        _metrics: &MetricSnapshot,
        out: &mut Vec<DiagnosticWarning>,
    ) {
        if self.components.len() < 2 {
            return;
        }

        let all: Vec<&MetricSnapshot> = self.history.iter().collect();

        // Compute mean grad_norm per component, filtering out dead gradients
        // Tries grad_norm_min first, then grad_norm (same pattern as variance key resolution)
        let grad_floor = 1e-7;
        let mut component_means: Vec<(&str, f64)> = Vec::new();

        for component in &self.components {
            if let Some(mean) = Self::mean_grad_metric(&all, component) {
                if mean >= grad_floor {
                    component_means.push((component, mean));
                }
            }
        }

        // Need at least 2 live components to compare
        if component_means.len() < 2 {
            return;
        }

        let max_mean = component_means
            .iter()
            .map(|(_, m)| *m)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_mean = component_means
            .iter()
            .map(|(_, m)| *m)
            .fold(f64::INFINITY, f64::min);

        let ratio = max_mean / min_mean;
        if ratio < self.config.gradient_domination_ratio {
            return;
        }

        let dominant = component_means
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let suppressed: Vec<&str> = component_means
            .iter()
            .filter(|(_, m)| *m < max_mean * 0.1)
            .map(|(c, _)| *c)
            .collect();

        let mut evidence = vec![format!(
            "Mean gradient norm ratio across components is {:.1}x \
             (dominant: {} at {:.2e}, weakest at {:.2e}).",
            ratio, dominant.0, max_mean, min_mean,
        )];
        if !suppressed.is_empty() {
            evidence.push(format!(
                "Suppressed components: [{}]. These components may not receive \
                 meaningful parameter updates while {} dominates the optimizer.",
                suppressed.join(", "),
                dominant.0,
            ));
        }

        let threshold = self.config.gradient_domination_ratio;
        let confidence = (0.3 + 0.4 * ((ratio - threshold) / threshold).min(1.0)).min(0.8);

        out.push(DiagnosticWarning {
            signal: DiagnosticSignal::GradientDomination,
            step,
            summary: format!(
                "Observed gradient domination by {} — gradient norm ratio {:.1}x \
                 exceeds threshold {:.0}x. Suppressed components may not learn effectively.",
                dominant.0, ratio, threshold,
            ),
            evidence,
            confidence,
            acknowledged: false,
        });
    }

    // -----------------------------------------------------------------------
    // Signal 12: MetricAnomaly (NaN/Inf sentinel)
    // -----------------------------------------------------------------------
    //
    // Defensive check for corrupted metric values. NaN and Inf propagate
    // silently through float arithmetic and corrupt all downstream analysis.

    fn detect_metric_anomaly(
        &self,
        step: u64,
        metrics: &MetricSnapshot,
        out: &mut Vec<DiagnosticWarning>,
    ) {
        for (key, &value) in metrics {
            if value.is_nan() || value.is_infinite() {
                let anomaly_type = if value.is_nan() { "NaN" } else { "Inf" };

                let component = key.split('.').next().unwrap_or("global");

                out.push(DiagnosticWarning {
                    signal: DiagnosticSignal::MetricAnomaly,
                    step,
                    summary: format!(
                        "Metric '{}' in {} has value {}. Training state may be corrupted.",
                        key, component, anomaly_type,
                    ),
                    evidence: vec![format!(
                        "Metric '{}' reported as {} at step {}. IEEE 754 non-finite \
                         values indicate numerical corruption — likely a division by \
                         zero, log of zero, or gradient explosion.",
                        key, anomaly_type, step,
                    )],
                    confidence: 0.95,
                    acknowledged: false,
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // Signal 13: TrainValDivergence (overfitting detection)
    // -----------------------------------------------------------------------
    //
    // Training loss improving while validation loss worsening indicates
    // the model is memorizing training data rather than learning generalizable
    // patterns. Zero-config: just report `val_loss` alongside `loss`.

    fn detect_train_val_divergence(
        &self,
        step: u64,
        out: &mut Vec<DiagnosticWarning>,
    ) {
        if self.history.len() < 20 {
            return;
        }

        // Check if val_loss appears in at least half the history
        let val_loss_count = self
            .history
            .iter()
            .filter(|m| m.contains_key("val_loss"))
            .count();
        if val_loss_count < self.history.len() / 2 {
            return;
        }

        // Split history into first half and second half
        let all: Vec<&MetricSnapshot> = self.history.iter().collect();
        let mid = all.len() / 2;
        let first = &all[..mid];
        let second = &all[mid..];

        let (Some(ft), Some(st)) = (Self::mean_metric(first, "loss"), Self::mean_metric(second, "loss")) else {
            return;
        };
        let (Some(fv), Some(sv)) = (Self::mean_metric(first, "val_loss"), Self::mean_metric(second, "val_loss")) else {
            return;
        };

        // Train loss must be decreasing, val loss must be increasing
        if st >= ft || sv <= fv {
            return;
        }

        // Compute relative divergence
        let train_change = (ft - st) / ft.abs().max(1e-10);
        let val_change = (sv - fv) / fv.abs().max(1e-10);
        let divergence = train_change + val_change;

        if divergence < self.config.overfit_min_divergence {
            return;
        }

        let confidence = (0.3 + 0.4 * (divergence / self.config.overfit_min_divergence - 1.0).min(1.0))
            .min(0.85);

        out.push(DiagnosticWarning {
            signal: DiagnosticSignal::TrainValDivergence,
            step,
            summary: format!(
                "Training loss is decreasing while validation loss is increasing \
                 (divergence {:.1}%). This pattern is consistent with overfitting.",
                divergence * 100.0,
            ),
            evidence: vec![
                format!(
                    "Training loss: {:.4} -> {:.4} ({:.1}% decrease) over {} steps.",
                    ft, st, train_change * 100.0, self.history.len(),
                ),
                format!(
                    "Validation loss: {:.4} -> {:.4} ({:.1}% increase) over the same period.",
                    fv, sv, val_change * 100.0,
                ),
                "Diverging train/val loss is the canonical signature of overfitting. \
                 Consider regularization, data augmentation, or early stopping."
                    .into(),
            ],
            confidence,
            acknowledged: false,
        });
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Compute mean of a metric across a slice of snapshots.
    fn mean_metric(snapshots: &[&MetricSnapshot], key: &str) -> Option<f64> {
        let values: Vec<f64> = snapshots
            .iter()
            .filter_map(|m| m.get(key).copied())
            .collect();
        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        }
    }

    /// Resolve the actual gradient norm metric key for a component.
    /// Tries `{component}.grad_norm_min` first (standard invariant name),
    /// then falls back to `{component}.grad_norm`.
    fn resolve_grad_key(snapshot: &MetricSnapshot, component: &str) -> Option<String> {
        let min_key = format!("{}.grad_norm_min", component);
        if snapshot.contains_key(&min_key) {
            return Some(min_key);
        }
        let plain_key = format!("{}.grad_norm", component);
        if snapshot.contains_key(&plain_key) {
            return Some(plain_key);
        }
        None
    }

    /// Compute mean of a gradient norm metric across snapshots, trying `_min` suffix first.
    fn mean_grad_metric(snapshots: &[&MetricSnapshot], component: &str) -> Option<f64> {
        let min_key = format!("{}.grad_norm_min", component);
        let plain_key = format!("{}.grad_norm", component);
        Self::mean_metric(snapshots, &min_key)
            .or_else(|| Self::mean_metric(snapshots, &plain_key))
    }

    /// Resolve the actual variance metric key for a component.
    /// Tries `{component}.activation_variance_min` first (standard invariant name),
    /// then falls back to `{component}.activation_variance`.
    fn resolve_var_key(snapshot: &MetricSnapshot, component: &str) -> Option<String> {
        let min_key = format!("{}.activation_variance_min", component);
        if snapshot.contains_key(&min_key) {
            return Some(min_key);
        }
        let plain_key = format!("{}.activation_variance", component);
        if snapshot.contains_key(&plain_key) {
            return Some(plain_key);
        }
        None
    }

    /// Compute mean of a variance metric across snapshots, trying `_min` suffix first.
    fn mean_var_metric(snapshots: &[&MetricSnapshot], component: &str) -> Option<f64> {
        let min_key = format!("{}.activation_variance_min", component);
        let plain_key = format!("{}.activation_variance", component);
        Self::mean_metric(snapshots, &min_key)
            .or_else(|| Self::mean_metric(snapshots, &plain_key))
    }

    /// Compute variance of a slice of f64s.
    fn variance(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
    }

    /// Extract a key for deduplication from a warning (component or "global").
    fn warning_key(&self, w: &DiagnosticWarning) -> String {
        // Extract component name from summary if present
        for comp in &self.components {
            if w.summary.contains(comp) {
                return comp.clone();
            }
        }
        "global".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_config() -> DiagnosticConfig {
        DiagnosticConfig {
            warmup_steps: 10,
            cadence: 1, // Every step for testing
            history_window: 50,
            min_confidence: 0.1,
            ..Default::default()
        }
    }

    fn healthy_metrics() -> MetricSnapshot {
        let mut m = HashMap::new();
        m.insert("loss".into(), 2.0);
        m.insert("backbone.activation_variance".into(), 0.1);
        m.insert("backbone.grad_norm".into(), 0.05);
        m.insert("backbone.pairwise_cosine".into(), 0.5);
        m.insert("backbone.attention_entropy".into(), 0.8);
        m.insert("head.activation_variance".into(), 0.08);
        m.insert("head.grad_norm".into(), 0.04);
        m.insert("head.pairwise_cosine".into(), 0.4);
        m.insert("head.attention_entropy".into(), 0.7);
        m
    }

    #[test]
    fn no_warnings_on_healthy_training() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Run 30 steps with healthy metrics
        for step in 0..30 {
            let warnings = diag.diagnose(step, &healthy_metrics());
            assert!(
                warnings.is_empty(),
                "Step {}: unexpected warning: {:?}",
                step,
                warnings,
            );
        }
    }

    #[test]
    fn detects_unused_capacity() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Head has near-zero activation and gradients
        let mut metrics = healthy_metrics();
        metrics.insert("head.activation_variance".into(), 1e-7);
        metrics.insert("head.grad_norm".into(), 1e-8);

        for step in 0..50 {
            let warnings = diag.diagnose(step, &metrics);
            if let Some(w) = warnings.iter().find(|w| w.signal == DiagnosticSignal::UnusedCapacity) {
                assert!(w.summary.contains("head"));
                return;
            }
        }
        panic!("Expected unused capacity warning");
    }

    #[test]
    fn detects_missing_structural_signal() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        let mut metrics = healthy_metrics();
        metrics.insert("head.attention_entropy".into(), 0.05); // very low

        for step in 0..50 {
            let warnings = diag.diagnose(step, &metrics);
            if let Some(w) = warnings.iter().find(|w| w.signal == DiagnosticSignal::MissingStructuralSignal) {
                assert!(w.summary.contains("head"));
                return;
            }
        }
        panic!("Expected missing structural signal warning");
    }

    #[test]
    fn detects_loss_repr_misalignment() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Loss decreasing but representation metrics frozen
        for step in 0..80 {
            let mut metrics = healthy_metrics();
            // Loss steadily decreasing
            metrics.insert("loss".into(), 3.0 - (step as f64 * 0.03));
            // Representation metrics completely frozen
            metrics.insert("backbone.activation_variance".into(), 0.1);
            metrics.insert("backbone.pairwise_cosine".into(), 0.5);
            metrics.insert("head.activation_variance".into(), 0.08);
            metrics.insert("head.pairwise_cosine".into(), 0.4);

            let warnings = diag.diagnose(step, &metrics);
            if let Some(_) = warnings.iter().find(|w| w.signal == DiagnosticSignal::LossRepresentationMisalignment) {
                return;
            }
        }
        panic!("Expected loss-representation misalignment warning");
    }

    #[test]
    fn detects_unlearnable_regime() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Loss plateaued, gradients tiny but non-zero
        for step in 0..80 {
            let mut metrics = healthy_metrics();
            metrics.insert("loss".into(), 2.5); // constant
            metrics.insert("backbone.grad_norm".into(), 1e-9); // tiny
            metrics.insert("head.grad_norm".into(), 1e-8); // tiny

            let warnings = diag.diagnose(step, &metrics);
            if let Some(_) = warnings.iter().find(|w| w.signal == DiagnosticSignal::DynamicallyUnlearnableRegime) {
                return;
            }
        }
        panic!("Expected dynamically unlearnable regime warning");
    }

    #[test]
    fn detects_shortcut_learning() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Loss improving but cosine increasing and variance decreasing
        for step in 0..80 {
            let mut metrics = healthy_metrics();
            let t = step as f64 / 80.0;
            metrics.insert("loss".into(), 3.0 - t * 1.5); // improving
            metrics.insert("backbone.pairwise_cosine".into(), 0.4 + t * 0.3); // increasing
            metrics.insert("backbone.activation_variance".into(), 0.1 * (1.0 - t * 0.8)); // decreasing
            metrics.insert("head.pairwise_cosine".into(), 0.4 + t * 0.2);
            metrics.insert("head.activation_variance".into(), 0.08 * (1.0 - t * 0.6));

            let warnings = diag.diagnose(step, &metrics);
            if let Some(_) = warnings.iter().find(|w| w.signal == DiagnosticSignal::ShortcutLearning) {
                return;
            }
        }
        panic!("Expected shortcut learning warning");
    }

    #[test]
    fn detects_shortcut_learning_via_variance_explosion() {
        let config = DiagnosticConfig {
            warmup_steps: 10,
            cadence: 1,
            history_window: 50,
            min_confidence: 0.1,
            shortcut_variance_explosion: 0.5, // 50% increase triggers
            ..Default::default()
        };
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Loss improving while variance EXPLODES (shortcut amplification pattern).
        // Cosine is stable or decreasing — the model looks healthy by collapse metrics
        // but is amplifying a shortcut feature.
        for step in 0..80 {
            let mut metrics = healthy_metrics();
            let t = step as f64 / 80.0;
            metrics.insert("loss".into(), 3.0 - t * 2.5); // rapidly improving
            metrics.insert("backbone.pairwise_cosine".into(), 0.5 - t * 0.1); // decreasing (looks healthy!)
            metrics.insert("backbone.activation_variance".into(), 0.1 * (1.0 + t * 3.0)); // exploding: 0.1 → 0.4
            metrics.insert("head.pairwise_cosine".into(), 0.4);
            metrics.insert("head.activation_variance".into(), 0.08);

            let warnings = diag.diagnose(step, &metrics);
            if let Some(w) = warnings.iter().find(|w| w.signal == DiagnosticSignal::ShortcutLearning) {
                assert!(
                    w.evidence.iter().any(|e| e.contains("increased")),
                    "Warning should mention variance increase: {:?}", w.evidence,
                );
                return;
            }
        }
        panic!("Expected shortcut learning warning from variance explosion");
    }

    #[test]
    fn deduplicates_warnings() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        let mut metrics = healthy_metrics();
        metrics.insert("head.activation_variance".into(), 1e-7);
        metrics.insert("head.grad_norm".into(), 1e-8);

        for step in 0..50 {
            diag.diagnose(step, &metrics);
        }

        // Should only fire once per (signal, component) pair — not every step
        let unused_capacity_warnings: Vec<_> = diag
            .warnings()
            .iter()
            .filter(|w| w.signal == DiagnosticSignal::UnusedCapacity)
            .collect();

        assert!(
            unused_capacity_warnings.len() <= 2,
            "Expected at most 2 unused capacity warnings (one per component), got {}",
            unused_capacity_warnings.len(),
        );
    }

    #[test]
    fn respects_warmup() {
        let config = DiagnosticConfig {
            warmup_steps: 100,
            cadence: 1,
            ..Default::default()
        };
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into()],
        );

        let mut metrics = HashMap::new();
        metrics.insert("loss".into(), 2.0);
        metrics.insert("backbone.activation_variance".into(), 1e-8);
        metrics.insert("backbone.grad_norm".into(), 1e-9);

        for step in 0..50 {
            let warnings = diag.diagnose(step, &metrics);
            assert!(warnings.is_empty(), "No warnings during warmup");
        }
    }

    #[test]
    fn acknowledge_and_resolve() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["head".into()],
        );

        let mut metrics = HashMap::new();
        metrics.insert("loss".into(), 2.0);
        metrics.insert("head.activation_variance".into(), 1e-8);
        metrics.insert("head.grad_norm".into(), 1e-9);

        // Generate a warning
        for step in 0..50 {
            diag.diagnose(step, &metrics);
        }

        assert!(diag.unacknowledged_count() > 0);
        diag.acknowledge(0);
        assert_eq!(diag.warnings()[0].acknowledged, true);

        // Resolve the signal so it can fire again
        diag.resolve_signal(DiagnosticSignal::UnusedCapacity, "head");
    }

    #[test]
    fn detects_missing_expected_metrics() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );
        // Tell the diagnostic layer to expect these metric keys
        diag.set_expected_metrics(vec![
            "head.pairwise_cosine".into(),
            "head.grad_norm".into(),
            "backbone.grad_norm".into(),
            "head.mystery_metric".into(), // This one will never appear
        ]);

        // Feed metrics that include everything except mystery_metric
        for step in 0..30 {
            let warnings = diag.diagnose(step, &healthy_metrics());
            if let Some(w) = warnings.iter().find(|w| w.signal == DiagnosticSignal::MissingExpectedMetric) {
                assert!(w.summary.contains("mystery_metric"));
                assert!(!w.summary.contains("pairwise_cosine")); // should be present
                assert!(w.confidence >= 0.9);
                return;
            }
        }
        panic!("Expected missing expected metric warning for mystery_metric");
    }

    #[test]
    fn no_missing_metric_warning_when_all_present() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );
        diag.set_expected_metrics(vec![
            "backbone.grad_norm".into(),
            "head.grad_norm".into(),
        ]);

        for step in 0..30 {
            let warnings = diag.diagnose(step, &healthy_metrics());
            let missing_warnings: Vec<_> = warnings.iter()
                .filter(|w| w.signal == DiagnosticSignal::MissingExpectedMetric)
                .collect();
            assert!(missing_warnings.is_empty(), "No warning when all expected metrics present");
        }
    }

    #[test]
    fn phase_transition_resets_history() {
        use crate::types::Phase;

        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Build up history with bad metrics
        let mut bad_metrics = healthy_metrics();
        bad_metrics.insert("head.activation_variance".into(), 1e-7);
        bad_metrics.insert("head.grad_norm".into(), 1e-8);

        for step in 0..20 {
            diag.diagnose(step, &bad_metrics);
        }

        // Should have fired unused capacity warning
        let pre_warnings = diag.warnings().len();
        assert!(pre_warnings > 0, "Should have warnings before phase transition");

        // Phase transition — resets history and active signals
        diag.on_phase_transition(Phase::Bootstrap, Phase::RepresentationFormation);

        // Feed healthy metrics now — old bad history is gone
        for step in 20..40 {
            diag.diagnose(step, &healthy_metrics());
        }

        // The old warning remains in the permanent record, but the signal
        // should not re-fire because current metrics are healthy
        let _post_warnings = diag.warnings().len();
        // Warnings might be same count or +1 if something else fires,
        // but specifically UnusedCapacity should not fire again on healthy data
        let post_unused: Vec<_> = diag.warnings().iter()
            .filter(|w| w.signal == DiagnosticSignal::UnusedCapacity && w.step >= 20)
            .collect();
        assert!(post_unused.is_empty(), "Should not re-fire unused capacity on healthy data after phase reset");
    }

    // --- Signal 7: Loss Stagnation ---

    fn make_stagnation_config() -> DiagnosticConfig {
        DiagnosticConfig {
            warmup_steps: 5,
            cadence: 1,
            history_window: 50,
            min_confidence: 0.1,
            stagnation_patience_steps: 20,
            stagnation_improvement_threshold: 0.01,
            stagnation_grad_floor: 1e-5,
            ..Default::default()
        }
    }

    fn stagnant_metrics(loss: f64, grad_norm: f64) -> MetricSnapshot {
        let mut m = HashMap::new();
        m.insert("loss".into(), loss);
        m.insert("backbone.activation_variance".into(), 0.1);
        m.insert("backbone.grad_norm".into(), grad_norm);
        m.insert("backbone.pairwise_cosine".into(), 0.5);
        m.insert("head.activation_variance".into(), 0.08);
        m.insert("head.grad_norm".into(), grad_norm);
        m.insert("head.pairwise_cosine".into(), 0.4);
        m
    }

    #[test]
    fn detects_loss_stagnation() {
        let config = make_stagnation_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Flat loss at 1.4 with healthy gradients for 40 steps (> patience=20)
        for step in 0..40 {
            let warnings = diag.diagnose(step, &stagnant_metrics(1.4, 0.01));
            if let Some(w) = warnings.iter().find(|w| w.signal == DiagnosticSignal::LossStagnation) {
                assert!(w.summary.contains("1.4000"));
                assert!(w.confidence >= 0.3);
                return;
            }
        }
        panic!("Expected loss stagnation warning");
    }

    #[test]
    fn no_stagnation_on_improving_loss() {
        let config = make_stagnation_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Loss keeps dropping — 2.0 down to ~1.2 over 40 steps
        for step in 0..40 {
            let loss = 2.0 - (step as f64 * 0.02);
            let warnings = diag.diagnose(step, &stagnant_metrics(loss, 0.01));
            let stag: Vec<_> = warnings.iter()
                .filter(|w| w.signal == DiagnosticSignal::LossStagnation)
                .collect();
            assert!(
                stag.is_empty(),
                "Step {}: unexpected stagnation warning on improving loss",
                step,
            );
        }
    }

    #[test]
    fn no_stagnation_on_convergence() {
        let config = make_stagnation_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Loss flat at 0.001 with vanishing gradients (1e-7) — convergence, not stagnation
        for step in 0..40 {
            let warnings = diag.diagnose(step, &stagnant_metrics(0.001, 1e-7));
            let stag: Vec<_> = warnings.iter()
                .filter(|w| w.signal == DiagnosticSignal::LossStagnation)
                .collect();
            assert!(
                stag.is_empty(),
                "Step {}: stagnation should not fire when gradients are vanishing (convergence)",
                step,
            );
        }
    }

    #[test]
    fn stagnation_resets_on_phase_transition() {
        use crate::types::Phase;

        let config = make_stagnation_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Build up stagnation: flat loss for 30 steps
        for step in 0..30 {
            diag.diagnose(step, &stagnant_metrics(1.4, 0.01));
        }

        // Should have fired by now
        let pre_count = diag.warnings().iter()
            .filter(|w| w.signal == DiagnosticSignal::LossStagnation)
            .count();
        assert!(pre_count > 0, "Should have stagnation warning before phase transition");

        // Phase transition resets tracker
        diag.on_phase_transition(Phase::Bootstrap, Phase::RepresentationFormation);

        // Now feed slightly better loss — the new phase's best_loss should be freshly tracked
        for step in 30..60 {
            let loss = 1.0 - (step as f64 - 30.0) * 0.005; // slowly improving
            let warnings = diag.diagnose(step, &stagnant_metrics(loss, 0.01));
            let stag: Vec<_> = warnings.iter()
                .filter(|w| w.signal == DiagnosticSignal::LossStagnation)
                .collect();
            assert!(
                stag.is_empty(),
                "Step {}: stagnation should not fire after phase reset with improving loss",
                step,
            );
        }
    }

    // ===================================================================
    // Signal 8: ThresholdDrift
    // ===================================================================

    fn make_drift_config() -> DiagnosticConfig {
        DiagnosticConfig {
            warmup_steps: 5,
            cadence: 1,
            history_window: 50,
            min_confidence: 0.1,
            drift_window_steps: 20,
            drift_crossing_horizon: 50,
            drift_monotonic_pct: 0.7,
            ..Default::default()
        }
    }

    fn drift_metrics(cosine: f64) -> MetricSnapshot {
        let mut m = HashMap::new();
        m.insert("loss".into(), 1.0);
        m.insert("backbone.pairwise_cosine".into(), cosine);
        m.insert("backbone.activation_variance".into(), 0.1);
        m.insert("backbone.grad_norm".into(), 0.05);
        m
    }

    #[test]
    fn detects_threshold_drift_toward_max() {
        let config = make_drift_config();
        let mut diag = DiagnosticLayer::new(config, vec!["backbone".into()]);

        // Set a Max threshold at 0.98 for cosine
        let mut thresholds = HashMap::new();
        thresholds.insert(
            "backbone.pairwise_cosine".into(),
            (0.98, ThresholdDirection::Max),
        );
        diag.set_invariant_thresholds(thresholds);

        // Feed steadily increasing cosine: 0.90, 0.91, ..., 0.97
        let mut found_drift = false;
        for i in 0..30 {
            let cosine = 0.90 + i as f64 * 0.003; // slow upward drift
            let warnings = diag.diagnose(i, &drift_metrics(cosine));
            if warnings.iter().any(|w| w.signal == DiagnosticSignal::ThresholdDrift) {
                found_drift = true;
            }
        }
        assert!(found_drift, "Should detect drift toward Max threshold");
    }

    #[test]
    fn no_drift_when_stable() {
        let config = make_drift_config();
        let mut diag = DiagnosticLayer::new(config, vec!["backbone".into()]);

        let mut thresholds = HashMap::new();
        thresholds.insert(
            "backbone.pairwise_cosine".into(),
            (0.98, ThresholdDirection::Max),
        );
        diag.set_invariant_thresholds(thresholds);

        // Oscillate around 0.85 — no monotonic trend
        for i in 0..30 {
            let cosine = 0.85 + if i % 2 == 0 { 0.01 } else { -0.01 };
            let warnings = diag.diagnose(i, &drift_metrics(cosine));
            let drift: Vec<_> = warnings.iter()
                .filter(|w| w.signal == DiagnosticSignal::ThresholdDrift)
                .collect();
            assert!(
                drift.is_empty(),
                "Step {}: drift should not fire on oscillating metric",
                i,
            );
        }
    }

    #[test]
    fn no_drift_when_moving_away() {
        let config = make_drift_config();
        let mut diag = DiagnosticLayer::new(config, vec!["backbone".into()]);

        let mut thresholds = HashMap::new();
        thresholds.insert(
            "backbone.pairwise_cosine".into(),
            (0.98, ThresholdDirection::Max),
        );
        diag.set_invariant_thresholds(thresholds);

        // Cosine decreasing: 0.95, 0.94, ..., 0.88 — moving away from 0.98
        for i in 0..30 {
            let cosine = 0.95 - i as f64 * 0.003;
            let warnings = diag.diagnose(i, &drift_metrics(cosine));
            let drift: Vec<_> = warnings.iter()
                .filter(|w| w.signal == DiagnosticSignal::ThresholdDrift)
                .collect();
            assert!(
                drift.is_empty(),
                "Step {}: drift should not fire when metric moves away from threshold",
                i,
            );
        }
    }

    // ===================================================================
    // Signal 9: MetricInstability
    // ===================================================================

    fn make_instability_config() -> DiagnosticConfig {
        DiagnosticConfig {
            warmup_steps: 5,
            cadence: 1,
            history_window: 50,
            min_confidence: 0.1,
            instability_cv_threshold: 0.3,
            ..Default::default()
        }
    }

    fn instability_metrics(cosine: f64) -> MetricSnapshot {
        let mut m = HashMap::new();
        m.insert("loss".into(), 1.0);
        m.insert("backbone.pairwise_cosine".into(), cosine);
        m.insert("backbone.activation_variance".into(), 0.1);
        m.insert("backbone.grad_norm".into(), 0.05);
        m
    }

    #[test]
    fn detects_metric_instability_high_cv() {
        let config = make_instability_config();
        let mut diag = DiagnosticLayer::new(config, vec!["backbone".into()]);

        let mut thresholds = HashMap::new();
        thresholds.insert(
            "backbone.pairwise_cosine".into(),
            (0.98, ThresholdDirection::Max),
        );
        diag.set_invariant_thresholds(thresholds);

        // Oscillate wildly: 0.5, 0.9, 0.5, 0.9 — CV well above 0.3
        let mut found_instability = false;
        for i in 0..30 {
            let cosine = if i % 2 == 0 { 0.5 } else { 0.9 };
            let warnings = diag.diagnose(i, &instability_metrics(cosine));
            if warnings.iter().any(|w| w.signal == DiagnosticSignal::MetricInstability) {
                found_instability = true;
            }
        }
        assert!(found_instability, "Should detect instability with high CV");
    }

    #[test]
    fn no_instability_when_stable() {
        let config = make_instability_config();
        let mut diag = DiagnosticLayer::new(config, vec!["backbone".into()]);

        let mut thresholds = HashMap::new();
        thresholds.insert(
            "backbone.pairwise_cosine".into(),
            (0.98, ThresholdDirection::Max),
        );
        diag.set_invariant_thresholds(thresholds);

        // Very stable: 0.50, 0.51, 0.49, 0.50 — CV << 0.3
        for i in 0..30 {
            let cosine = 0.50 + (i % 3) as f64 * 0.005;
            let warnings = diag.diagnose(i, &instability_metrics(cosine));
            let inst: Vec<_> = warnings.iter()
                .filter(|w| w.signal == DiagnosticSignal::MetricInstability)
                .collect();
            assert!(
                inst.is_empty(),
                "Step {}: instability should not fire on stable metric",
                i,
            );
        }
    }

    #[test]
    fn instability_confidence_scales_with_cv() {
        let config = make_instability_config();

        // Run 1: moderate oscillation — 0.4/0.9 → mean=0.65, std≈0.25, CV≈0.39
        let mut diag1 = DiagnosticLayer::new(config.clone(), vec!["backbone".into()]);
        let mut thresholds = HashMap::new();
        thresholds.insert(
            "backbone.pairwise_cosine".into(),
            (0.98, ThresholdDirection::Max),
        );
        diag1.set_invariant_thresholds(thresholds.clone());

        for i in 0..30 {
            let cosine = if i % 2 == 0 { 0.4 } else { 0.9 };
            diag1.diagnose(i, &instability_metrics(cosine));
        }
        let warnings1: Vec<_> = diag1.warnings().iter()
            .filter(|w| w.signal == DiagnosticSignal::MetricInstability)
            .collect();

        // Run 2: extreme oscillation — 0.2/0.95 → mean=0.575, std≈0.375, CV≈0.65
        let mut diag2 = DiagnosticLayer::new(config, vec!["backbone".into()]);
        diag2.set_invariant_thresholds(thresholds);

        for i in 0..30 {
            let cosine = if i % 2 == 0 { 0.2 } else { 0.95 };
            diag2.diagnose(i, &instability_metrics(cosine));
        }
        let warnings2: Vec<_> = diag2.warnings().iter()
            .filter(|w| w.signal == DiagnosticSignal::MetricInstability)
            .collect();

        assert!(!warnings1.is_empty(), "Moderate oscillation should trigger instability");
        assert!(!warnings2.is_empty(), "Extreme oscillation should trigger instability");
        // Extreme oscillation should have higher confidence
        assert!(
            warnings2[0].confidence >= warnings1[0].confidence,
            "Higher CV should produce higher confidence: {:.3} vs {:.3}",
            warnings2[0].confidence, warnings1[0].confidence,
        );
    }

    // ===================================================================
    // Signal 10: InterventionFutility
    // ===================================================================

    fn make_futility_config() -> DiagnosticConfig {
        DiagnosticConfig {
            warmup_steps: 5,
            cadence: 1,
            history_window: 50,
            min_confidence: 0.1,
            futility_lookback_interventions: 3,
            ..Default::default()
        }
    }

    fn make_failed_outcome(step: u64, component: &str) -> InterventionOutcomeRecord {
        InterventionOutcomeRecord {
            step,
            component: component.into(),
            action: Action::Reinitialize { component: component.into() },
            recovered: false,
            recovery_steps: Some(50),
        }
    }

    fn make_recovered_outcome(step: u64, component: &str) -> InterventionOutcomeRecord {
        InterventionOutcomeRecord {
            step,
            component: component.into(),
            action: Action::Reinitialize { component: component.into() },
            recovered: true,
            recovery_steps: Some(20),
        }
    }

    #[test]
    fn detects_intervention_futility() {
        let config = make_futility_config();
        let mut diag = DiagnosticLayer::new(config, vec!["backbone".into()]);

        // 3 consecutive failed interventions
        diag.record_intervention_outcome(make_failed_outcome(10, "backbone"));
        diag.record_intervention_outcome(make_failed_outcome(20, "backbone"));
        diag.record_intervention_outcome(make_failed_outcome(30, "backbone"));

        // Warm up and diagnose
        for step in 0..15 {
            diag.diagnose(step, &healthy_metrics());
        }

        let futility: Vec<_> = diag.warnings().iter()
            .filter(|w| w.signal == DiagnosticSignal::InterventionFutility)
            .collect();
        assert!(
            !futility.is_empty(),
            "Should detect futility after 3 consecutive failed interventions",
        );
        assert!(
            futility[0].confidence >= 0.5,
            "Confidence should be at least 0.5 for 3 failures",
        );
    }

    #[test]
    fn no_futility_when_too_few_interventions() {
        let config = make_futility_config();
        let mut diag = DiagnosticLayer::new(config, vec!["backbone".into()]);

        // Only 2 interventions — below lookback threshold of 3
        diag.record_intervention_outcome(make_failed_outcome(10, "backbone"));
        diag.record_intervention_outcome(make_recovered_outcome(20, "backbone"));

        for step in 0..15 {
            diag.diagnose(step, &healthy_metrics());
        }

        let futility: Vec<_> = diag.warnings().iter()
            .filter(|w| w.signal == DiagnosticSignal::InterventionFutility)
            .collect();
        assert!(
            futility.is_empty(),
            "Should not detect futility with fewer than lookback interventions",
        );
    }

    #[test]
    fn detects_chronic_futility_even_with_recovery() {
        let config = make_futility_config();
        let mut diag = DiagnosticLayer::new(config, vec!["backbone".into()]);

        // 3 interventions, all recovered — but repeated intervention IS futile
        diag.record_intervention_outcome(make_recovered_outcome(10, "backbone"));
        diag.record_intervention_outcome(make_recovered_outcome(30, "backbone"));
        diag.record_intervention_outcome(make_recovered_outcome(50, "backbone"));

        for step in 0..15 {
            diag.diagnose(step, &healthy_metrics());
        }

        let futility: Vec<_> = diag.warnings().iter()
            .filter(|w| w.signal == DiagnosticSignal::InterventionFutility)
            .collect();
        assert!(
            !futility.is_empty(),
            "Should detect chronic futility even when interventions recover temporarily",
        );
        // Chronic (recovered) futility has lower confidence than total failure
        assert!(
            futility[0].confidence < 0.5,
            "Chronic futility with recovery should have lower confidence than all-failed: got {:.2}",
            futility[0].confidence,
        );
    }

    #[test]
    fn futility_per_component() {
        let config = make_futility_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // backbone: 3 failures, head: 1 failure
        diag.record_intervention_outcome(make_failed_outcome(10, "backbone"));
        diag.record_intervention_outcome(make_failed_outcome(20, "backbone"));
        diag.record_intervention_outcome(make_failed_outcome(30, "backbone"));
        diag.record_intervention_outcome(make_failed_outcome(15, "head"));

        for step in 0..15 {
            diag.diagnose(step, &healthy_metrics());
        }

        let futility: Vec<_> = diag.warnings().iter()
            .filter(|w| w.signal == DiagnosticSignal::InterventionFutility)
            .collect();
        assert_eq!(futility.len(), 1, "Should only fire for backbone, not head");
        assert!(
            futility[0].summary.contains("backbone"),
            "Futility warning should mention backbone",
        );
    }

    #[test]
    fn futility_clears_on_phase_transition() {
        use crate::types::Phase;

        let config = make_futility_config();
        let mut diag = DiagnosticLayer::new(config, vec!["backbone".into()]);

        // 2 failures
        diag.record_intervention_outcome(make_failed_outcome(10, "backbone"));
        diag.record_intervention_outcome(make_failed_outcome(20, "backbone"));

        // Phase transition clears intervention outcomes
        diag.on_phase_transition(Phase::Bootstrap, Phase::RepresentationFormation);

        // 1 more failure (but fresh phase — total is only 1, not 3)
        diag.record_intervention_outcome(make_failed_outcome(30, "backbone"));

        for step in 0..15 {
            diag.diagnose(step, &healthy_metrics());
        }

        let futility: Vec<_> = diag.warnings().iter()
            .filter(|w| w.signal == DiagnosticSignal::InterventionFutility)
            .collect();
        assert!(
            futility.is_empty(),
            "Should not detect futility after phase transition reset (only 1 failure)",
        );
    }

    // -----------------------------------------------------------------------
    // Signal 11: GradientDomination
    // -----------------------------------------------------------------------

    fn grad_domination_metrics(backbone_grad: f64, head_grad: f64) -> MetricSnapshot {
        let mut m = HashMap::new();
        m.insert("loss".into(), 1.0);
        m.insert("backbone.activation_variance".into(), 0.1);
        m.insert("backbone.grad_norm".into(), backbone_grad);
        m.insert("backbone.pairwise_cosine".into(), 0.5);
        m.insert("head.activation_variance".into(), 0.08);
        m.insert("head.grad_norm".into(), head_grad);
        m.insert("head.pairwise_cosine".into(), 0.4);
        m
    }

    #[test]
    fn detects_gradient_domination() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // backbone grad_norm 200x head grad_norm
        for step in 0..30 {
            let warnings = diag.diagnose(step, &grad_domination_metrics(1.0, 0.005));
            if let Some(w) = warnings.iter().find(|w| w.signal == DiagnosticSignal::GradientDomination) {
                assert!(w.summary.contains("backbone"));
                assert!(w.confidence >= 0.3);
                return;
            }
        }
        panic!("Expected gradient domination warning");
    }

    #[test]
    fn no_domination_when_balanced() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Both components within 5x — well below 100x threshold
        for step in 0..30 {
            let warnings = diag.diagnose(step, &grad_domination_metrics(0.05, 0.01));
            let dom: Vec<_> = warnings
                .iter()
                .filter(|w| w.signal == DiagnosticSignal::GradientDomination)
                .collect();
            assert!(
                dom.is_empty(),
                "Step {}: should not fire gradient domination when balanced",
                step,
            );
        }
    }

    #[test]
    fn domination_ignores_dead_gradients() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // head has dead gradients (below 1e-7 floor), backbone is normal
        // Should NOT fire — dead gradient is not "suppressed", it's dead
        for step in 0..30 {
            let warnings = diag.diagnose(step, &grad_domination_metrics(0.5, 1e-9));
            let dom: Vec<_> = warnings
                .iter()
                .filter(|w| w.signal == DiagnosticSignal::GradientDomination)
                .collect();
            assert!(
                dom.is_empty(),
                "Step {}: should not fire when one component has dead gradients",
                step,
            );
        }
    }

    #[test]
    fn domination_works_with_grad_norm_min_keys() {
        // CRUX convention: reports grad_norm_min, not grad_norm
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["encoder".into(), "bow_head".into(), "category_head".into()],
        );

        // Simulate CRUX's 400:1 gradient ratio using grad_norm_min keys
        for step in 0..30 {
            let mut m = HashMap::new();
            m.insert("loss".into(), 5.0);
            m.insert("encoder.grad_norm_min".into(), 0.1);
            m.insert("encoder.pairwise_cosine".into(), 0.5);
            m.insert("encoder.activation_variance_min".into(), 0.1);
            m.insert("bow_head.grad_norm_min".into(), 2.0);       // dominant
            m.insert("bow_head.pairwise_cosine".into(), 0.3);
            m.insert("category_head.grad_norm_min".into(), 0.005); // starved (400x)
            m.insert("category_head.pairwise_cosine".into(), 0.9);
            let warnings = diag.diagnose(step, &m);
            if let Some(w) = warnings.iter().find(|w| w.signal == DiagnosticSignal::GradientDomination) {
                assert!(w.summary.contains("bow_head"), "should name bow_head as dominant");
                assert!(w.confidence >= 0.3);
                return;
            }
        }
        panic!("Expected gradient domination warning with grad_norm_min keys");
    }

    // -----------------------------------------------------------------------
    // Signal 12: MetricAnomaly
    // -----------------------------------------------------------------------

    #[test]
    fn detects_nan_metric() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into()],
        );

        // Warm up with healthy metrics
        for step in 0..15 {
            diag.diagnose(step, &healthy_metrics());
        }

        // Inject NaN
        let mut bad = healthy_metrics();
        bad.insert("backbone.grad_norm".into(), f64::NAN);
        let warnings = diag.diagnose(15, &bad);

        let anomaly: Vec<_> = warnings
            .iter()
            .filter(|w| w.signal == DiagnosticSignal::MetricAnomaly)
            .collect();
        assert!(!anomaly.is_empty(), "Should detect NaN metric");
        assert!(anomaly[0].confidence >= 0.9);
        assert!(anomaly[0].summary.contains("NaN"));
    }

    #[test]
    fn detects_inf_metric() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into()],
        );

        for step in 0..15 {
            diag.diagnose(step, &healthy_metrics());
        }

        let mut bad = healthy_metrics();
        bad.insert("backbone.activation_variance".into(), f64::INFINITY);
        let warnings = diag.diagnose(15, &bad);

        let anomaly: Vec<_> = warnings
            .iter()
            .filter(|w| w.signal == DiagnosticSignal::MetricAnomaly)
            .collect();
        assert!(!anomaly.is_empty(), "Should detect Inf metric");
        assert!(anomaly[0].summary.contains("Inf"));
    }

    #[test]
    fn no_anomaly_on_clean_metrics() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        for step in 0..30 {
            let warnings = diag.diagnose(step, &healthy_metrics());
            let anomalies: Vec<_> = warnings
                .iter()
                .filter(|w| w.signal == DiagnosticSignal::MetricAnomaly)
                .collect();
            assert!(
                anomalies.is_empty(),
                "Step {}: should not fire anomaly on clean metrics",
                step,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Signal 13: TrainValDivergence
    // -----------------------------------------------------------------------

    #[test]
    fn detects_train_val_divergence() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into()],
        );

        // Train loss decreasing, val loss increasing
        for step in 0..80 {
            let t = step as f64 / 80.0;
            let mut metrics = healthy_metrics();
            metrics.insert("loss".into(), 2.0 - t * 1.0); // 2.0 -> 1.0
            metrics.insert("val_loss".into(), 1.5 + t * 0.5); // 1.5 -> 2.0
            let warnings = diag.diagnose(step, &metrics);
            if let Some(w) = warnings
                .iter()
                .find(|w| w.signal == DiagnosticSignal::TrainValDivergence)
            {
                assert!(w.summary.contains("overfitting"));
                assert!(w.confidence >= 0.3);
                return;
            }
        }
        panic!("Expected train-val divergence warning");
    }

    #[test]
    fn no_divergence_when_both_decrease() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into()],
        );

        // Both train and val loss decreasing — healthy generalization
        for step in 0..80 {
            let t = step as f64 / 80.0;
            let mut metrics = healthy_metrics();
            metrics.insert("loss".into(), 2.0 - t * 0.5);
            metrics.insert("val_loss".into(), 2.5 - t * 0.3);
            let warnings = diag.diagnose(step, &metrics);
            let div: Vec<_> = warnings
                .iter()
                .filter(|w| w.signal == DiagnosticSignal::TrainValDivergence)
                .collect();
            assert!(
                div.is_empty(),
                "Step {}: should not fire divergence when both losses decrease",
                step,
            );
        }
    }

    #[test]
    fn no_divergence_without_val_loss() {
        let config = make_config();
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into()],
        );

        // Only train loss, no val_loss
        for step in 0..80 {
            let t = step as f64 / 80.0;
            let mut metrics = healthy_metrics();
            metrics.insert("loss".into(), 2.0 - t * 1.0);
            let warnings = diag.diagnose(step, &metrics);
            let div: Vec<_> = warnings
                .iter()
                .filter(|w| w.signal == DiagnosticSignal::TrainValDivergence)
                .collect();
            assert!(
                div.is_empty(),
                "Step {}: should not fire divergence without val_loss",
                step,
            );
        }
    }

    // ===================================================================
    // Rank-based shortcut discrimination (V1.5 / §15.7)
    // ===================================================================

    #[test]
    fn shortcut_suppressed_when_rank_growing() {
        let config = DiagnosticConfig {
            warmup_steps: 10,
            cadence: 1,
            history_window: 50,
            min_confidence: 0.1,
            shortcut_variance_explosion: 0.5, // 50% increase triggers
            ..Default::default()
        };
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Loss improving + variance exploding, but rank is GROWING — legitimate learning
        for step in 0..80 {
            let mut metrics = healthy_metrics();
            let t = step as f64 / 80.0;
            metrics.insert("loss".into(), 3.0 - t * 2.5);
            metrics.insert("backbone.pairwise_cosine".into(), 0.5 - t * 0.1);
            metrics.insert("backbone.activation_variance".into(), 0.1 * (1.0 + t * 3.0));
            // Rank growing — complex distributed features, not a shortcut
            metrics.insert("backbone.effective_rank".into(), 50.0 + t * 30.0);
            metrics.insert("head.pairwise_cosine".into(), 0.4);
            metrics.insert("head.activation_variance".into(), 0.08);

            let warnings = diag.diagnose(step, &metrics);
            assert!(
                warnings.iter().all(|w| w.signal != DiagnosticSignal::ShortcutLearning),
                "Step {}: shortcut learning should be suppressed when rank is growing",
                step,
            );
        }
    }

    #[test]
    fn shortcut_boosted_when_rank_drops() {
        let config = DiagnosticConfig {
            warmup_steps: 10,
            cadence: 1,
            history_window: 50,
            min_confidence: 0.1,
            shortcut_variance_explosion: 0.5,
            shortcut_rank_threshold: 0.3,
            ..Default::default()
        };
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        // Loss improving + variance exploding + rank COLLAPSING — strong shortcut signal.
        // Rank drops sharply at step 10 (step function) so the first/second half ratio
        // is well below 0.3 by the time the variance explosion triggers (~step 40).
        let mut fired = false;
        for step in 0..80 {
            let mut metrics = healthy_metrics();
            let t = step as f64 / 80.0;
            metrics.insert("loss".into(), 3.0 - t * 2.5);
            metrics.insert("backbone.pairwise_cosine".into(), 0.5 - t * 0.1);
            metrics.insert("backbone.activation_variance".into(), 0.1 * (1.0 + t * 3.0));
            // Sharp rank collapse — from 50 to 2 at step 10
            let rank = if step < 10 { 50.0 } else { 2.0 };
            metrics.insert("backbone.effective_rank".into(), rank);
            metrics.insert("head.pairwise_cosine".into(), 0.4);
            metrics.insert("head.activation_variance".into(), 0.08);

            let warnings = diag.diagnose(step, &metrics);
            if let Some(w) = warnings.iter().find(|w| w.signal == DiagnosticSignal::ShortcutLearning) {
                fired = true;
                assert!(
                    w.evidence.iter().any(|e| e.contains("Effective rank")),
                    "Warning should mention effective rank: {:?}", w.evidence,
                );
                break;
            }
        }
        assert!(fired, "Expected shortcut learning warning when rank is collapsing");
    }

    #[test]
    fn shortcut_unchanged_without_rank_data() {
        // Without effective_rank metric, behavior should be identical to existing
        let config = DiagnosticConfig {
            warmup_steps: 10,
            cadence: 1,
            history_window: 50,
            min_confidence: 0.1,
            shortcut_variance_explosion: 0.5,
            ..Default::default()
        };
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        let mut fired = false;
        for step in 0..80 {
            let mut metrics = healthy_metrics();
            let t = step as f64 / 80.0;
            metrics.insert("loss".into(), 3.0 - t * 2.5);
            metrics.insert("backbone.pairwise_cosine".into(), 0.5 - t * 0.1);
            metrics.insert("backbone.activation_variance".into(), 0.1 * (1.0 + t * 3.0));
            // No effective_rank metric — behavior unchanged
            metrics.insert("head.pairwise_cosine".into(), 0.4);
            metrics.insert("head.activation_variance".into(), 0.08);

            let warnings = diag.diagnose(step, &metrics);
            if let Some(w) = warnings.iter().find(|w| w.signal == DiagnosticSignal::ShortcutLearning) {
                fired = true;
                // Without rank data, confidence should be capped at 0.5
                assert!(
                    w.confidence <= 0.5,
                    "Without rank data, confidence should be <= 0.5, got {}",
                    w.confidence,
                );
                assert!(
                    w.evidence.iter().all(|e| !e.contains("Effective rank")),
                    "Should not mention effective rank without rank data",
                );
                break;
            }
        }
        assert!(fired, "Expected shortcut learning warning from variance explosion");
    }

    #[test]
    fn shortcut_rank_threshold_configurable() {
        // With threshold at 0.5, a rank at ~44% of initial (ratio 0.44) should trigger.
        // At default threshold 0.3, this ratio wouldn't trigger the rank boost.
        let config = DiagnosticConfig {
            warmup_steps: 10,
            cadence: 1,
            history_window: 50,
            min_confidence: 0.1,
            shortcut_variance_explosion: 0.5,
            shortcut_rank_threshold: 0.5, // higher threshold = easier to trigger
            ..Default::default()
        };
        let mut diag = DiagnosticLayer::new(
            config,
            vec!["backbone".into(), "head".into()],
        );

        let mut fired = false;
        for step in 0..80 {
            let mut metrics = healthy_metrics();
            let t = step as f64 / 80.0;
            metrics.insert("loss".into(), 3.0 - t * 2.5);
            metrics.insert("backbone.pairwise_cosine".into(), 0.5 - t * 0.1);
            metrics.insert("backbone.activation_variance".into(), 0.1 * (1.0 + t * 3.0));
            // Moderate rank drop — from 50 to 14 at step 10. By the time the variance
            // explosion triggers (~step 40), the half-window ratio is ~0.44.
            // Above 0.3 (default) but below 0.5 (custom).
            let rank = if step < 10 { 50.0 } else { 14.0 };
            metrics.insert("backbone.effective_rank".into(), rank);
            metrics.insert("head.pairwise_cosine".into(), 0.4);
            metrics.insert("head.activation_variance".into(), 0.08);

            let warnings = diag.diagnose(step, &metrics);
            if let Some(w) = warnings.iter().find(|w| w.signal == DiagnosticSignal::ShortcutLearning) {
                fired = true;
                assert!(
                    w.evidence.iter().any(|e| e.contains("Effective rank")),
                    "Should mention effective rank with custom threshold: {:?}",
                    w.evidence,
                );
                break;
            }
        }
        assert!(fired, "Expected shortcut learning warning with custom rank threshold 0.5");
    }
}
