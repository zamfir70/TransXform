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

use std::collections::VecDeque;

use crate::types::*;

/// Configuration for the diagnostic layer.
#[derive(Debug, Clone)]
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
        }
    }
}

/// The V2 diagnostic layer. Observes training dynamics and surfaces
/// advisory warnings when patterns are inconsistent with stated intent.
///
/// Does not intervene. Does not modify model state. Does not claim certainty.
pub struct DiagnosticLayer {
    config: DiagnosticConfig,
    /// Rolling window of recent metric snapshots.
    history: VecDeque<MetricSnapshot>,
    /// Component names from the training spec.
    components: Vec<String>,
    /// All warnings emitted during the run (including resolved ones).
    warnings: Vec<DiagnosticWarning>,
    /// Active (unresolved) warning signals — used for deduplication.
    active_signals: Vec<(DiagnosticSignal, String)>, // (signal, component_or_global)
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
        }
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

        // Run all five detectors
        self.detect_unused_capacity(step, metrics, &mut new_warnings);
        self.detect_missing_structural_signal(step, metrics, &mut new_warnings);
        self.detect_loss_representation_misalignment(step, &mut new_warnings);
        self.detect_unlearnable_regime(step, metrics, &mut new_warnings);
        self.detect_shortcut_learning(step, &mut new_warnings);

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
            let var_key = format!("{}.activation_variance", component);
            let grad_key = format!("{}.grad_norm", component);

            // Count how many recent steps had near-zero variance
            let near_zero_var_count = self
                .history
                .iter()
                .filter(|m| {
                    m.get(&var_key)
                        .map_or(false, |&v| v < self.config.unused_capacity_variance_floor)
                })
                .count();

            let near_zero_grad_count = self
                .history
                .iter()
                .filter(|m| {
                    m.get(&grad_key)
                        .map_or(false, |&v| v < self.config.unused_capacity_grad_floor)
                })
                .count();

            let window = self.history.len();
            let var_ratio = near_zero_var_count as f64 / window as f64;
            let grad_ratio = near_zero_grad_count as f64 / window as f64;

            if var_ratio >= self.config.unused_capacity_persistence
                || grad_ratio >= self.config.unused_capacity_persistence
            {
                let current_var = metrics.get(&var_key).copied();
                let current_grad = metrics.get(&grad_key).copied();

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
            let var_key = format!("{}.activation_variance", component);
            let cos_key = format!("{}.pairwise_cosine", component);

            let mut stagnant = true;

            for key in [&var_key, &cos_key] {
                let first_val = Self::mean_metric(first, key);
                let second_val = Self::mean_metric(second, key);
                if let (Some(fv), Some(sv)) = (first_val, second_val) {
                    let change = (sv - fv).abs() / fv.abs().max(1e-10);
                    if change > self.config.alignment_repr_stagnation_threshold {
                        stagnant = false;
                    }
                }
            }

            if stagnant {
                // Check we actually had metrics for this component
                let has_data = first.iter().any(|m| m.contains_key(&var_key) || m.contains_key(&cos_key));
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
            let grad_key = format!("{}.grad_norm", component);

            // Get recent grad norms
            let recent_grads: Vec<f64> = self
                .history
                .iter()
                .rev()
                .take(10)
                .filter_map(|m| m.get(&grad_key).copied())
                .collect();

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

        for component in &self.components {
            let cos_key = format!("{}.pairwise_cosine", component);
            let var_key = format!("{}.activation_variance", component);

            let first_cos = Self::mean_metric(first, &cos_key);
            let second_cos = Self::mean_metric(second, &cos_key);
            let first_var = Self::mean_metric(first, &var_key);
            let second_var = Self::mean_metric(second, &var_key);

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

            if signals > 0 {
                collapsing_components.push(component.clone());
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

            let confidence = (collapsing_components.len() as f64 / self.components.len() as f64)
                .min(0.9)
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
}
