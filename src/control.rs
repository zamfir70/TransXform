use std::collections::HashMap;

use crate::spec::ControlConfig;
use crate::types::*;

/// Implements control law logic (whitepaper §6).
///
/// Maps violations to actions, enforces cooldown and intervention budgets.
pub struct ControlLaws {
    config: ControlConfig,
    cooldown_until: HashMap<String, u64>,
    intervention_counts: HashMap<(String, Phase), u32>,
    /// Component → upstream component. Declared in spec roles.
    /// Used for upstream attribution: when a head collapses, check
    /// its declared upstream before deciding where to intervene.
    upstream_map: HashMap<String, String>,
}

impl ControlLaws {
    pub fn new(config: &ControlConfig) -> Self {
        Self {
            config: config.clone(),
            cooldown_until: HashMap::new(),
            intervention_counts: HashMap::new(),
            upstream_map: HashMap::new(),
        }
    }

    /// Set the upstream dependency map from the training spec's role declarations.
    /// Call after construction if the spec declares `upstream` on any component.
    pub fn set_upstream_map(&mut self, map: HashMap<String, String>) {
        self.upstream_map = map;
    }

    /// Decide the action for a hard violation (whitepaper §6.1).
    ///
    /// Returns `None` if the component is in cooldown or budget exhausted.
    /// `metrics` is used for upstream attribution — if a head collapses but its
    /// backbone inputs are also degenerate, route intervention to backbone instead.
    pub fn hard_action(
        &self,
        violation: &Violation,
        phase: Phase,
        step: u64,
        allowed: Option<&[String]>,
        metrics: &MetricSnapshot,
    ) -> Option<Action> {
        // Check cooldown
        if self.is_in_cooldown(&violation.component, step) {
            return None;
        }

        // Check budget
        if self.budget_exhausted(&violation.component, phase) {
            return None;
        }

        // Determine action based on violation pattern + upstream attribution
        let action = self.select_hard_action(violation, metrics);

        // Check if action is allowed in current phase
        if let Some(allowed_list) = allowed {
            if !allowed_list.iter().any(|a| a == action.action_name()) {
                return None;
            }
        }

        Some(action)
    }

    /// Decide the action for a soft violation (whitepaper §6.2).
    pub fn soft_action(
        &self,
        violation: &Violation,
        _phase: Phase,
        allowed: Option<&[String]>,
    ) -> Option<Action> {
        let action = self.select_soft_action(violation);

        if let Some(allowed_list) = allowed {
            if !allowed_list.iter().any(|a| a == action.action_name()) {
                return None;
            }
        }

        Some(action)
    }

    /// Record that a hard intervention was executed. Updates cooldown and budget.
    pub fn record_intervention(
        &mut self,
        component: &str,
        phase: Phase,
        step: u64,
    ) {
        // Set cooldown
        self.cooldown_until.insert(
            component.to_string(),
            step + self.config.cooldown_steps,
        );

        // Increment budget counter
        let key = (component.to_string(), phase);
        *self.intervention_counts.entry(key).or_insert(0) += 1;
    }

    /// Check if a component is in cooldown.
    pub fn is_in_cooldown(&self, component: &str, step: u64) -> bool {
        self.cooldown_until
            .get(component)
            .map_or(false, |&until| step < until)
    }

    /// Check if intervention budget is exhausted for a component in a phase.
    pub fn budget_exhausted(&self, component: &str, phase: Phase) -> bool {
        let key = (component.to_string(), phase);
        self.intervention_counts
            .get(&key)
            .map_or(false, |&count| count >= self.config.max_hard_interventions)
    }

    /// Get total hard intervention count for a component in a phase.
    pub fn intervention_count(&self, component: &str, phase: Phase) -> u32 {
        let key = (component.to_string(), phase);
        self.intervention_counts.get(&key).copied().unwrap_or(0)
    }

    /// Get all intervention counts for the current phase (for phase controller).
    pub fn counts_for_phase(&self, phase: Phase) -> HashMap<String, u32> {
        self.intervention_counts
            .iter()
            .filter(|((_, p), _)| *p == phase)
            .map(|((c, _), &count)| (c.clone(), count))
            .collect()
    }

    /// Get the damping factor.
    pub fn damping_factor(&self) -> f64 {
        self.config.damping_factor
    }

    /// Extract mutable state for checkpointing.
    ///
    /// Encodes `intervention_counts` tuple keys as `"component::phase_name"`
    /// strings for JSON compatibility.
    pub fn save_state(&self) -> crate::checkpoint::ControlLawsState {
        let counts = self
            .intervention_counts
            .iter()
            .map(|((comp, phase), &count)| (format!("{}::{}", comp, phase), count))
            .collect();
        crate::checkpoint::ControlLawsState {
            cooldown_until: self.cooldown_until.clone(),
            intervention_counts: counts,
        }
    }

    /// Restore mutable state from a checkpoint.
    ///
    /// Decodes `"component::phase_name"` string keys back to `(String, Phase)` tuples.
    pub fn restore_state(&mut self, state: crate::checkpoint::ControlLawsState) {
        self.cooldown_until = state.cooldown_until;
        self.intervention_counts = state
            .intervention_counts
            .into_iter()
            .filter_map(|(key, count)| {
                let parts: Vec<&str> = key.splitn(2, "::").collect();
                if parts.len() == 2 {
                    let phase: Phase = parts[1].parse().ok()?;
                    Some(((parts[0].to_string(), phase), count))
                } else {
                    None
                }
            })
            .collect();
    }

    /// Reset counts for a new phase (called on phase transition).
    pub fn on_phase_change(&mut self, _new_phase: Phase) {
        // Note: we do NOT clear intervention_counts here because the phase
        // controller needs historical counts for regression decisions.
        // Cooldowns persist across phases.
    }

    // -----------------------------------------------------------------------
    // Private: action selection
    // -----------------------------------------------------------------------

    fn select_hard_action(&self, violation: &Violation, metrics: &MetricSnapshot) -> Action {
        let name = violation.invariant_name.to_lowercase();
        let component = violation.component.clone();

        // Pattern matching on violation characteristics (whitepaper §6.1)
        if name.contains("pairwise_cosine") || name.contains("cosine") {
            // Representation collapse — check upstream attribution first.
            // If the component's input source (backbone) also shows degenerate
            // metrics, reinitializing the head alone won't help.
            if let Some(upstream) = self.find_degenerate_upstream(&component, metrics) {
                log::info!(
                    "Upstream attribution: {} collapse traced to degenerate backbone '{}' — routing intervention upstream",
                    component, upstream
                );
                Action::Reinitialize { component: upstream }
            } else {
                Action::Reinitialize { component }
            }
        } else if name.contains("grad_norm") && violation.direction == ThresholdDirection::Min {
            // Dead submodule: gradient norm below floor
            Action::Reinitialize { component }
        } else if name.contains("loss_explosion") || name.contains("loss_stability") {
            // Loss explosion: reduce LR globally
            Action::AdjustLr {
                component,
                factor: self.config.damping_factor,
            }
        } else if name.contains("activation_variance") && violation.direction == ThresholdDirection::Min {
            // Variance collapse — check upstream attribution
            if let Some(upstream) = self.find_degenerate_upstream(&component, metrics) {
                log::info!(
                    "Upstream attribution: {} variance collapse traced to '{}' — routing upstream",
                    component, upstream
                );
                Action::Reinitialize { component: upstream }
            } else {
                Action::Reinitialize { component }
            }
        } else if name.contains("entropy") && violation.direction == ThresholdDirection::Min {
            // Dead attention head: entropy at zero
            Action::Reinitialize { component }
        } else {
            // Default: reinitialize the component
            Action::Reinitialize { component }
        }
    }

    /// Check if a component's upstream source is degenerate.
    ///
    /// Walks the upstream dependency chain declared in the spec's role
    /// declarations. If a component's upstream is also showing degenerate
    /// metrics (high cosine or low variance), the collapse is a symptom —
    /// route the intervention to the upstream source instead.
    ///
    /// Falls back to the "backbone" heuristic for components without an
    /// explicit upstream declaration (backward-compatible with v1.0 specs).
    fn find_degenerate_upstream(&self, component: &str, metrics: &MetricSnapshot) -> Option<String> {
        if component == "global" {
            return None;
        }

        // Resolve the upstream: spec declaration takes priority, then default to "backbone"
        let upstream = if let Some(declared) = self.upstream_map.get(component) {
            declared.as_str()
        } else if component != "backbone" {
            "backbone"
        } else {
            return None; // backbone has no upstream
        };

        // Check the upstream component's metrics for degeneration
        if self.is_component_degenerate(upstream, metrics) {
            // Walk one more hop: check if the upstream's upstream is also degenerate
            // (catches multi-level poisoning like backbone → compressor → emission_head)
            if let Some(grandparent) = self.upstream_map.get(upstream) {
                if self.is_component_degenerate(grandparent, metrics) {
                    log::info!(
                        "Multi-level upstream attribution: {} → {} → {} (all degenerate, routing to {})",
                        component, upstream, grandparent, grandparent
                    );
                    return Some(grandparent.clone());
                }
            }
            Some(upstream.to_string())
        } else {
            None
        }
    }

    /// Check if a component's metrics indicate degenerate representations.
    fn is_component_degenerate(&self, component: &str, metrics: &MetricSnapshot) -> bool {
        let cosine_key = format!("{}.pairwise_cosine", component);
        let variance_key = format!("{}.activation_variance", component);
        // Also check the legacy key format
        let variance_key_min = format!("{}.activation_variance_min", component);

        let cosine_degenerate = metrics
            .get(&cosine_key)
            .map_or(false, |&v| v > 0.98);
        let variance_degenerate = metrics
            .get(&variance_key)
            .or_else(|| metrics.get(&variance_key_min))
            .map_or(false, |&v| v < 0.0001);

        cosine_degenerate || variance_degenerate
    }

    fn select_soft_action(&self, violation: &Violation) -> Action {
        let name = violation.invariant_name.to_lowercase();
        let component = violation.component.clone();

        if name.contains("spike") || name.contains("grad_norm") {
            // Gradient spike: reduce LR
            Action::AdjustLr {
                component,
                factor: self.config.damping_factor,
            }
        } else if name.contains("drift") {
            // Representation drift: rescale weights toward threshold.
            // ratio < 1.0 for Max violations (too high), > 1.0 for Min (too low).
            // Dampened by damping_factor so corrections are gentle.
            let ratio = if violation.observed.abs() > 1e-12 {
                violation.threshold / violation.observed
            } else {
                self.config.damping_factor
            };
            let factor = 1.0 + self.config.damping_factor * (ratio - 1.0);
            Action::Rescale {
                component,
                factor,
            }
        } else {
            // Default soft action: dampen LR
            Action::AdjustLr {
                component,
                factor: self.config.damping_factor,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ControlConfig;

    fn test_config() -> ControlConfig {
        ControlConfig {
            cooldown_steps: 50,
            max_hard_interventions: 3,
            hysteresis_margin: 0.05,
            hysteresis_pct: 0.0,
            catastrophic_overrides: HashMap::new(),
            damping_factor: 0.5,
            base_lr: None,
            regret_window_steps: 100,
            readiness_gate: true,
            readiness_patience_steps: 200,
            max_threshold_relaxation: 0.02,
            discovery_mode: false,
            shadow_step: false,
        }
    }

    fn empty_metrics() -> MetricSnapshot {
        MetricSnapshot::new()
    }

    fn cosine_violation(step: u64) -> Violation {
        Violation {
            invariant_name: "pairwise_cosine".into(),
            component: "head".into(),
            severity: Severity::Hard,
            observed: 0.99,
            threshold: 0.95,
            direction: ThresholdDirection::Max,
            step,
            passive: false,
        }
    }

    #[test]
    fn hard_action_reinitialize_on_cosine() {
        let laws = ControlLaws::new(&test_config());
        let v = cosine_violation(100);
        let action = laws.hard_action(&v, Phase::RepresentationFormation, 100, None, &empty_metrics());
        assert!(matches!(action, Some(Action::Reinitialize { .. })));
    }

    #[test]
    fn hard_action_adjust_lr_on_loss_explosion() {
        let laws = ControlLaws::new(&test_config());
        let v = Violation {
            invariant_name: "loss_explosion_factor".into(),
            component: "global".into(),
            severity: Severity::Hard,
            observed: 10.0,
            threshold: 3.0,
            direction: ThresholdDirection::Max,
            step: 100,
            passive: false,
        };
        let action = laws.hard_action(&v, Phase::Bootstrap, 100, None, &empty_metrics());
        assert!(matches!(action, Some(Action::AdjustLr { .. })));
    }

    #[test]
    fn cooldown_blocks_repeated_intervention() {
        let mut laws = ControlLaws::new(&test_config());
        let v = cosine_violation(100);

        // First intervention succeeds
        let action1 = laws.hard_action(&v, Phase::Bootstrap, 100, None, &empty_metrics());
        assert!(action1.is_some());
        laws.record_intervention("head", Phase::Bootstrap, 100);

        // During cooldown (step 100 + 50 = 150), blocked
        let v2 = cosine_violation(120);
        let action2 = laws.hard_action(&v2, Phase::Bootstrap, 120, None, &empty_metrics());
        assert!(action2.is_none());

        // After cooldown, succeeds
        let v3 = cosine_violation(151);
        let action3 = laws.hard_action(&v3, Phase::Bootstrap, 151, None, &empty_metrics());
        assert!(action3.is_some());
    }

    #[test]
    fn budget_exhaustion() {
        let mut laws = ControlLaws::new(&test_config());

        // Use all 3 interventions
        for step in [0u64, 51, 102] {
            laws.record_intervention("head", Phase::Bootstrap, step);
        }

        assert!(laws.budget_exhausted("head", Phase::Bootstrap));

        let v = cosine_violation(200);
        let action = laws.hard_action(&v, Phase::Bootstrap, 200, None, &empty_metrics());
        assert!(action.is_none());
    }

    #[test]
    fn phase_allowed_interventions() {
        let laws = ControlLaws::new(&test_config());
        let v = cosine_violation(100);

        // Reinitialize not in allowed list
        let allowed = vec!["adjust_lr".to_string(), "freeze".to_string()];
        let action = laws.hard_action(&v, Phase::Refinement, 100, Some(&allowed), &empty_metrics());
        assert!(action.is_none());
    }

    #[test]
    fn upstream_attribution_routes_to_backbone() {
        let laws = ControlLaws::new(&test_config());
        let v = cosine_violation(100);

        // Backbone has degenerate cosine — head collapse is a symptom
        let mut metrics = MetricSnapshot::new();
        metrics.insert("backbone.pairwise_cosine".into(), 0.99);

        let action = laws.hard_action(&v, Phase::Bootstrap, 100, None, &metrics);
        match action {
            Some(Action::Reinitialize { component }) => {
                assert_eq!(component, "backbone", "Should route to degenerate backbone");
            }
            other => panic!("Expected Reinitialize(backbone), got {:?}", other),
        }
    }

    #[test]
    fn upstream_attribution_stays_local_when_backbone_healthy() {
        let laws = ControlLaws::new(&test_config());
        let v = cosine_violation(100);

        // Backbone is healthy — head collapse is genuine
        let mut metrics = MetricSnapshot::new();
        metrics.insert("backbone.pairwise_cosine".into(), 0.5);
        metrics.insert("backbone.activation_variance_min".into(), 1.0);

        let action = laws.hard_action(&v, Phase::Bootstrap, 100, None, &metrics);
        match action {
            Some(Action::Reinitialize { component }) => {
                assert_eq!(component, "head", "Should target head when backbone is healthy");
            }
            other => panic!("Expected Reinitialize(head), got {:?}", other),
        }
    }

    #[test]
    fn soft_action_for_spike() {
        let laws = ControlLaws::new(&test_config());
        let v = Violation {
            invariant_name: "grad_norm_spike_threshold".into(),
            component: "backbone".into(),
            severity: Severity::Soft,
            observed: 150.0,
            threshold: 100.0,
            direction: ThresholdDirection::Max,
            step: 50,
            passive: false,
        };
        let action = laws.soft_action(&v, Phase::Stabilization, None);
        assert!(matches!(action, Some(Action::AdjustLr { .. })));
    }

    #[test]
    fn upstream_attribution_follows_declared_upstream() {
        let mut laws = ControlLaws::new(&test_config());
        // Declare: emission_head → compressor → backbone
        let mut upstream = HashMap::new();
        upstream.insert("emission_head".into(), "compressor".into());
        upstream.insert("compressor".into(), "backbone".into());
        laws.set_upstream_map(upstream);

        // emission_head collapses, compressor is degenerate
        let v = Violation {
            invariant_name: "pairwise_cosine".into(),
            component: "emission_head".into(),
            severity: Severity::Hard,
            observed: 0.99,
            threshold: 0.95,
            direction: ThresholdDirection::Max,
            step: 100,
            passive: false,
        };
        let mut metrics = MetricSnapshot::new();
        metrics.insert("compressor.pairwise_cosine".into(), 0.99); // degenerate
        metrics.insert("backbone.pairwise_cosine".into(), 0.5);    // healthy

        let action = laws.hard_action(&v, Phase::Bootstrap, 100, None, &metrics);
        match action {
            Some(Action::Reinitialize { component }) => {
                assert_eq!(component, "compressor", "Should route to declared degenerate upstream");
            }
            other => panic!("Expected Reinitialize(compressor), got {:?}", other),
        }
    }

    #[test]
    fn upstream_attribution_multi_level_poisoning() {
        let mut laws = ControlLaws::new(&test_config());
        let mut upstream = HashMap::new();
        upstream.insert("emission_head".into(), "compressor".into());
        upstream.insert("compressor".into(), "backbone".into());
        laws.set_upstream_map(upstream);

        // All three are degenerate — should route to backbone (grandparent)
        let v = Violation {
            invariant_name: "pairwise_cosine".into(),
            component: "emission_head".into(),
            severity: Severity::Hard,
            observed: 0.99,
            threshold: 0.95,
            direction: ThresholdDirection::Max,
            step: 100,
            passive: false,
        };
        let mut metrics = MetricSnapshot::new();
        metrics.insert("compressor.pairwise_cosine".into(), 0.99);
        metrics.insert("backbone.pairwise_cosine".into(), 0.99);

        let action = laws.hard_action(&v, Phase::Bootstrap, 100, None, &metrics);
        match action {
            Some(Action::Reinitialize { component }) => {
                assert_eq!(component, "backbone", "Should trace through to degenerate grandparent");
            }
            other => panic!("Expected Reinitialize(backbone), got {:?}", other),
        }
    }

    #[test]
    fn soft_drift_rescale_factor_not_one() {
        let laws = ControlLaws::new(&test_config());
        let v = Violation {
            invariant_name: "representation_drift".into(),
            component: "backbone".into(),
            severity: Severity::Soft,
            observed: 0.8, // drifted to 0.8
            threshold: 0.5, // should be below 0.5
            direction: ThresholdDirection::Max,
            step: 100,
            passive: false,
        };
        let action = laws.soft_action(&v, Phase::Stabilization, None);
        match action {
            Some(Action::Rescale { factor, .. }) => {
                // ratio = 0.5/0.8 = 0.625, factor = 1 + 0.5*(0.625-1) = 0.8125
                assert!(
                    (factor - 0.8125).abs() < 1e-6,
                    "Expected rescale factor ~0.8125, got {}",
                    factor,
                );
                assert!(factor < 1.0, "Max violation should scale DOWN");
            }
            other => panic!("Expected Rescale, got {:?}", other),
        }
    }

    #[test]
    fn soft_drift_rescale_min_direction() {
        let laws = ControlLaws::new(&test_config());
        let v = Violation {
            invariant_name: "variance_drift".into(),
            component: "head".into(),
            severity: Severity::Soft,
            observed: 0.001,  // too low
            threshold: 0.01,  // should be above 0.01
            direction: ThresholdDirection::Min,
            step: 100,
            passive: false,
        };
        let action = laws.soft_action(&v, Phase::Stabilization, None);
        match action {
            Some(Action::Rescale { factor, .. }) => {
                // ratio = 0.01/0.001 = 10.0, factor = 1 + 0.5*(10-1) = 5.5
                assert!(factor > 1.0, "Min violation should scale UP, got {}", factor);
            }
            other => panic!("Expected Rescale, got {:?}", other),
        }
    }

    #[test]
    fn counts_for_phase() {
        let mut laws = ControlLaws::new(&test_config());
        laws.record_intervention("head", Phase::Bootstrap, 0);
        laws.record_intervention("head", Phase::Bootstrap, 51);
        laws.record_intervention("backbone", Phase::RepresentationFormation, 100);

        let counts = laws.counts_for_phase(Phase::Bootstrap);
        assert_eq!(counts.get("head"), Some(&2));
        assert!(counts.get("backbone").is_none());
    }
}
