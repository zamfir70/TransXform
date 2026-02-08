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
}

impl ControlLaws {
    pub fn new(config: &ControlConfig) -> Self {
        Self {
            config: config.clone(),
            cooldown_until: HashMap::new(),
            intervention_counts: HashMap::new(),
        }
    }

    /// Decide the action for a hard violation (whitepaper §6.1).
    ///
    /// Returns `None` if the component is in cooldown or budget exhausted.
    pub fn hard_action(
        &self,
        violation: &Violation,
        phase: Phase,
        step: u64,
        allowed: Option<&[String]>,
    ) -> Option<Action> {
        // Check cooldown
        if self.is_in_cooldown(&violation.component, step) {
            return None;
        }

        // Check budget
        if self.budget_exhausted(&violation.component, phase) {
            return None;
        }

        // Determine action based on violation pattern
        let action = self.select_hard_action(violation);

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

    /// Reset counts for a new phase (called on phase transition).
    pub fn on_phase_change(&mut self, _new_phase: Phase) {
        // Note: we do NOT clear intervention_counts here because the phase
        // controller needs historical counts for regression decisions.
        // Cooldowns persist across phases.
    }

    // -----------------------------------------------------------------------
    // Private: action selection
    // -----------------------------------------------------------------------

    fn select_hard_action(&self, violation: &Violation) -> Action {
        let name = violation.invariant_name.to_lowercase();
        let component = violation.component.clone();

        // Pattern matching on violation characteristics (whitepaper §6.1)
        if name.contains("pairwise_cosine") || name.contains("cosine") {
            // Representation collapse: cosine above threshold AND likely zero gradient
            Action::Reinitialize { component }
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
            // Variance collapse: reinitialize + noise
            Action::Reinitialize { component }
        } else if name.contains("entropy") && violation.direction == ThresholdDirection::Min {
            // Dead attention head: entropy at zero
            Action::Reinitialize { component }
        } else {
            // Default: reinitialize the component
            Action::Reinitialize { component }
        }
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
            // Representation drift: rescale
            Action::Rescale {
                component,
                factor: 1.0, // rescale to baseline (implementation would compute actual factor)
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
            damping_factor: 0.5,
            base_lr: None,
            regret_window_steps: 100,
        }
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
        }
    }

    #[test]
    fn hard_action_reinitialize_on_cosine() {
        let laws = ControlLaws::new(&test_config());
        let v = cosine_violation(100);
        let action = laws.hard_action(&v, Phase::RepresentationFormation, 100, None);
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
        };
        let action = laws.hard_action(&v, Phase::Bootstrap, 100, None);
        assert!(matches!(action, Some(Action::AdjustLr { .. })));
    }

    #[test]
    fn cooldown_blocks_repeated_intervention() {
        let mut laws = ControlLaws::new(&test_config());
        let v = cosine_violation(100);

        // First intervention succeeds
        let action1 = laws.hard_action(&v, Phase::Bootstrap, 100, None);
        assert!(action1.is_some());
        laws.record_intervention("head", Phase::Bootstrap, 100);

        // During cooldown (step 100 + 50 = 150), blocked
        let v2 = cosine_violation(120);
        let action2 = laws.hard_action(&v2, Phase::Bootstrap, 120, None);
        assert!(action2.is_none());

        // After cooldown, succeeds
        let v3 = cosine_violation(151);
        let action3 = laws.hard_action(&v3, Phase::Bootstrap, 151, None);
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
        let action = laws.hard_action(&v, Phase::Bootstrap, 200, None);
        assert!(action.is_none());
    }

    #[test]
    fn phase_allowed_interventions() {
        let laws = ControlLaws::new(&test_config());
        let v = cosine_violation(100);

        // Reinitialize not in allowed list
        let allowed = vec!["adjust_lr".to_string(), "freeze".to_string()];
        let action = laws.hard_action(&v, Phase::Refinement, 100, Some(&allowed));
        assert!(action.is_none());
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
        };
        let action = laws.soft_action(&v, Phase::Stabilization, None);
        assert!(matches!(action, Some(Action::AdjustLr { .. })));
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
