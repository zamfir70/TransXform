use std::collections::HashMap;

use crate::spec::PhasesDecl;
use crate::types::*;

/// Finite state machine over training phases (whitepaper §8).
pub struct PhaseController {
    current: Phase,
    phases_config: PhasesDecl,
    consecutive_satisfied: u64,
    phase_entry_step: u64,
    regression_count: HashMap<String, u32>,
    phase_history: Vec<PhaseTransition>,
    /// Step when the readiness gate first blocked a forward transition.
    /// `None` when not blocked.
    readiness_blocked_since: Option<u64>,
}

impl PhaseController {
    pub fn new(phases: &PhasesDecl) -> Self {
        Self {
            current: Phase::Bootstrap,
            phases_config: phases.clone(),
            consecutive_satisfied: 0,
            phase_entry_step: 0,
            regression_count: HashMap::new(),
            phase_history: Vec::new(),
            readiness_blocked_since: None,
        }
    }

    pub fn current_phase(&self) -> Phase {
        self.current
    }

    /// Get phase-specific thresholds for the current phase.
    pub fn current_thresholds(&self) -> HashMap<String, f64> {
        self.thresholds_for(self.current)
    }

    /// Get thresholds for a specific phase.
    pub fn thresholds_for(&self, phase: Phase) -> HashMap<String, f64> {
        let decl = match phase {
            Phase::Bootstrap => &self.phases_config.bootstrap,
            Phase::RepresentationFormation => &self.phases_config.representation_formation,
            Phase::Stabilization => &self.phases_config.stabilization,
            Phase::Refinement => &self.phases_config.refinement,
            Phase::Aborted => &None,
        };
        decl.as_ref()
            .map(|d| d.thresholds.clone())
            .unwrap_or_default()
    }

    /// Get allowed interventions for the current phase (None = all allowed).
    pub fn allowed_interventions(&self) -> Option<Vec<String>> {
        let decl = match self.current {
            Phase::Bootstrap => &self.phases_config.bootstrap,
            Phase::RepresentationFormation => &self.phases_config.representation_formation,
            Phase::Stabilization => &self.phases_config.stabilization,
            Phase::Refinement => &self.phases_config.refinement,
            Phase::Aborted => &None,
        };
        decl.as_ref()
            .and_then(|d| d.allowed_interventions.clone())
    }

    /// How long the readiness gate has been blocking a forward transition.
    /// Returns `None` if not currently blocked.
    pub fn readiness_blocked_since(&self) -> Option<u64> {
        self.readiness_blocked_since
    }

    /// Update phase state based on violations and intervention counts.
    ///
    /// `readiness_ok`: if `false`, forward phase transitions are blocked even
    /// when the transition guard is satisfied. This implements the V1.3 phase
    /// readiness gate — the supervisor checks whether current metrics satisfy
    /// the next phase's thresholds before allowing advancement.
    ///
    /// Returns `Some(PhaseTransition)` if a transition occurred.
    pub fn update(
        &mut self,
        violations: &[Violation],
        hard_intervention_counts: &HashMap<String, u32>,
        max_hard_interventions: u32,
        step: u64,
        readiness_ok: bool,
    ) -> Option<PhaseTransition> {
        if self.current.is_terminal() {
            return None;
        }

        let has_hard_violations = violations
            .iter()
            .any(|v| v.severity == Severity::Hard);

        // Track consecutive steps with all hard invariants satisfied
        if has_hard_violations {
            self.consecutive_satisfied = 0;
        } else {
            self.consecutive_satisfied += 1;
        }

        // Check for phase regression: any component exhausted its intervention budget
        for (component, &count) in hard_intervention_counts {
            if count >= max_hard_interventions {
                // Check if this component has already caused a regression
                let reg_count = self.regression_count.entry(component.clone()).or_insert(0);
                if *reg_count >= 1 {
                    // Second regression attempt → abort
                    return Some(self.transition_to(
                        Phase::Aborted,
                        step,
                        format!(
                            "Component '{}' exhausted intervention budget across multiple phases",
                            component
                        ),
                    ));
                }

                if let Some(prev) = self.current.prev() {
                    *reg_count += 1;
                    return Some(self.transition_to(
                        prev,
                        step,
                        format!(
                            "Component '{}' exhausted intervention budget ({}/{})",
                            component, count, max_hard_interventions
                        ),
                    ));
                } else {
                    // Already at Bootstrap and budget exhausted → abort
                    return Some(self.transition_to(
                        Phase::Aborted,
                        step,
                        format!(
                            "Component '{}' exhausted intervention budget in Bootstrap",
                            component
                        ),
                    ));
                }
            }
        }

        // Check for phase advance
        let guard_steps = self.transition_guard_steps();
        if self.consecutive_satisfied >= guard_steps {
            if readiness_ok {
                self.readiness_blocked_since = None;
                if let Some(next) = self.current.next() {
                    return Some(self.transition_to(
                        next,
                        step,
                        format!(
                            "All hard invariants satisfied for {} consecutive steps",
                            self.consecutive_satisfied
                        ),
                    ));
                }
            } else {
                // Readiness gate blocking — track how long
                if self.readiness_blocked_since.is_none() {
                    self.readiness_blocked_since = Some(step);
                }
            }
        }

        // Check for max_duration_steps exceeded
        if self.is_phase_expired(step) {
            if self.consecutive_satisfied > 0 && readiness_ok {
                // Some progress and ready — advance
                self.readiness_blocked_since = None;
                if let Some(next) = self.current.next() {
                    return Some(self.transition_to(
                        next,
                        step,
                        format!(
                            "Phase duration exceeded; advancing with {} consecutive clean steps",
                            self.consecutive_satisfied
                        ),
                    ));
                }
            }
            // Duration exceeded but not ready or invariants failing — log but don't force
        }

        None
    }

    /// Check if max_duration_steps for the current phase has been exceeded.
    pub fn is_phase_expired(&self, step: u64) -> bool {
        let decl = match self.current {
            Phase::Bootstrap => &self.phases_config.bootstrap,
            Phase::RepresentationFormation => &self.phases_config.representation_formation,
            Phase::Stabilization => &self.phases_config.stabilization,
            Phase::Refinement => &self.phases_config.refinement,
            Phase::Aborted => &None,
        };

        if let Some(d) = decl {
            if let Some(max_steps) = d.max_duration_steps {
                return step.saturating_sub(self.phase_entry_step) >= max_steps;
            }
        }
        false
    }

    /// Get the full phase history.
    pub fn history(&self) -> &[PhaseTransition] {
        &self.phase_history
    }

    /// Get the consecutive steps with all hard invariants satisfied.
    pub fn consecutive_satisfied(&self) -> u64 {
        self.consecutive_satisfied
    }

    /// Get the transition guard steps for the current phase.
    pub fn transition_guard_steps_public(&self) -> u64 {
        self.transition_guard_steps()
    }

    /// Extract mutable state for checkpointing.
    pub fn save_state(&self) -> crate::checkpoint::PhaseControllerState {
        crate::checkpoint::PhaseControllerState {
            current: self.current,
            consecutive_satisfied: self.consecutive_satisfied,
            phase_entry_step: self.phase_entry_step,
            regression_count: self.regression_count.clone(),
            phase_history: self.phase_history.clone(),
            readiness_blocked_since: self.readiness_blocked_since,
        }
    }

    /// Restore mutable state from a checkpoint.
    /// The `phases_config` is rebuilt from the spec, not checkpointed.
    pub fn restore_state(&mut self, state: crate::checkpoint::PhaseControllerState) {
        self.current = state.current;
        self.consecutive_satisfied = state.consecutive_satisfied;
        self.phase_entry_step = state.phase_entry_step;
        self.regression_count = state.regression_count;
        self.phase_history = state.phase_history;
        self.readiness_blocked_since = state.readiness_blocked_since;
    }

    /// Force transition to Aborted.
    pub fn abort(&mut self, step: u64, reason: String) -> PhaseTransition {
        self.transition_to(Phase::Aborted, step, reason)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn transition_guard_steps(&self) -> u64 {
        let decl = match self.current {
            Phase::Bootstrap => &self.phases_config.bootstrap,
            Phase::RepresentationFormation => &self.phases_config.representation_formation,
            Phase::Stabilization => &self.phases_config.stabilization,
            Phase::Refinement => &self.phases_config.refinement,
            Phase::Aborted => &None,
        };

        decl.as_ref()
            .and_then(|d| d.transition_guard.as_ref())
            .map(|g| g.all_hard_invariants_satisfied_for)
            .unwrap_or(50) // default: 50 clean steps to advance
    }

    fn transition_to(&mut self, to: Phase, step: u64, reason: String) -> PhaseTransition {
        let transition = PhaseTransition {
            from: self.current,
            to,
            step,
            reason,
        };
        self.phase_history.push(transition.clone());
        self.current = to;
        self.phase_entry_step = step;
        self.consecutive_satisfied = 0;
        transition
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::parse_spec;

    fn test_phase_ctrl() -> PhaseController {
        let spec = parse_spec(
            r#"
model:
  name: "test"
  components: [backbone, head]
invariants:
  hard:
    head.grad_norm: 0.001
  soft: {}
phases:
  bootstrap:
    max_duration_steps: 100
    transition_guard:
      all_hard_invariants_satisfied_for: 10
  representation_formation:
    transition_guard:
      all_hard_invariants_satisfied_for: 20
  stabilization:
    transition_guard:
      all_hard_invariants_satisfied_for: 30
  refinement:
    allowed_interventions:
      - adjust_lr
      - freeze
control: {}
"#,
        )
        .unwrap();
        PhaseController::new(&spec.phases)
    }

    #[test]
    fn starts_at_bootstrap() {
        let ctrl = test_phase_ctrl();
        assert_eq!(ctrl.current_phase(), Phase::Bootstrap);
    }

    #[test]
    fn advances_after_clean_steps() {
        let mut ctrl = test_phase_ctrl();
        let empty_violations = vec![];
        let empty_counts = HashMap::new();

        // Run 10 clean steps to satisfy bootstrap transition guard
        for step in 0..10 {
            ctrl.update(&empty_violations, &empty_counts, 3, step, true);
        }

        assert_eq!(ctrl.current_phase(), Phase::RepresentationFormation);
        assert_eq!(ctrl.history().len(), 1);
    }

    #[test]
    fn violations_reset_counter() {
        let mut ctrl = test_phase_ctrl();
        let empty_violations = vec![];
        let hard_violation = vec![Violation {
            invariant_name: "test".into(),
            component: "head".into(),
            severity: Severity::Hard,
            observed: 0.0,
            threshold: 0.001,
            direction: ThresholdDirection::Min,
            step: 0,
            passive: false,
        }];
        let empty_counts = HashMap::new();

        // 9 clean steps
        for step in 0..9 {
            ctrl.update(&empty_violations, &empty_counts, 3, step, true);
        }
        assert_eq!(ctrl.current_phase(), Phase::Bootstrap);

        // 1 violation resets counter
        ctrl.update(&hard_violation, &empty_counts, 3, 9, true);
        assert_eq!(ctrl.current_phase(), Phase::Bootstrap);

        // Need 10 more clean steps now
        for step in 10..19 {
            ctrl.update(&empty_violations, &empty_counts, 3, step, true);
        }
        assert_eq!(ctrl.current_phase(), Phase::Bootstrap);

        ctrl.update(&empty_violations, &empty_counts, 3, 19, true);
        assert_eq!(ctrl.current_phase(), Phase::RepresentationFormation);
    }

    #[test]
    fn regression_on_budget_exhaustion() {
        let mut ctrl = test_phase_ctrl();
        let empty_violations = vec![];

        // Advance to RepresentationFormation
        for step in 0..10 {
            ctrl.update(&empty_violations, &HashMap::new(), 3, step, true);
        }
        assert_eq!(ctrl.current_phase(), Phase::RepresentationFormation);

        // Exhaust budget for head component
        let mut counts = HashMap::new();
        counts.insert("head".to_string(), 3u32);

        let hard_violation = vec![Violation {
            invariant_name: "test".into(),
            component: "head".into(),
            severity: Severity::Hard,
            observed: 0.0,
            threshold: 0.001,
            direction: ThresholdDirection::Min,
            step: 10,
            passive: false,
        }];

        ctrl.update(&hard_violation, &counts, 3, 10, true);
        assert_eq!(ctrl.current_phase(), Phase::Bootstrap);
    }

    #[test]
    fn abort_on_double_regression() {
        let mut ctrl = test_phase_ctrl();
        let empty = vec![];

        // Advance to RepresentationFormation
        for step in 0..10 {
            ctrl.update(&empty, &HashMap::new(), 3, step, true);
        }

        // First regression
        let mut counts = HashMap::new();
        counts.insert("head".to_string(), 3u32);
        ctrl.update(&empty, &counts, 3, 10, true);
        assert_eq!(ctrl.current_phase(), Phase::Bootstrap);

        // Advance again
        for step in 11..21 {
            ctrl.update(&empty, &HashMap::new(), 3, step, true);
        }

        // Second regression attempt → abort
        ctrl.update(&empty, &counts, 3, 21, true);
        assert_eq!(ctrl.current_phase(), Phase::Aborted);
    }

    #[test]
    fn aborted_is_terminal() {
        let mut ctrl = test_phase_ctrl();
        ctrl.abort(0, "test".into());
        assert_eq!(ctrl.current_phase(), Phase::Aborted);

        let result = ctrl.update(&[], &HashMap::new(), 3, 1, true);
        assert!(result.is_none());
        assert_eq!(ctrl.current_phase(), Phase::Aborted);
    }

    #[test]
    fn allowed_interventions_by_phase() {
        let mut ctrl = test_phase_ctrl();
        // Bootstrap: no restrictions
        assert!(ctrl.allowed_interventions().is_none());

        // Advance through all phases to Refinement
        for step in 0..60 {
            ctrl.update(&[], &HashMap::new(), 3, step, true);
        }
        assert_eq!(ctrl.current_phase(), Phase::Refinement);

        let allowed = ctrl.allowed_interventions().unwrap();
        assert!(allowed.contains(&"adjust_lr".to_string()));
        assert!(allowed.contains(&"freeze".to_string()));
        assert!(!allowed.contains(&"reinitialize".to_string()));
    }

    #[test]
    fn phase_duration_expired() {
        let ctrl = test_phase_ctrl();
        // Bootstrap has max_duration_steps: 100
        assert!(!ctrl.is_phase_expired(50));
        assert!(ctrl.is_phase_expired(100));
    }
}
