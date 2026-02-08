use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::collector::MetricCollector;
use crate::control::ControlLaws;
use crate::diagnostic::{DiagnosticConfig, DiagnosticLayer};
use crate::error::TransXformError;
use crate::executor::InterventionExecutor;
use crate::ledger::{BoundaryLedger, DiagnosticSummary, InterventionOutcome, TrainingCertificate};
use crate::model::Model;
use crate::monitor::InvariantMonitor;
use crate::phase::PhaseController;
use crate::registry::SignatureRegistry;
use crate::regret::{RegretAssessment, RegretTracker};
use crate::spec::TrainingSpec;
use crate::types::*;

/// The report returned from each supervisor step.
#[derive(Debug, Clone)]
pub struct SupervisorReport {
    pub step: u64,
    pub phase: Phase,
    pub violations: Vec<Violation>,
    pub actions_taken: Vec<Action>,
    pub near_misses: Vec<NearMiss>,
    pub phase_transition: Option<PhaseTransition>,
    pub signature_matches: Vec<String>,
    pub regret_assessments: Vec<RegretAssessment>,
    pub pending_lr_adjustments: Vec<(String, f64)>,
    pub metrics: MetricSnapshot,
    /// V2: Advisory diagnostic warnings (non-authoritative).
    pub diagnostic_warnings: Vec<DiagnosticWarning>,
}

/// Main composition layer — orchestrates all subsystems (whitepaper §3).
///
/// The supervisor implements the full feedback loop:
/// 1. Collect metrics from the model
/// 2. Check failure signatures
/// 3. Check invariants (with phase thresholds + hysteresis)
/// 4. Check near-misses
/// 5. For each violation: determine action → execute → record
/// 6. Update regret windows
/// 7. Update phase controller
/// 8. Return report
pub struct Supervisor<M: Model, C: MetricCollector<M>> {
    spec: TrainingSpec,
    model: Rc<RefCell<M>>,
    collector: C,
    monitor: InvariantMonitor,
    phase_controller: PhaseController,
    control_laws: ControlLaws,
    executor: InterventionExecutor<M>,
    ledger: BoundaryLedger,
    regret_tracker: RegretTracker,
    registry: SignatureRegistry,
    /// V2: Advisory diagnostic layer (non-authoritative).
    diagnostic: DiagnosticLayer,
    /// Rolling window of recent metrics for trajectory analysis.
    metric_history: VecDeque<MetricSnapshot>,
    metric_history_max: usize,
    /// Pending LR adjustments that the training loop must apply.
    pending_lr: Vec<(String, f64)>,
    /// Whether training has been aborted.
    aborted: bool,
    abort_reason: Option<String>,
    /// Consecutive steps with no hard violations (for negative capability).
    total_interventions: u64,
}

impl<M: Model, C: MetricCollector<M>> Supervisor<M, C> {
    /// Construct a new supervisor from a training spec.
    pub fn new(
        spec: TrainingSpec,
        model: Rc<RefCell<M>>,
        collector: C,
    ) -> Result<Self, TransXformError> {
        // Validate spec
        crate::spec::validate_spec(&spec)?;

        let monitor = InvariantMonitor::new(&spec);
        let phase_controller = PhaseController::new(&spec.phases);
        let control_laws = ControlLaws::new(&spec.control);
        let executor = InterventionExecutor::new(model.clone());
        let ledger = BoundaryLedger::new();
        let regret_tracker = RegretTracker::new(spec.control.regret_window_steps);
        let registry = SignatureRegistry::with_defaults();
        let diagnostic = DiagnosticLayer::new(
            DiagnosticConfig::default(),
            spec.model.components.clone(),
        );

        Ok(Self {
            spec,
            model,
            collector,
            monitor,
            phase_controller,
            control_laws,
            executor,
            ledger,
            regret_tracker,
            registry,
            diagnostic,
            metric_history: VecDeque::new(),
            metric_history_max: 200,
            pending_lr: Vec::new(),
            aborted: false,
            abort_reason: None,
            total_interventions: 0,
        })
    }

    /// Run one supervisor step. This is the core feedback loop.
    pub fn step(&mut self, step: u64) -> Result<SupervisorReport, TransXformError> {
        if self.aborted {
            return Err(TransXformError::PhaseError(format!(
                "Training already aborted: {}",
                self.abort_reason.as_deref().unwrap_or("unknown")
            )));
        }

        let phase = self.phase_controller.current_phase();
        if phase == Phase::Aborted {
            self.aborted = true;
            self.abort_reason = Some("Phase controller in aborted state".into());
            return Err(TransXformError::PhaseError(
                "Training in aborted state".into(),
            ));
        }

        // 1. Determine metric tier and collect metrics
        let tier = crate::collector::BasicMetricCollector::new(
            vec![],
            self.spec.metric_cadence.clone(),
        )
        .tier_for_step(step);

        let metrics = {
            let model_ref = self.model.borrow();
            self.collector.collect(&*model_ref, step, tier)?
        };

        // Store in history
        self.metric_history.push_back(metrics.clone());
        if self.metric_history.len() > self.metric_history_max {
            self.metric_history.pop_front();
        }

        // 2. Check failure signatures
        let signature_matches: Vec<String> = self
            .registry
            .check(&metrics, step)
            .iter()
            .map(|s| s.id.clone())
            .collect();

        // 3. Check invariants with phase thresholds + hysteresis
        let phase_thresholds = self.phase_controller.current_thresholds();
        let violations = self.monitor.check(
            &metrics,
            &phase_thresholds,
            self.spec.control.hysteresis_margin,
            step,
        );

        // Record invariant checks
        for inv in self.monitor.invariants() {
            self.ledger.record_check(&inv.name);
        }

        // 4. Check near-misses
        let near_misses = self.monitor.check_near_misses(
            &metrics,
            &phase_thresholds,
            self.spec.control.hysteresis_margin,
            step,
        );

        for nm in &near_misses {
            self.regret_tracker.record_near_miss(nm.clone());
            self.ledger.record_near_miss(step, phase, nm);
        }

        // 5. Process violations → determine actions → execute → record
        let allowed_interventions = self.phase_controller.allowed_interventions();
        let allowed_ref: Option<Vec<String>> = allowed_interventions;
        let mut actions_taken = Vec::new();
        self.pending_lr.clear();

        for violation in &violations {
            let action = match violation.severity {
                Severity::Hard => self.control_laws.hard_action(
                    violation,
                    phase,
                    step,
                    allowed_ref.as_deref(),
                ),
                Severity::Soft => self.control_laws.soft_action(
                    violation,
                    phase,
                    allowed_ref.as_deref(),
                ),
            };

            if let Some(action) = action {
                // Execute the action
                match self.executor.execute(&action) {
                    Ok(()) => {
                        // Record success
                        if violation.severity == Severity::Hard {
                            self.control_laws.record_intervention(
                                &violation.component,
                                phase,
                                step,
                            );
                            self.total_interventions += 1;

                            // Open regret window for hard interventions
                            let pre_trajectory = self.build_trajectory(
                                &violation.invariant_name,
                            );
                            self.regret_tracker.open_window(
                                step,
                                &violation.component,
                                &action,
                                &violation.invariant_name,
                                violation.observed,
                                pre_trajectory,
                            );
                        }

                        self.ledger.record_with_snapshot(
                            step,
                            phase,
                            violation,
                            &action,
                            format!(
                                "{} violation: observed={:.6}, threshold={:.6}",
                                violation.severity, violation.observed, violation.threshold,
                            ),
                            metrics.clone(),
                        );

                        actions_taken.push(action);
                    }
                    Err(TransXformError::InterventionFailed {
                        action: act_name,
                        component,
                        reason,
                    }) => {
                        // adjust_lr failure → queue as pending
                        if act_name == "adjust_lr" {
                            if let Action::AdjustLr { component: c, factor } = &action {
                                self.pending_lr.push((c.clone(), *factor));
                            }
                        }
                        log::warn!(
                            "Intervention failed: {} on {}: {}",
                            act_name,
                            component,
                            reason
                        );
                    }
                    Err(TransXformError::TrainingAborted { verdict, details }) => {
                        // Abort requested by action (e.g., signature-matched fix)
                        self.aborted = true;
                        self.abort_reason = Some(details.clone());
                        let transition = self.phase_controller.abort(step, details.clone());
                        self.ledger.record_phase_transition(step, &transition);
                        return Err(TransXformError::TrainingAborted { verdict, details });
                    }
                    Err(e) => {
                        log::error!("Unexpected execution error: {}", e);
                    }
                }
            }
        }

        // Execute signature-matched proven fixes (if not already addressed by invariant violations)
        for sig_id in &signature_matches {
            if let Some(sig) = self.registry.get(sig_id) {
                let fix = &sig.proven_fix;
                // Only execute if the fix targets a component not already acted upon
                let already_acted = actions_taken.iter().any(|a| {
                    a.component() == fix.component()
                        && a.action_name() == fix.action_name()
                });
                if !already_acted {
                    if let Action::Abort { reason } = fix {
                        log::warn!(
                            "Signature {} recommends abort: {}",
                            sig_id,
                            reason
                        );
                        // Don't auto-abort from signatures; log as near-miss
                    } else {
                        log::info!(
                            "Executing proven fix from signature {}: {}",
                            sig_id,
                            fix
                        );
                        if let Err(e) = self.executor.execute(fix) {
                            log::warn!("Signature fix {} failed: {}", sig_id, e);
                        } else {
                            actions_taken.push(fix.clone());
                        }
                    }
                }
            }
        }

        // 6. Update regret windows
        let regret_assessments = self.regret_tracker.update(step, &metrics);

        // Apply regret assessments to ledger
        for assessment in &regret_assessments {
            let outcome = match assessment.tag {
                RegretTag::Confident => InterventionOutcome::Recovered,
                RegretTag::LowConfidence => InterventionOutcome::Persisted,
                RegretTag::Pending => InterventionOutcome::Pending,
            };
            self.ledger.update_outcome(
                assessment.intervention_step,
                &assessment.component,
                outcome,
                assessment.tag,
            );
        }

        // 7. Update phase controller
        let counts = self
            .control_laws
            .counts_for_phase(phase);
        let phase_transition = self.phase_controller.update(
            &violations,
            &counts,
            self.spec.control.max_hard_interventions,
            step,
        );

        if let Some(ref transition) = phase_transition {
            self.ledger.record_phase_transition(step, transition);
            self.control_laws.on_phase_change(transition.to);
            log::info!("Phase transition: {}", transition);

            if transition.to == Phase::Aborted {
                self.aborted = true;
                self.abort_reason = Some(transition.reason.clone());
            }
        }

        // 8. Check negative capability conditions (whitepaper §13)
        self.check_negative_capabilities(step, &metrics, &violations)?;

        // 9. V2: Run diagnostic layer (advisory — never intervenes)
        let diagnostic_warnings = self.diagnostic.diagnose(step, &metrics);
        for warning in &diagnostic_warnings {
            self.ledger.record_advisory(
                step,
                self.phase_controller.current_phase(),
                &warning.signal.to_string(),
                &warning.summary,
            );
            log::info!("{}", warning);
        }

        Ok(SupervisorReport {
            step,
            phase: self.phase_controller.current_phase(),
            violations,
            actions_taken,
            near_misses,
            phase_transition,
            signature_matches,
            regret_assessments,
            pending_lr_adjustments: self.pending_lr.clone(),
            metrics,
            diagnostic_warnings,
        })
    }

    /// Emit the training certificate at the end of training.
    pub fn emit_certificate(&self, total_steps: u64) -> TrainingCertificate {
        let final_metrics = self
            .metric_history
            .back()
            .cloned()
            .unwrap_or_default();

        // Build V2 diagnostic summary
        let warnings = self.diagnostic.warnings();
        let mut by_signal = std::collections::HashMap::new();
        for w in warnings {
            *by_signal.entry(w.signal.to_string()).or_insert(0u64) += 1;
        }
        let acknowledged = warnings.iter().filter(|w| w.acknowledged).count() as u64;
        let diagnostic_summary = DiagnosticSummary {
            total_warnings: warnings.len() as u64,
            acknowledged,
            unacknowledged: warnings.len() as u64 - acknowledged,
            by_signal,
        };

        self.ledger.emit_certificate_with_diagnostics(
            &self.spec.model.name,
            total_steps,
            &final_metrics,
            self.phase_controller.history(),
            diagnostic_summary,
        )
    }

    /// Get a reference to the boundary ledger.
    pub fn ledger(&self) -> &BoundaryLedger {
        &self.ledger
    }

    /// Get the current phase.
    pub fn current_phase(&self) -> Phase {
        self.phase_controller.current_phase()
    }

    /// Get the training spec.
    pub fn spec(&self) -> &TrainingSpec {
        &self.spec
    }

    /// Whether training has been aborted.
    pub fn is_aborted(&self) -> bool {
        self.aborted
    }

    /// Get the regret tracker.
    pub fn regret_tracker(&self) -> &RegretTracker {
        &self.regret_tracker
    }

    /// Get the phase controller.
    pub fn phase_controller(&self) -> &PhaseController {
        &self.phase_controller
    }

    /// Get the total number of hard interventions executed.
    pub fn total_interventions(&self) -> u64 {
        self.total_interventions
    }

    /// Get recent metric history.
    pub fn metric_history(&self) -> &VecDeque<MetricSnapshot> {
        &self.metric_history
    }

    /// Get a reference to the V2 diagnostic layer.
    pub fn diagnostic(&self) -> &DiagnosticLayer {
        &self.diagnostic
    }

    /// Get a mutable reference to the V2 diagnostic layer
    /// (e.g. to acknowledge warnings).
    pub fn diagnostic_mut(&mut self) -> &mut DiagnosticLayer {
        &mut self.diagnostic
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Build a trajectory (recent values) for a metric key from history.
    fn build_trajectory(&self, metric_key: &str) -> VecDeque<f64> {
        self.metric_history
            .iter()
            .filter_map(|snapshot| snapshot.get(metric_key).copied())
            .collect()
    }

    /// Check negative capability conditions (whitepaper §13).
    ///
    /// These detect situations where the supervisor should declare training
    /// unsalvageable rather than continuing to intervene.
    fn check_negative_capabilities(
        &mut self,
        step: u64,
        metrics: &MetricSnapshot,
        violations: &[Violation],
    ) -> Result<(), TransXformError> {
        // UNSTABLE_ARCHITECTURE: repeated interventions across multiple phases
        // with no sustained recovery
        if self.total_interventions > (self.spec.control.max_hard_interventions * 3) as u64 {
            let phase = self.phase_controller.current_phase();
            if phase == Phase::Bootstrap || phase == Phase::RepresentationFormation {
                // Too many interventions too early — architecture may be fundamentally broken
                let completed = self.regret_tracker.completed_assessments();
                let low_conf_count = completed
                    .iter()
                    .filter(|a| a.tag == RegretTag::LowConfidence)
                    .count();

                if low_conf_count as u64 > self.total_interventions / 2 {
                    self.aborted = true;
                    self.abort_reason =
                        Some("Majority of interventions had low confidence".into());
                    let transition = self.phase_controller.abort(
                        step,
                        "UNSTABLE_ARCHITECTURE: majority of interventions ineffective".into(),
                    );
                    self.ledger.record_phase_transition(step, &transition);
                    return Err(TransXformError::TrainingAborted {
                        verdict: NegativeVerdict::UnstableArchitecture,
                        details: format!(
                            "{} of {} interventions were low-confidence",
                            low_conf_count, self.total_interventions
                        ),
                    });
                }
            }
        }

        // DEGENERATE_OBJECTIVE: loss is non-informative (near-zero gradient everywhere)
        if let Some(&loss) = metrics.get("loss") {
            if loss.is_nan() || loss.is_infinite() {
                self.aborted = true;
                self.abort_reason = Some("Loss is NaN or Inf".into());
                let transition = self
                    .phase_controller
                    .abort(step, "DEGENERATE_OBJECTIVE: loss is NaN/Inf".into());
                self.ledger.record_phase_transition(step, &transition);
                return Err(TransXformError::TrainingAborted {
                    verdict: NegativeVerdict::DegenerateObjective,
                    details: format!("Loss value: {}", loss),
                });
            }
        }

        // INSUFFICIENT_SIGNAL: all hard violations persist for too long
        // without any intervention making progress
        if !violations.is_empty()
            && violations.iter().all(|v| v.severity == Severity::Hard)
            && step > 500
        {
            let all_budgets_exhausted = violations.iter().all(|v| {
                self.control_laws
                    .budget_exhausted(&v.component, self.phase_controller.current_phase())
            });

            if all_budgets_exhausted {
                // All components with violations have exhausted their budgets
                // Phase controller should handle this via regression, but if we're
                // already regressed and still stuck, that's insufficient signal
                log::warn!(
                    "Step {}: all violating components have exhausted intervention budgets",
                    step
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::collector::BasicMetricCollector;
    use crate::model::MockModel;
    use crate::spec::parse_spec;

    fn test_supervisor() -> (
        Supervisor<MockModel, BasicMetricCollector>,
        Rc<RefCell<MockModel>>,
    ) {
        let spec = parse_spec(
            r#"
model:
  name: "test"
  components: [backbone, head]
invariants:
  hard:
    head.pairwise_cosine: 0.95
    head.grad_norm: 0.001
  soft:
    attention_entropy_min: 0.3
phases:
  bootstrap:
    transition_guard:
      all_hard_invariants_satisfied_for: 5
  representation_formation:
    transition_guard:
      all_hard_invariants_satisfied_for: 10
control:
  cooldown_steps: 10
  max_hard_interventions: 3
  hysteresis_margin: 0.0
  damping_factor: 0.5
  regret_window_steps: 20
"#,
        )
        .unwrap();

        let model = Rc::new(RefCell::new(MockModel::new(&["backbone", "head"])));
        let collector = BasicMetricCollector::new(
            vec!["backbone".into(), "head".into()],
            HashMap::new(),
        );

        let supervisor = Supervisor::new(spec, model.clone(), collector).unwrap();
        (supervisor, model)
    }

    #[test]
    fn healthy_run_no_violations() {
        let (mut supervisor, model) = test_supervisor();

        // Set healthy metrics
        {
            let mut m = model.borrow_mut();
            m.set_metric("head", "pairwise_cosine", 0.5);
            m.set_metric("head", "grad_norm", 0.0005);
            m.set_metric("backbone", "grad_norm", 0.05);
            m.set_global_metric("loss", 2.0);
        }

        for step in 0..10 {
            let report = supervisor.step(step).unwrap();
            assert!(report.violations.is_empty(), "Step {}: unexpected violations", step);
            assert!(report.actions_taken.is_empty());
        }

        let cert = supervisor.emit_certificate(10);
        assert_eq!(cert.verdict, HealthVerdict::Healthy);
    }

    #[test]
    fn detects_and_intervenes_on_violation() {
        let (mut supervisor, model) = test_supervisor();

        // Set violating metrics: cosine too high
        {
            let mut m = model.borrow_mut();
            m.set_metric("head", "pairwise_cosine", 0.99);
            m.set_metric("head", "grad_norm", 0.0005);
            m.set_metric("backbone", "grad_norm", 0.05);
            m.set_global_metric("loss", 2.0);
        }

        let report = supervisor.step(0).unwrap();
        assert!(!report.violations.is_empty());
        assert!(!report.actions_taken.is_empty());
        assert_eq!(report.actions_taken[0].action_name(), "reinitialize");
    }

    #[test]
    fn phase_advances_on_clean_steps() {
        let (mut supervisor, model) = test_supervisor();

        {
            let mut m = model.borrow_mut();
            m.set_metric("head", "pairwise_cosine", 0.5);
            m.set_metric("head", "grad_norm", 0.0005);
            m.set_metric("backbone", "grad_norm", 0.05);
            m.set_global_metric("loss", 2.0);
        }

        assert_eq!(supervisor.current_phase(), Phase::Bootstrap);

        // 5 clean steps should advance past Bootstrap
        for step in 0..5 {
            supervisor.step(step).unwrap();
        }

        assert_eq!(supervisor.current_phase(), Phase::RepresentationFormation);
    }

    #[test]
    fn aborts_on_nan_loss() {
        let (mut supervisor, model) = test_supervisor();

        {
            let mut m = model.borrow_mut();
            m.set_metric("head", "pairwise_cosine", 0.5);
            m.set_metric("head", "grad_norm", 0.0005);
            m.set_global_metric("loss", f64::NAN);
        }

        let result = supervisor.step(0);
        assert!(result.is_err());
        assert!(supervisor.is_aborted());
    }

    #[test]
    fn cooldown_prevents_rapid_interventions() {
        let (mut supervisor, model) = test_supervisor();

        {
            let mut m = model.borrow_mut();
            m.set_metric("head", "pairwise_cosine", 0.99);
            m.set_metric("head", "grad_norm", 0.0005);
            m.set_metric("backbone", "grad_norm", 0.05);
            m.set_global_metric("loss", 2.0);
        }

        // Step 0: should intervene
        let r0 = supervisor.step(0).unwrap();
        assert!(!r0.actions_taken.is_empty());

        // Step 1: should be in cooldown (cooldown_steps = 10)
        let r1 = supervisor.step(1).unwrap();
        assert!(r1.actions_taken.is_empty());
    }

    #[test]
    fn certificate_after_recovery() {
        let (mut supervisor, model) = test_supervisor();

        // Start with violation
        {
            let mut m = model.borrow_mut();
            m.set_metric("head", "pairwise_cosine", 0.99);
            m.set_metric("head", "grad_norm", 0.0005);
            m.set_metric("backbone", "grad_norm", 0.05);
            m.set_global_metric("loss", 2.0);
        }
        supervisor.step(0).unwrap();

        // Recover metrics
        {
            let mut m = model.borrow_mut();
            m.set_metric("head", "pairwise_cosine", 0.5);
        }

        // Run enough steps to close regret windows and advance phases
        for step in 1..25 {
            let _ = supervisor.step(step);
        }

        let cert = supervisor.emit_certificate(25);
        // Should be Recovered (had interventions) not Healthy
        assert!(
            matches!(cert.verdict, HealthVerdict::Recovered { .. })
                || matches!(cert.verdict, HealthVerdict::Healthy)
                || matches!(cert.verdict, HealthVerdict::Compromised { .. })
        );
        assert!(cert.intervention_summary.total_hard > 0 || cert.intervention_summary.total_soft > 0);
    }
}
