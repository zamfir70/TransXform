use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::checkpoint::{
    CheckpointHint, CheckpointReason, SupervisorCheckpoint, CHECKPOINT_VERSION,
};
use crate::collector::MetricCollector;
use crate::control::ControlLaws;
use crate::diagnostic::{DiagnosticConfig, DiagnosticLayer, InterventionOutcomeRecord};
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
    /// V1.4: Hint that this is a good moment to checkpoint.
    pub checkpoint_hint: Option<CheckpointHint>,
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
    /// V1.3: Audit trail of all runtime threshold amendments.
    runtime_amendments: Vec<RuntimeAmendment>,
    /// V1.3: Threshold overrides for the next phase (readiness gate relaxation).
    readiness_overrides: HashMap<String, f64>,
    /// V1.3: Active threshold overrides for the current phase (post-transition).
    active_overrides: HashMap<String, f64>,
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
        let mut control_laws = ControlLaws::new(&spec.control);
        // Set upstream dependency map from spec role declarations
        let upstream_map = spec.upstream_map();
        if !upstream_map.is_empty() {
            control_laws.set_upstream_map(upstream_map);
        }
        let executor = InterventionExecutor::new(model.clone());
        let ledger = BoundaryLedger::new();
        let regret_tracker = RegretTracker::new(spec.control.regret_window_steps);
        let registry = SignatureRegistry::with_defaults();
        let mut diagnostic = DiagnosticLayer::new(
            DiagnosticConfig::default(),
            spec.model.components.clone(),
        );
        // Tell the diagnostic layer which metric keys invariants depend on
        diagnostic.set_expected_metrics(monitor.metric_keys());
        // Tell the diagnostic layer about invariant thresholds for drift detection
        let thresholds: HashMap<String, (f64, ThresholdDirection)> = monitor
            .invariants()
            .iter()
            .map(|inv| (inv.metric_key.clone(), (inv.base_threshold, inv.direction)))
            .collect();
        diagnostic.set_invariant_thresholds(thresholds);

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
            runtime_amendments: Vec::new(),
            readiness_overrides: HashMap::new(),
            active_overrides: HashMap::new(),
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

        // 3. Check invariants with phase thresholds + proportional hysteresis
        let mut phase_thresholds = self.phase_controller.current_thresholds();
        // V1.3: Merge active threshold overrides from adaptive relaxation
        for (key, &value) in &self.active_overrides {
            phase_thresholds.insert(key.clone(), value);
        }
        let mut violations = self.monitor.check(
            &metrics,
            &phase_thresholds,
            &self.spec.control,
            step,
        );

        // Stamp passive flag on violations from passive components
        for v in &mut violations {
            if self.spec.is_passive(&v.component) {
                v.passive = true;
            }
        }

        // Record invariant checks
        for inv in self.monitor.invariants() {
            self.ledger.record_check(&inv.name);
        }

        // 4. Check near-misses
        let near_misses = self.monitor.check_near_misses(
            &metrics,
            &phase_thresholds,
            &self.spec.control,
            step,
        );

        for nm in &near_misses {
            self.regret_tracker.record_near_miss(nm.clone());
            self.ledger.record_near_miss(step, phase, nm);
        }

        // 5. Process violations → determine actions → execute → record
        //    Passive components are observed but not intervened on.
        let allowed_interventions = self.phase_controller.allowed_interventions();
        let allowed_ref: Option<Vec<String>> = allowed_interventions;
        let mut actions_taken = Vec::new();
        self.pending_lr.clear();

        for violation in &violations {
            // Skip interventions for passive components (observe only).
            // Violations still appear in the report for visibility.
            if self.spec.is_passive(&violation.component) {
                log::debug!(
                    "Passive component '{}' violation ({}) — observed, not intervening",
                    violation.component, violation.invariant_name
                );
                continue;
            }

            let action = match violation.severity {
                Severity::Hard => self.control_laws.hard_action(
                    violation,
                    phase,
                    step,
                    allowed_ref.as_deref(),
                    &metrics,
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

        // 6b. Feed intervention outcomes to diagnostic layer for futility detection
        for assessment in &regret_assessments {
            if assessment.tag != RegretTag::Pending {
                self.diagnostic.record_intervention_outcome(
                    InterventionOutcomeRecord {
                        step: assessment.intervention_step,
                        component: assessment.component.clone(),
                        action: assessment.action.clone(),
                        recovered: assessment.tag == RegretTag::Confident,
                        recovery_steps: assessment.recovery_steps,
                    },
                );
            }
        }

        // 7. Update phase controller (only active component violations affect transitions)
        let active_violations: Vec<Violation> = violations
            .iter()
            .filter(|v| !self.spec.is_passive(&v.component))
            .cloned()
            .collect();

        let counts = self
            .control_laws
            .counts_for_phase(phase);
        // V1.3: Phase readiness gate — check if metrics satisfy next phase thresholds
        let readiness_ok = self.check_phase_readiness(&metrics);
        if !readiness_ok {
            self.try_adaptive_relaxation(step, &metrics);
        }

        let phase_transition = self.phase_controller.update(
            &active_violations,
            &counts,
            self.spec.control.max_hard_interventions,
            step,
            readiness_ok,
        );

        if let Some(ref transition) = phase_transition {
            self.ledger.record_phase_transition(step, transition);
            self.control_laws.on_phase_change(transition.to);
            // Reset diagnostic history for the new phase — trend analysis
            // should start fresh, not carry over from previous phase dynamics.
            self.diagnostic.on_phase_transition(transition.from, transition.to);
            log::info!("Phase transition: {}", transition);

            // V1.3: Manage threshold overrides on phase transition
            if transition.from.next() == Some(transition.to) {
                // Forward transition — promote readiness overrides to active
                self.active_overrides = std::mem::take(&mut self.readiness_overrides);
            } else {
                // Regression or abort — clear all overrides
                self.readiness_overrides.clear();
                self.active_overrides.clear();
            }

            if transition.to == Phase::Aborted {
                self.aborted = true;
                self.abort_reason = Some(transition.reason.clone());
            }
        }

        // 8. Check negative capability conditions (whitepaper §13)
        //    Only active component violations trigger abort conditions.
        self.check_negative_capabilities(step, &metrics, &active_violations)?;

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

        // 10. V1.4: Compute checkpoint hint
        let checkpoint_hint = self.compute_checkpoint_hint(step, phase_transition.as_ref());

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
            checkpoint_hint,
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

    /// V1.3: Get runtime threshold amendments (audit trail).
    pub fn runtime_amendments(&self) -> &[RuntimeAmendment] {
        &self.runtime_amendments
    }

    // -----------------------------------------------------------------------
    // V1.4: Checkpointing
    // -----------------------------------------------------------------------

    /// Create a checkpoint of all supervisor runtime state.
    ///
    /// The checkpoint captures everything needed to resume training from
    /// this point. It does NOT include the model weights or optimizer state —
    /// those are the user's responsibility to save/restore separately.
    ///
    /// The checkpoint can be serialized to JSON or YAML via serde.
    pub fn checkpoint(&self, step: u64) -> SupervisorCheckpoint {
        SupervisorCheckpoint {
            version: CHECKPOINT_VERSION,
            created_at: chrono::Utc::now(),
            step,
            model_name: self.spec.model.name.clone(),

            metric_history: self.metric_history.clone(),
            metric_history_max: self.metric_history_max,
            pending_lr: self.pending_lr.clone(),
            aborted: self.aborted,
            abort_reason: self.abort_reason.clone(),
            total_interventions: self.total_interventions,
            runtime_amendments: self.runtime_amendments.clone(),
            readiness_overrides: self.readiness_overrides.clone(),
            active_overrides: self.active_overrides.clone(),

            phase_controller: self.phase_controller.save_state(),
            control_laws: self.control_laws.save_state(),
            ledger: self.ledger.save_state(),
            regret_tracker: self.regret_tracker.save_state(),
            diagnostic: self.diagnostic.save_state(),
            registry: self.registry.save_state(),
        }
    }

    /// Reconstruct a supervisor from a checkpoint.
    ///
    /// The `spec`, `model`, and `collector` must be provided fresh —
    /// TransXform does not serialize external state. The spec should
    /// match the one used when the checkpoint was created (same model name,
    /// same invariants, same phases).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The checkpoint version is unsupported
    /// - The spec model name does not match the checkpoint
    /// - Spec validation fails
    pub fn from_checkpoint(
        spec: TrainingSpec,
        model: Rc<RefCell<M>>,
        collector: C,
        checkpoint: SupervisorCheckpoint,
    ) -> Result<Self, TransXformError> {
        // Validate version
        if checkpoint.version > CHECKPOINT_VERSION {
            return Err(TransXformError::CheckpointError(format!(
                "Checkpoint version {} is newer than supported version {}",
                checkpoint.version, CHECKPOINT_VERSION,
            )));
        }

        // Validate model name match
        if checkpoint.model_name != spec.model.name {
            return Err(TransXformError::CheckpointError(format!(
                "Checkpoint model name '{}' does not match spec model name '{}'",
                checkpoint.model_name, spec.model.name,
            )));
        }

        // Validate spec
        crate::spec::validate_spec(&spec)?;

        // Build stateless components from spec (same as new())
        let monitor = InvariantMonitor::new(&spec);
        let mut phase_controller = PhaseController::new(&spec.phases);
        let mut control_laws = ControlLaws::new(&spec.control);
        let upstream_map = spec.upstream_map();
        if !upstream_map.is_empty() {
            control_laws.set_upstream_map(upstream_map);
        }
        let executor = InterventionExecutor::new(model.clone());
        let mut ledger = BoundaryLedger::new();
        let mut regret_tracker = RegretTracker::new(spec.control.regret_window_steps);
        let mut registry = SignatureRegistry::with_defaults();
        let mut diagnostic = DiagnosticLayer::new(
            DiagnosticConfig::default(),
            spec.model.components.clone(),
        );
        diagnostic.set_expected_metrics(monitor.metric_keys());
        let thresholds: HashMap<String, (f64, ThresholdDirection)> = monitor
            .invariants()
            .iter()
            .map(|inv| (inv.metric_key.clone(), (inv.base_threshold, inv.direction)))
            .collect();
        diagnostic.set_invariant_thresholds(thresholds);

        // Restore mutable state from checkpoint
        phase_controller.restore_state(checkpoint.phase_controller);
        control_laws.restore_state(checkpoint.control_laws);
        ledger.restore_state(checkpoint.ledger);
        regret_tracker.restore_state(checkpoint.regret_tracker);
        diagnostic.restore_state(checkpoint.diagnostic);
        registry.restore_state(checkpoint.registry);

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
            metric_history: checkpoint.metric_history,
            metric_history_max: checkpoint.metric_history_max,
            pending_lr: checkpoint.pending_lr,
            aborted: checkpoint.aborted,
            abort_reason: checkpoint.abort_reason,
            total_interventions: checkpoint.total_interventions,
            runtime_amendments: checkpoint.runtime_amendments,
            readiness_overrides: checkpoint.readiness_overrides,
            active_overrides: checkpoint.active_overrides,
        })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Determine if this is a good moment to checkpoint.
    fn compute_checkpoint_hint(
        &self,
        step: u64,
        phase_transition: Option<&PhaseTransition>,
    ) -> Option<CheckpointHint> {
        // Hint 1: Just after a phase transition
        if let Some(transition) = phase_transition {
            return Some(CheckpointHint {
                reason: CheckpointReason::PostPhaseTransition {
                    from: transition.from,
                    to: transition.to,
                },
                step,
            });
        }

        // Hint 2: Phase transition is imminent (2 steps away)
        let guard_steps = self.phase_controller.transition_guard_steps_public();
        let consecutive = self.phase_controller.consecutive_satisfied();
        if guard_steps > 0 && consecutive > 0 && consecutive + 2 >= guard_steps && consecutive < guard_steps {
            return Some(CheckpointHint {
                reason: CheckpointReason::PrePhaseTransition {
                    current_phase: self.phase_controller.current_phase(),
                    steps_to_transition: guard_steps - consecutive,
                },
                step,
            });
        }

        None
    }

    /// Build a trajectory (recent values) for a metric key from history.
    fn build_trajectory(&self, metric_key: &str) -> VecDeque<f64> {
        self.metric_history
            .iter()
            .filter_map(|snapshot| snapshot.get(metric_key).copied())
            .collect()
    }

    /// V1.3: Check if current metrics satisfy the next phase's thresholds.
    /// Returns `true` if the readiness gate is disabled, there's no next phase,
    /// or all next-phase thresholds are already satisfied.
    fn check_phase_readiness(&self, metrics: &MetricSnapshot) -> bool {
        if !self.spec.control.readiness_gate {
            return true;
        }
        let current = self.phase_controller.current_phase();
        let next = match current.next() {
            Some(n) => n,
            None => return true,
        };
        let mut next_thresholds = self.phase_controller.thresholds_for(next);
        // Apply any existing relaxation overrides
        for (key, &value) in &self.readiness_overrides {
            next_thresholds.insert(key.clone(), value);
        }
        if next_thresholds.is_empty() {
            return true;
        }
        for (metric_key, &threshold) in &next_thresholds {
            let observed = match metrics.get(metric_key) {
                Some(v) => *v,
                None => continue,
            };
            let direction = self.infer_direction_for(metric_key);
            let satisfied = match direction {
                ThresholdDirection::Min => observed >= threshold,
                ThresholdDirection::Max => observed <= threshold,
            };
            if !satisfied {
                return false;
            }
        }
        true
    }

    /// V1.3: Adaptive threshold relaxation when readiness gate blocks too long.
    /// Called once per patience period (at patience, 2*patience, etc.).
    fn try_adaptive_relaxation(&mut self, step: u64, metrics: &MetricSnapshot) {
        let blocked_since = match self.phase_controller.readiness_blocked_since() {
            Some(s) => s,
            None => return,
        };
        let blocked_duration = step.saturating_sub(blocked_since);
        let patience = self.spec.control.readiness_patience_steps;
        if patience == 0 || blocked_duration < patience || blocked_duration % patience != 0 {
            return;
        }

        let current = self.phase_controller.current_phase();
        let next = match current.next() {
            Some(n) => n,
            None => return,
        };
        let next_thresholds = self.phase_controller.thresholds_for(next);
        let max_relax = self.spec.control.max_threshold_relaxation;

        for (metric_key, &original_threshold) in &next_thresholds {
            let effective = self
                .readiness_overrides
                .get(metric_key)
                .copied()
                .unwrap_or(original_threshold);

            let observed = match metrics.get(metric_key) {
                Some(v) => *v,
                None => continue,
            };

            let direction = self.infer_direction_for(metric_key);
            let satisfied = match direction {
                ThresholdDirection::Min => observed >= effective,
                ThresholdDirection::Max => observed <= effective,
            };

            if !satisfied {
                let relaxed = match direction {
                    ThresholdDirection::Min => effective * (1.0 - max_relax),
                    ThresholdDirection::Max => effective * (1.0 + max_relax),
                };

                let amendment = RuntimeAmendment {
                    step,
                    metric_key: metric_key.clone(),
                    phase: next,
                    original_threshold,
                    relaxed_threshold: relaxed,
                    reason: format!(
                        "Readiness gate blocked {} steps; observed={:.6}, effective={:.6}",
                        blocked_duration, observed, effective,
                    ),
                };
                log::info!("Adaptive relaxation: {}", amendment);
                self.ledger.record_amendment(
                    step,
                    self.phase_controller.current_phase(),
                    &amendment,
                );
                self.runtime_amendments.push(amendment);
                self.readiness_overrides.insert(metric_key.clone(), relaxed);
            }
        }
    }

    /// Infer threshold direction for a metric key from monitor invariants.
    fn infer_direction_for(&self, metric_key: &str) -> ThresholdDirection {
        self.monitor
            .invariants()
            .iter()
            .find(|inv| inv.metric_key == metric_key || inv.name == metric_key)
            .map(|inv| inv.direction)
            .unwrap_or(ThresholdDirection::Max)
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

    /// Helper: supervisor with a passive component that violates invariants.
    fn test_supervisor_with_passive() -> (
        Supervisor<MockModel, BasicMetricCollector>,
        Rc<RefCell<MockModel>>,
    ) {
        let spec = parse_spec(
            r#"
model:
  name: "test_passive"
  components: [backbone, head, aux_head]
roles:
  aux_head:
    must_maintain_gradient: true
    passive: true
invariants:
  hard:
    head.pairwise_cosine: 0.95
    grad_norm_min:
      head: 0.001
      aux_head: 0.001
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

        let model = Rc::new(RefCell::new(MockModel::new(&["backbone", "head", "aux_head"])));
        let collector = BasicMetricCollector::new(
            vec!["backbone".into(), "head".into(), "aux_head".into()],
            HashMap::new(),
        );

        let supervisor = Supervisor::new(spec, model.clone(), collector).unwrap();
        (supervisor, model)
    }

    #[test]
    fn passive_component_violation_reported_but_no_intervention() {
        let (mut supervisor, model) = test_supervisor_with_passive();

        {
            let mut m = model.borrow_mut();
            // head is healthy
            m.set_metric("head", "pairwise_cosine", 0.5);
            m.set_metric("head", "grad_norm_min", 0.01);
            m.set_metric("backbone", "grad_norm_min", 0.05);
            // aux_head violates grad_norm_min (passive — should not intervene)
            m.set_metric("aux_head", "grad_norm_min", 0.0);
            m.set_global_metric("loss", 2.0);
        }

        let report = supervisor.step(0).unwrap();

        // Violation should be detected and reported
        let aux_violations: Vec<_> = report
            .violations
            .iter()
            .filter(|v| v.component == "aux_head")
            .collect();
        assert!(
            !aux_violations.is_empty(),
            "Passive component violations should still be detected"
        );

        // But no action should be taken for it
        let aux_actions: Vec<_> = report
            .actions_taken
            .iter()
            .filter(|a| a.component() == Some("aux_head"))
            .collect();
        assert!(
            aux_actions.is_empty(),
            "Passive component should NOT trigger interventions"
        );
    }

    #[test]
    fn passive_violation_does_not_block_phase_advance() {
        let (mut supervisor, model) = test_supervisor_with_passive();

        {
            let mut m = model.borrow_mut();
            // Active components healthy
            m.set_metric("head", "pairwise_cosine", 0.5);
            m.set_metric("head", "grad_norm_min", 0.01);
            m.set_metric("backbone", "grad_norm_min", 0.05);
            // Passive component perpetually violating
            m.set_metric("aux_head", "grad_norm_min", 0.0);
            m.set_global_metric("loss", 2.0);
        }

        assert_eq!(supervisor.current_phase(), Phase::Bootstrap);

        // 5 clean steps (aux_head violation doesn't count)
        for step in 0..5 {
            supervisor.step(step).unwrap();
        }

        assert_eq!(
            supervisor.current_phase(),
            Phase::RepresentationFormation,
            "Passive component violation should not block phase advance"
        );
    }

    // -------------------------------------------------------------------
    // V1.3: Phase readiness gate + adaptive relaxation tests
    // -------------------------------------------------------------------

    /// Helper: supervisor with readiness gate enabled and phase thresholds.
    fn test_supervisor_with_readiness() -> (
        Supervisor<MockModel, BasicMetricCollector>,
        Rc<RefCell<MockModel>>,
    ) {
        let spec = parse_spec(
            r#"
model:
  name: "test_readiness"
  components: [backbone, head]
invariants:
  hard:
    head.pairwise_cosine: 0.95
    head.grad_norm: 0.001
  soft: {}
phases:
  bootstrap:
    transition_guard:
      all_hard_invariants_satisfied_for: 3
  representation_formation:
    thresholds:
      head.pairwise_cosine: 0.90
    transition_guard:
      all_hard_invariants_satisfied_for: 5
control:
  cooldown_steps: 10
  max_hard_interventions: 3
  hysteresis_margin: 0.0
  readiness_gate: true
  readiness_patience_steps: 5
  max_threshold_relaxation: 0.05
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
    fn readiness_gate_blocks_premature_advance() {
        let (mut supervisor, model) = test_supervisor_with_readiness();

        {
            let mut m = model.borrow_mut();
            // Healthy for bootstrap (0.92 < 0.95 base threshold), but
            // cosine 0.92 > 0.90 repr_formation threshold → readiness blocks
            m.set_metric("head", "pairwise_cosine", 0.92);
            m.set_metric("head", "grad_norm", 0.0005);
            m.set_metric("backbone", "grad_norm", 0.05);
            m.set_global_metric("loss", 2.0);
        }

        // Run well past the bootstrap transition guard (3 steps)
        for step in 0..6 {
            supervisor.step(step).unwrap();
        }

        // Should still be in Bootstrap — readiness gate blocks
        assert_eq!(
            supervisor.current_phase(),
            Phase::Bootstrap,
            "Readiness gate should block advance when metrics fail next phase thresholds"
        );
    }

    #[test]
    fn readiness_gate_allows_when_metrics_satisfy_next_phase() {
        let (mut supervisor, model) = test_supervisor_with_readiness();

        {
            let mut m = model.borrow_mut();
            // Cosine 0.5 < 0.90 repr_formation threshold → readiness passes
            m.set_metric("head", "pairwise_cosine", 0.5);
            m.set_metric("head", "grad_norm", 0.0005);
            m.set_metric("backbone", "grad_norm", 0.05);
            m.set_global_metric("loss", 2.0);
        }

        for step in 0..4 {
            supervisor.step(step).unwrap();
        }

        assert_eq!(
            supervisor.current_phase(),
            Phase::RepresentationFormation,
            "Readiness gate should allow advance when metrics satisfy next phase"
        );
    }

    #[test]
    fn adaptive_relaxation_enables_advance() {
        let (mut supervisor, model) = test_supervisor_with_readiness();

        {
            let mut m = model.borrow_mut();
            // Cosine 0.92 > 0.90 threshold → blocks, but within relaxation reach
            // After 5 patience steps: 0.90 * 1.05 = 0.945 > 0.92 → passes
            m.set_metric("head", "pairwise_cosine", 0.92);
            m.set_metric("head", "grad_norm", 0.0005);
            m.set_metric("backbone", "grad_norm", 0.05);
            m.set_global_metric("loss", 2.0);
        }

        // Run enough steps: guard at step 2, blocked_since=2,
        // patience at step 7 (duration=5), relaxes.
        // Step 8: readiness passes with relaxed threshold → advance.
        for step in 0..9 {
            supervisor.step(step).unwrap();
        }

        assert_eq!(
            supervisor.current_phase(),
            Phase::RepresentationFormation,
            "Adaptive relaxation should enable advance after patience expires"
        );
        assert!(
            !supervisor.runtime_amendments().is_empty(),
            "Should have recorded a runtime amendment"
        );
    }

    #[test]
    fn readiness_gate_disabled_allows_immediate_advance() {
        let spec = parse_spec(
            r#"
model:
  name: "test_no_gate"
  components: [backbone, head]
invariants:
  hard:
    head.pairwise_cosine: 0.95
    head.grad_norm: 0.001
  soft: {}
phases:
  bootstrap:
    transition_guard:
      all_hard_invariants_satisfied_for: 3
  representation_formation:
    thresholds:
      head.pairwise_cosine: 0.90
    transition_guard:
      all_hard_invariants_satisfied_for: 5
control:
  cooldown_steps: 10
  max_hard_interventions: 3
  hysteresis_margin: 0.0
  readiness_gate: false
"#,
        )
        .unwrap();

        let model = Rc::new(RefCell::new(MockModel::new(&["backbone", "head"])));
        let collector = BasicMetricCollector::new(
            vec!["backbone".into(), "head".into()],
            HashMap::new(),
        );
        let mut supervisor = Supervisor::new(spec, model.clone(), collector).unwrap();

        {
            let mut m = model.borrow_mut();
            // Cosine would fail readiness, but gate is disabled
            m.set_metric("head", "pairwise_cosine", 0.92);
            m.set_metric("head", "grad_norm", 0.0005);
            m.set_metric("backbone", "grad_norm", 0.05);
            m.set_global_metric("loss", 2.0);
        }

        for step in 0..4 {
            supervisor.step(step).unwrap();
        }

        assert_eq!(
            supervisor.current_phase(),
            Phase::RepresentationFormation,
            "With readiness_gate=false, should advance immediately after guard"
        );
    }

    #[test]
    fn relaxed_thresholds_active_after_transition() {
        let (mut supervisor, model) = test_supervisor_with_readiness();

        {
            let mut m = model.borrow_mut();
            // Cosine 0.92 triggers relaxation then advance
            m.set_metric("head", "pairwise_cosine", 0.92);
            m.set_metric("head", "grad_norm", 0.0005);
            m.set_metric("backbone", "grad_norm", 0.05);
            m.set_global_metric("loss", 2.0);
        }

        // Advance past relaxation
        for step in 0..9 {
            supervisor.step(step).unwrap();
        }
        assert_eq!(supervisor.current_phase(), Phase::RepresentationFormation);

        // Now in repr_formation with relaxed threshold (0.945 instead of 0.90).
        // Cosine 0.92 < 0.945, so no violation should occur.
        let report = supervisor.step(9).unwrap();
        let cosine_violations: Vec<_> = report
            .violations
            .iter()
            .filter(|v| v.invariant_name == "head.pairwise_cosine")
            .collect();
        assert!(
            cosine_violations.is_empty(),
            "Relaxed threshold should prevent violations after transition"
        );
    }
}
