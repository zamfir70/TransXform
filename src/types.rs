use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity of an invariant — determines intervention style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Hard,
    Soft,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Hard => write!(f, "hard"),
            Severity::Soft => write!(f, "soft"),
        }
    }
}

// ---------------------------------------------------------------------------
// Phase
// ---------------------------------------------------------------------------

/// Training phase (whitepaper §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Bootstrap,
    RepresentationFormation,
    Stabilization,
    Refinement,
    Aborted,
}

impl Phase {
    /// Return the next phase in the default sequence, or `None` if at the end
    /// or already aborted.
    pub fn next(self) -> Option<Phase> {
        match self {
            Phase::Bootstrap => Some(Phase::RepresentationFormation),
            Phase::RepresentationFormation => Some(Phase::Stabilization),
            Phase::Stabilization => Some(Phase::Refinement),
            Phase::Refinement => None,
            Phase::Aborted => None,
        }
    }

    /// Return the previous phase in the default sequence, or `None` if at the
    /// beginning or aborted.
    pub fn prev(self) -> Option<Phase> {
        match self {
            Phase::Bootstrap => None,
            Phase::RepresentationFormation => Some(Phase::Bootstrap),
            Phase::Stabilization => Some(Phase::RepresentationFormation),
            Phase::Refinement => Some(Phase::Stabilization),
            Phase::Aborted => None,
        }
    }

    /// Whether this is a terminal phase (no further progression).
    pub fn is_terminal(self) -> bool {
        matches!(self, Phase::Aborted)
    }
}

impl std::str::FromStr for Phase {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bootstrap" => Ok(Phase::Bootstrap),
            "representation_formation" => Ok(Phase::RepresentationFormation),
            "stabilization" => Ok(Phase::Stabilization),
            "refinement" => Ok(Phase::Refinement),
            "aborted" => Ok(Phase::Aborted),
            _ => Err(format!("Unknown phase: {}", s)),
        }
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Phase::Bootstrap => write!(f, "bootstrap"),
            Phase::RepresentationFormation => write!(f, "representation_formation"),
            Phase::Stabilization => write!(f, "stabilization"),
            Phase::Refinement => write!(f, "refinement"),
            Phase::Aborted => write!(f, "aborted"),
        }
    }
}

// ---------------------------------------------------------------------------
// ThresholdDirection
// ---------------------------------------------------------------------------

/// Whether the threshold is an upper bound or lower bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdDirection {
    /// Metric must stay above this value (e.g., variance floor).
    Min,
    /// Metric must stay below this value (e.g., cosine ceiling).
    Max,
}

impl fmt::Display for ThresholdDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThresholdDirection::Min => write!(f, "min"),
            ThresholdDirection::Max => write!(f, "max"),
        }
    }
}

// ---------------------------------------------------------------------------
// MetricTier
// ---------------------------------------------------------------------------

/// Metric collection tier (whitepaper §5.1) — controls collection cadence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricTier {
    /// Computed every step: grad norms, activation variance, pairwise cosine.
    Tier0,
    /// Computed every N steps: attention entropy, MI, drift.
    Tier1,
    /// Computed on demand: spectrum analysis, curvature, Jacobian.
    Tier2,
}

// ---------------------------------------------------------------------------
// Violation
// ---------------------------------------------------------------------------

/// A detected invariant violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub invariant_name: String,
    pub component: String,
    pub severity: Severity,
    pub observed: f64,
    pub threshold: f64,
    pub direction: ThresholdDirection,
    pub step: u64,
    /// Passive violations are observed but not acted on.
    /// Set by the supervisor based on the component's role declaration.
    #[serde(default)]
    pub passive: bool,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tag = if self.passive { " (passive)" } else { "" };
        write!(
            f,
            "[{}] {}.{}: observed={:.6}, threshold={:.6} ({}){}",
            self.severity, self.component, self.invariant_name, self.observed,
            self.threshold, self.direction, tag,
        )
    }
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

/// An intervention action the supervisor can take (whitepaper §7).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    Reinitialize { component: String },
    Freeze { component: String },
    Unfreeze { component: String },
    Rescale { component: String, factor: f64 },
    InjectNoise { component: String, magnitude: f64 },
    AdjustLr { component: String, factor: f64 },
    Abort { reason: String },
}

impl Action {
    /// The action name as a string, for matching against allowed_interventions.
    pub fn action_name(&self) -> &str {
        match self {
            Action::Reinitialize { .. } => "reinitialize",
            Action::Freeze { .. } => "freeze",
            Action::Unfreeze { .. } => "unfreeze",
            Action::Rescale { .. } => "rescale",
            Action::InjectNoise { .. } => "inject_noise",
            Action::AdjustLr { .. } => "adjust_lr",
            Action::Abort { .. } => "abort",
        }
    }

    /// The component this action targets, if any.
    pub fn component(&self) -> Option<&str> {
        match self {
            Action::Reinitialize { component }
            | Action::Freeze { component }
            | Action::Unfreeze { component }
            | Action::Rescale { component, .. }
            | Action::InjectNoise { component, .. }
            | Action::AdjustLr { component, .. } => Some(component),
            Action::Abort { .. } => None,
        }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Reinitialize { component } => write!(f, "reinitialize({})", component),
            Action::Freeze { component } => write!(f, "freeze({})", component),
            Action::Unfreeze { component } => write!(f, "unfreeze({})", component),
            Action::Rescale { component, factor } => {
                write!(f, "rescale({}, {:.4})", component, factor)
            }
            Action::InjectNoise {
                component,
                magnitude,
            } => write!(f, "inject_noise({}, {:.6})", component, magnitude),
            Action::AdjustLr { component, factor } => {
                write!(f, "adjust_lr({}, {:.4})", component, factor)
            }
            Action::Abort { reason } => write!(f, "abort({})", reason),
        }
    }
}

// ---------------------------------------------------------------------------
// MetricSnapshot
// ---------------------------------------------------------------------------

/// A snapshot of metrics at a given step.
pub type MetricSnapshot = HashMap<String, f64>;

// ---------------------------------------------------------------------------
// NegativeVerdict
// ---------------------------------------------------------------------------

/// Negative capability verdicts (whitepaper §13).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NegativeVerdict {
    UnsatisfiableSpec,
    UnstableArchitecture,
    InsufficientSignal,
    DegenerateObjective,
}

// ---------------------------------------------------------------------------
// HealthVerdict
// ---------------------------------------------------------------------------

/// Training health verdict for the certificate (whitepaper §10.2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HealthVerdict {
    Healthy,
    Recovered { intervention_count: u64 },
    Compromised { details: String },
}

impl fmt::Display for HealthVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HealthVerdict::Healthy => write!(f, "HEALTHY"),
            HealthVerdict::Recovered {
                intervention_count,
            } => write!(f, "RECOVERED ({} interventions)", intervention_count),
            HealthVerdict::Compromised { details } => write!(f, "COMPROMISED: {}", details),
        }
    }
}

// ---------------------------------------------------------------------------
// RegretTag
// ---------------------------------------------------------------------------

/// Regret tag for interventions (whitepaper §6.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegretTag {
    Confident,
    LowConfidence,
    Pending,
}

// ---------------------------------------------------------------------------
// NearMiss
// ---------------------------------------------------------------------------

/// Near-miss record (whitepaper §6.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearMiss {
    pub step: u64,
    pub invariant_name: String,
    pub component: String,
    pub observed: f64,
    pub hard_threshold: f64,
    pub margin: f64,
    pub metric_snapshot: MetricSnapshot,
}

// ---------------------------------------------------------------------------
// RuntimeAmendment
// ---------------------------------------------------------------------------

/// A runtime threshold amendment — applied when the supervisor determines
/// that a spec threshold is unachievable and bounded relaxation is needed
/// to prevent stuck training.
///
/// Amendments are logged in the boundary ledger, appear in the training
/// certificate, and use calm language: "relaxed," "adapted," "within bounds."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeAmendment {
    pub step: u64,
    pub metric_key: String,
    pub phase: Phase,
    pub original_threshold: f64,
    pub relaxed_threshold: f64,
    pub reason: String,
}

impl fmt::Display for RuntimeAmendment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "step {}: {}.{} relaxed {:.6} → {:.6} ({})",
            self.step, self.phase, self.metric_key,
            self.original_threshold, self.relaxed_threshold, self.reason,
        )
    }
}

// ---------------------------------------------------------------------------
// ShadowStepVerdict (V1.5)
// ---------------------------------------------------------------------------

/// Shadow-step verdict — recommendation to the training loop about whether
/// the most recent optimizer step should be rolled back (V1.5, §15.6).
///
/// TransXform is framework-agnostic: it recommends, the training loop decides.
/// Model checkpointing is the user's responsibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum ShadowStepVerdict {
    /// Shadow-stepping is disabled — no recommendation.
    None,
    /// The optimizer step did not introduce new hard violations — commit it.
    Commit,
    /// The optimizer step introduced new hard violations that were not present
    /// in the previous step. Rolling back is recommended.
    RollbackRecommended {
        violations: Vec<Violation>,
    },
    /// Hard violations existed before the optimizer step — rollback would not
    /// help. Normal intervention proceeds.
    InterventionRequired,
}

impl fmt::Display for ShadowStepVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShadowStepVerdict::None => write!(f, "none"),
            ShadowStepVerdict::Commit => write!(f, "commit"),
            ShadowStepVerdict::RollbackRecommended { violations } => {
                write!(f, "rollback_recommended ({} violations)", violations.len())
            }
            ShadowStepVerdict::InterventionRequired => write!(f, "intervention_required"),
        }
    }
}

// ---------------------------------------------------------------------------
// PhaseTransition
// ---------------------------------------------------------------------------

/// A recorded phase transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransition {
    pub from: Phase,
    pub to: Phase,
    pub step: u64,
    pub reason: String,
}

impl fmt::Display for PhaseTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "step {}: {} → {} ({})",
            self.step, self.from, self.to, self.reason
        )
    }
}

// ---------------------------------------------------------------------------
// DiagnosticSignal (V2 — advisory layer)
// ---------------------------------------------------------------------------

/// The thirteen epistemic early-warning signals (V2 diagnostic layer).
///
/// These are advisory-only — they surface evidence of potential problems
/// but make no claims of correctness and perform no interventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSignal {
    /// Attention heads with near-zero variance, layers with pass-through
    /// behavior, parameters that never update meaningfully.
    UnusedCapacity,
    /// Higher-order pathways (multi-hop reasoning, compositional structure)
    /// remain dormant despite architectural capacity.
    MissingStructuralSignal,
    /// Loss decreases but declared objective metrics don't move; learning
    /// happens in subspaces unrelated to task.
    LossRepresentationMisalignment,
    /// Gradients exist but update scale prevents escape from basin; LR too
    /// low to climb out, too high to settle.
    DynamicallyUnlearnableRegime,
    /// High task performance but representation rank collapses; model ignores
    /// input structure, exploits surface statistics.
    ShortcutLearning,
    /// Loss has not improved meaningfully for an extended period despite
    /// healthy gradient flow. The model is trying but making no progress —
    /// possible causes include noisy data, capacity mismatch, or a flat
    /// loss basin.
    LossStagnation,
    /// An invariant's expected metric key has never appeared in any metric
    /// snapshot. The invariant is silently inactive — a configuration or
    /// integration error between the spec and the training loop.
    MissingExpectedMetric,
    /// A metric is trending monotonically toward its invariant threshold.
    /// Early detection enables preemptive advisory before reactive intervention.
    ThresholdDrift,
    /// A metric exhibits high-frequency oscillation (high coefficient of
    /// variation) indicating training instability even when the metric never
    /// crosses its invariant threshold.
    MetricInstability,
    /// Repeated interventions on the same component fail to improve the
    /// metric. The supervisor keeps trying, but the architecture or data
    /// may be the root cause.
    InterventionFutility,
    /// One component's gradient norms are orders of magnitude larger than
    /// others, monopolizing optimizer updates. Suppressed components may
    /// not receive meaningful parameter updates.
    GradientDomination,
    /// A metric value is NaN or infinite — numerical corruption that will
    /// propagate through all downstream computations.
    MetricAnomaly,
    /// Training loss is decreasing while validation loss is increasing.
    /// This divergence pattern is consistent with overfitting.
    TrainValDivergence,
}

impl fmt::Display for DiagnosticSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagnosticSignal::UnusedCapacity => write!(f, "unused_capacity"),
            DiagnosticSignal::MissingStructuralSignal => write!(f, "missing_structural_signal"),
            DiagnosticSignal::LossRepresentationMisalignment => {
                write!(f, "loss_representation_misalignment")
            }
            DiagnosticSignal::DynamicallyUnlearnableRegime => {
                write!(f, "dynamically_unlearnable_regime")
            }
            DiagnosticSignal::ShortcutLearning => write!(f, "shortcut_learning"),
            DiagnosticSignal::LossStagnation => write!(f, "loss_stagnation"),
            DiagnosticSignal::MissingExpectedMetric => write!(f, "missing_expected_metric"),
            DiagnosticSignal::ThresholdDrift => write!(f, "threshold_drift"),
            DiagnosticSignal::MetricInstability => write!(f, "metric_instability"),
            DiagnosticSignal::InterventionFutility => write!(f, "intervention_futility"),
            DiagnosticSignal::GradientDomination => write!(f, "gradient_domination"),
            DiagnosticSignal::MetricAnomaly => write!(f, "metric_anomaly"),
            DiagnosticSignal::TrainValDivergence => write!(f, "train_val_divergence"),
        }
    }
}

/// A diagnostic warning — advisory, never authoritative.
///
/// Language is deliberately calm and precise: "observed," "consistent with,"
/// "suggests," "unlikely under." No drama. No claims of correctness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticWarning {
    pub signal: DiagnosticSignal,
    pub step: u64,
    /// One-line summary in calm, precise language.
    pub summary: String,
    /// Evidence statements supporting the warning.
    pub evidence: Vec<String>,
    /// How confident the signal is (0.0–1.0). Higher values indicate
    /// stronger evidence, not certainty.
    pub confidence: f64,
    /// Whether the user has acknowledged this warning.
    pub acknowledged: bool,
}

impl fmt::Display for DiagnosticWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[advisory:{}] step {}: {} (confidence: {:.2})",
            self.signal, self.step, self.summary, self.confidence,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_sequence() {
        assert_eq!(Phase::Bootstrap.next(), Some(Phase::RepresentationFormation));
        assert_eq!(Phase::RepresentationFormation.next(), Some(Phase::Stabilization));
        assert_eq!(Phase::Stabilization.next(), Some(Phase::Refinement));
        assert_eq!(Phase::Refinement.next(), None);
        assert_eq!(Phase::Aborted.next(), None);
    }

    #[test]
    fn phase_regression() {
        assert_eq!(Phase::Refinement.prev(), Some(Phase::Stabilization));
        assert_eq!(Phase::Stabilization.prev(), Some(Phase::RepresentationFormation));
        assert_eq!(Phase::RepresentationFormation.prev(), Some(Phase::Bootstrap));
        assert_eq!(Phase::Bootstrap.prev(), None);
    }

    #[test]
    fn action_names() {
        let a = Action::Reinitialize { component: "x".into() };
        assert_eq!(a.action_name(), "reinitialize");
        assert_eq!(a.component(), Some("x"));

        let b = Action::Abort { reason: "test".into() };
        assert_eq!(b.action_name(), "abort");
        assert_eq!(b.component(), None);
    }

    #[test]
    fn action_serde_roundtrip() {
        let a = Action::Rescale { component: "head".into(), factor: 0.5 };
        let json = serde_json::to_string(&a).unwrap();
        let b: Action = serde_json::from_str(&json).unwrap();
        assert_eq!(a.action_name(), b.action_name());
    }

    #[test]
    fn severity_display() {
        assert_eq!(format!("{}", Severity::Hard), "hard");
        assert_eq!(format!("{}", Severity::Soft), "soft");
    }
}
