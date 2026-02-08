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
#[derive(Debug, Clone, Serialize)]
pub struct Violation {
    pub invariant_name: String,
    pub component: String,
    pub severity: Severity,
    pub observed: f64,
    pub threshold: f64,
    pub direction: ThresholdDirection,
    pub step: u64,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}.{}: observed={:.6}, threshold={:.6} ({})",
            self.severity, self.component, self.invariant_name, self.observed,
            self.threshold, self.direction,
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
#[derive(Debug, Clone, Serialize)]
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

/// The five epistemic early-warning signals (V2 diagnostic layer).
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
        }
    }
}

/// A diagnostic warning — advisory, never authoritative.
///
/// Language is deliberately calm and precise: "observed," "consistent with,"
/// "suggests," "unlikely under." No drama. No claims of correctness.
#[derive(Debug, Clone, Serialize)]
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
