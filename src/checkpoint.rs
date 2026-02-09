//! TransXform V1.4 — Supervisor Checkpointing
//!
//! Save and restore the supervisor's runtime state so training can be
//! resumed from a known-good point. The model and optimizer are the
//! user's responsibility — TransXform is framework-agnostic.

use std::collections::{HashMap, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::diagnostic::{DiagnosticConfig, InterventionOutcomeRecord};
use crate::error::TransXformError;
use crate::ledger::LedgerEntry;
use crate::regret::RegretWindow;
use crate::types::*;

/// Format version. Deserialization rejects checkpoints with a higher version.
pub const CHECKPOINT_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// CheckpointHint — advisory "good time to save"
// ---------------------------------------------------------------------------

/// Hint that the supervisor recommends checkpointing at this point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointHint {
    pub reason: CheckpointReason,
    pub step: u64,
}

/// Why the supervisor recommends checkpointing now.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointReason {
    /// Phase transition is imminent (N steps away).
    PrePhaseTransition {
        current_phase: Phase,
        steps_to_transition: u64,
    },
    /// Phase transition just completed — clean state to save.
    PostPhaseTransition { from: Phase, to: Phase },
    /// User-requested cadence (every N steps).
    Cadence,
}

// ---------------------------------------------------------------------------
// SupervisorCheckpoint — the full serializable state
// ---------------------------------------------------------------------------

/// The complete supervisor checkpoint — all mutable runtime state.
///
/// This struct is the **sole contract** between save and restore.
/// It contains no references to the model, collector, or spec.
/// Those must be provided separately on resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisorCheckpoint {
    /// Format version for forward-compatibility.
    pub version: u32,
    /// Timestamp when the checkpoint was created.
    pub created_at: DateTime<Utc>,
    /// The step number at which this checkpoint was taken.
    pub step: u64,
    /// Model name from the spec (for sanity checking on restore).
    pub model_name: String,

    // --- Supervisor-level state ---
    pub metric_history: VecDeque<MetricSnapshot>,
    pub metric_history_max: usize,
    pub pending_lr: Vec<(String, f64)>,
    pub aborted: bool,
    pub abort_reason: Option<String>,
    pub total_interventions: u64,
    pub runtime_amendments: Vec<RuntimeAmendment>,
    pub readiness_overrides: HashMap<String, f64>,
    pub active_overrides: HashMap<String, f64>,

    // --- Sub-component state ---
    pub phase_controller: PhaseControllerState,
    pub control_laws: ControlLawsState,
    pub ledger: LedgerState,
    pub regret_tracker: RegretTrackerState,
    pub diagnostic: DiagnosticState,
    pub registry: RegistryState,
}

/// Serializable state extracted from PhaseController.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseControllerState {
    pub current: Phase,
    pub consecutive_satisfied: u64,
    pub phase_entry_step: u64,
    pub regression_count: HashMap<String, u32>,
    pub phase_history: Vec<PhaseTransition>,
    pub readiness_blocked_since: Option<u64>,
}

/// Serializable state extracted from ControlLaws.
///
/// Note: `intervention_counts` uses `"component::phase_name"` string keys
/// because `HashMap<(String, Phase), u32>` does not serialize to JSON directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlLawsState {
    pub cooldown_until: HashMap<String, u64>,
    pub intervention_counts: HashMap<String, u32>,
}

/// Serializable state extracted from BoundaryLedger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerState {
    pub entries: Vec<LedgerEntry>,
    pub start_time: DateTime<Utc>,
    pub invariant_check_counts: HashMap<String, u64>,
    pub invariant_violation_counts: HashMap<String, u64>,
}

/// Serializable state extracted from RegretTracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegretTrackerState {
    pub windows: Vec<RegretWindow>,
    pub near_misses: Vec<NearMiss>,
    pub regret_window_length: u64,
}

/// Serializable state extracted from DiagnosticLayer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticState {
    pub config: DiagnosticConfig,
    pub history: VecDeque<MetricSnapshot>,
    pub components: Vec<String>,
    pub warnings: Vec<DiagnosticWarning>,
    pub active_signals: Vec<(DiagnosticSignal, String)>,
    pub expected_metrics: Vec<String>,
    #[serde(default)]
    pub best_loss: Option<f64>,
    #[serde(default)]
    pub best_loss_step: u64,
    #[serde(default)]
    pub invariant_thresholds: HashMap<String, (f64, ThresholdDirection)>,
    #[serde(default)]
    pub intervention_outcomes: Vec<InterventionOutcomeRecord>,
}

/// Serializable state extracted from SignatureRegistry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryState {
    pub match_state: HashMap<String, u64>,
}

// ---------------------------------------------------------------------------
// Serialization convenience methods
// ---------------------------------------------------------------------------

impl SupervisorCheckpoint {
    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, TransXformError> {
        serde_json::to_string_pretty(self).map_err(TransXformError::Json)
    }

    /// Deserialize from JSON string.
    pub fn from_json(s: &str) -> Result<Self, TransXformError> {
        let checkpoint: Self = serde_json::from_str(s).map_err(TransXformError::Json)?;
        if checkpoint.version > CHECKPOINT_VERSION {
            return Err(TransXformError::CheckpointError(format!(
                "Checkpoint version {} is newer than supported version {}",
                checkpoint.version, CHECKPOINT_VERSION,
            )));
        }
        Ok(checkpoint)
    }

    /// Serialize to YAML string.
    pub fn to_yaml(&self) -> Result<String, TransXformError> {
        serde_yaml::to_string(self).map_err(TransXformError::Yaml)
    }

    /// Deserialize from YAML string.
    pub fn from_yaml(s: &str) -> Result<Self, TransXformError> {
        let checkpoint: Self = serde_yaml::from_str(s).map_err(TransXformError::Yaml)?;
        if checkpoint.version > CHECKPOINT_VERSION {
            return Err(TransXformError::CheckpointError(format!(
                "Checkpoint version {} is newer than supported version {}",
                checkpoint.version, CHECKPOINT_VERSION,
            )));
        }
        Ok(checkpoint)
    }

    /// Write to a JSON file.
    pub fn save_json(&self, path: &std::path::Path) -> Result<(), TransXformError> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(TransXformError::Io)
    }

    /// Read from a JSON file.
    pub fn load_json(path: &std::path::Path) -> Result<Self, TransXformError> {
        let contents = std::fs::read_to_string(path).map_err(TransXformError::Io)?;
        Self::from_json(&contents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_checkpoint() -> SupervisorCheckpoint {
        SupervisorCheckpoint {
            version: CHECKPOINT_VERSION,
            created_at: Utc::now(),
            step: 100,
            model_name: "test_model".into(),
            metric_history: VecDeque::new(),
            metric_history_max: 200,
            pending_lr: vec![],
            aborted: false,
            abort_reason: None,
            total_interventions: 5,
            runtime_amendments: vec![],
            readiness_overrides: HashMap::new(),
            active_overrides: HashMap::new(),
            phase_controller: PhaseControllerState {
                current: Phase::RepresentationFormation,
                consecutive_satisfied: 12,
                phase_entry_step: 50,
                regression_count: HashMap::new(),
                phase_history: vec![PhaseTransition {
                    from: Phase::Bootstrap,
                    to: Phase::RepresentationFormation,
                    step: 50,
                    reason: "All hard invariants satisfied for 10 steps".into(),
                }],
                readiness_blocked_since: None,
            },
            control_laws: ControlLawsState {
                cooldown_until: HashMap::new(),
                intervention_counts: {
                    let mut m = HashMap::new();
                    m.insert("head::bootstrap".into(), 2);
                    m
                },
            },
            ledger: LedgerState {
                entries: vec![],
                start_time: Utc::now(),
                invariant_check_counts: HashMap::new(),
                invariant_violation_counts: HashMap::new(),
            },
            regret_tracker: RegretTrackerState {
                windows: vec![],
                near_misses: vec![],
                regret_window_length: 100,
            },
            diagnostic: DiagnosticState {
                config: crate::diagnostic::DiagnosticConfig::default(),
                history: VecDeque::new(),
                components: vec!["backbone".into(), "head".into()],
                warnings: vec![],
                active_signals: vec![],
                expected_metrics: vec![],
                best_loss: None,
                best_loss_step: 0,
                invariant_thresholds: HashMap::new(),
                intervention_outcomes: vec![],
            },
            registry: RegistryState {
                match_state: HashMap::new(),
            },
        }
    }

    #[test]
    fn json_roundtrip() {
        let checkpoint = minimal_checkpoint();
        let json = checkpoint.to_json().unwrap();
        let restored = SupervisorCheckpoint::from_json(&json).unwrap();
        assert_eq!(restored.version, CHECKPOINT_VERSION);
        assert_eq!(restored.step, 100);
        assert_eq!(restored.model_name, "test_model");
        assert_eq!(restored.total_interventions, 5);
        assert_eq!(
            restored.phase_controller.current,
            Phase::RepresentationFormation
        );
        assert_eq!(
            restored.control_laws.intervention_counts.get("head::bootstrap"),
            Some(&2)
        );
    }

    #[test]
    fn yaml_roundtrip() {
        let checkpoint = minimal_checkpoint();
        let yaml = checkpoint.to_yaml().unwrap();
        let restored = SupervisorCheckpoint::from_yaml(&yaml).unwrap();
        assert_eq!(restored.step, 100);
        assert_eq!(restored.model_name, "test_model");
    }

    #[test]
    fn rejects_future_version() {
        let mut checkpoint = minimal_checkpoint();
        checkpoint.version = 999;
        let json = serde_json::to_string(&checkpoint).unwrap();
        let result = SupervisorCheckpoint::from_json(&json);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("newer than supported"));
    }

    #[test]
    fn file_roundtrip() {
        let checkpoint = minimal_checkpoint();
        let dir = std::env::temp_dir();
        let path = dir.join("transxform_test_checkpoint.json");
        checkpoint.save_json(&path).unwrap();
        let restored = SupervisorCheckpoint::load_json(&path).unwrap();
        assert_eq!(restored.step, 100);
        assert_eq!(restored.model_name, "test_model");
        // Clean up
        let _ = std::fs::remove_file(&path);
    }
}
