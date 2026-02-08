use std::fmt;
use thiserror::Error;

use crate::types::NegativeVerdict;

#[derive(Debug, Error)]
pub enum TransXformError {
    #[error("Spec parse error: {0}")]
    SpecParse(String),

    #[error("Spec validation error: {0}")]
    SpecValidation(String),

    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Unknown component: {0}")]
    UnknownComponent(String),

    #[error("Intervention failed: {action} on {component}: {reason}")]
    InterventionFailed {
        action: String,
        component: String,
        reason: String,
    },

    #[error("Training aborted: {verdict}: {details}")]
    TrainingAborted {
        verdict: NegativeVerdict,
        details: String,
    },

    #[error("Phase error: {0}")]
    PhaseError(String),

    #[error("Metric error: {0}")]
    MetricError(String),
}

impl fmt::Display for NegativeVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NegativeVerdict::UnsatisfiableSpec => write!(f, "UNSATISFIABLE_SPEC"),
            NegativeVerdict::UnstableArchitecture => write!(f, "UNSTABLE_ARCHITECTURE"),
            NegativeVerdict::InsufficientSignal => write!(f, "INSUFFICIENT_SIGNAL"),
            NegativeVerdict::DegenerateObjective => write!(f, "DEGENERATE_OBJECTIVE"),
        }
    }
}
