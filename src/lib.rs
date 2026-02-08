pub mod error;
pub mod model;
pub mod spec;
pub mod types;

pub mod collector;
pub mod control;
pub mod diagnostic;
pub mod executor;
pub mod ledger;
pub mod monitor;
pub mod phase;
pub mod registry;
pub mod regret;
pub mod supervisor;

pub mod merkle;
pub mod report;

pub mod witness;

#[cfg(feature = "tch")]
pub mod tch_backend;

// Re-exports
pub use error::TransXformError;
pub use model::{MockModel, Model};
pub use spec::{parse_spec, parse_spec_from_file, TrainingSpec};
pub use types::*;

pub use collector::{BasicMetricCollector, MetricCollector};
pub use control::ControlLaws;
pub use executor::InterventionExecutor;
pub use ledger::{BoundaryLedger, TrainingCertificate};
pub use monitor::InvariantMonitor;
pub use phase::PhaseController;
pub use registry::SignatureRegistry;
pub use regret::RegretTracker;
pub use supervisor::Supervisor;

pub use diagnostic::{DiagnosticConfig, DiagnosticLayer};
pub use merkle::MerkleState;
pub use report::{generate_report, generate_report_with_diagnostics};

#[cfg(feature = "witness")]
pub use witness::WitnessApp;

#[cfg(feature = "tch")]
pub use tch_backend::{TchMetricCollector, TchModel};
