//! # TransXform — Boundary-Governed Transformer Training Supervisor
//!
//! TransXform is a spec-driven training supervisor for deep learning models.
//! It enforces structural invariants (gradient norms, representation similarity,
//! activation variance) across training phases, intervenes when invariants are
//! violated, and provides advisory diagnostics for failure modes that hard
//! invariants cannot catch.
//!
//! ## Core Concepts
//!
//! - **Spec**: A YAML declaration of model components, invariants, phase
//!   thresholds, and control parameters. See [`TrainingSpec`] and [`parse_spec`].
//! - **Supervisor**: The main orchestrator. Each call to [`Supervisor::step`]
//!   collects metrics, checks invariants, dispatches interventions, updates
//!   phase state, and runs diagnostics.
//! - **Phases**: Training progresses through Bootstrap → RepresentationFormation
//!   → Stabilization → Refinement, with phase-specific thresholds and allowed
//!   interventions.
//! - **Diagnostic Layer (V2)**: Thirteen advisory signals that detect failure
//!   modes like unused capacity, shortcut learning, loss stagnation, gradient
//!   domination, and overfitting — without intervening.
//! - **Checkpointing**: Save and restore the full supervisor state mid-run.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use transxform::*;
//! use std::rc::Rc;
//! use std::cell::RefCell;
//!
//! let spec = parse_spec(SPEC_YAML).unwrap();
//! let model = Rc::new(RefCell::new(my_model));
//! let collector = my_collector;
//! let mut supervisor = Supervisor::new(spec, model, collector).unwrap();
//!
//! for step in 0..total_steps {
//!     let report = supervisor.step(step).unwrap();
//!     // Apply pending LR adjustments, check report.actions_taken, etc.
//! }
//! let cert = supervisor.emit_certificate(total_steps);
//! ```

pub mod error;
pub mod model;
pub mod spec;
pub mod types;

pub mod checkpoint;
pub mod collector;
pub mod control;
pub mod diagnostic;
pub mod discovery;
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

pub use checkpoint::{CheckpointHint, CheckpointReason, SupervisorCheckpoint};
pub use diagnostic::{DiagnosticConfig, DiagnosticLayer};
pub use discovery::{DiscoveryConfig, DiscoveryReport, ThresholdProposal};
pub use merkle::MerkleState;
pub use report::{generate_report, generate_report_with_diagnostics};

#[cfg(feature = "witness")]
pub use witness::WitnessApp;

#[cfg(feature = "tch")]
pub use tch_backend::{TchMetricCollector, TchModel};
