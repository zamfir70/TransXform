//! PyTorch integration via `tch-rs` (feature = "tch").
//!
//! Provides `TchModel` (implementing the `Model` trait) and `TchMetricCollector`
//! for computing metrics directly from PyTorch tensors.

#[cfg(feature = "tch")]
mod model;
#[cfg(feature = "tch")]
mod collector;

#[cfg(feature = "tch")]
pub use model::TchModel;
#[cfg(feature = "tch")]
pub use collector::TchMetricCollector;
