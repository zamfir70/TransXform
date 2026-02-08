//! Witness Console — a live TUI dashboard for monitoring TransXform training runs.
//!
//! Available only with the `witness` feature.
//!
//! The console runs on a background thread and receives `SupervisorReport`s
//! via an `mpsc` channel. It is designed to be "quiet by default" — no
//! scrolling logs, no spam. Suitable for overnight training runs.

#[cfg(feature = "witness")]
mod app;
#[cfg(feature = "witness")]
mod markers;
#[cfg(feature = "witness")]
mod phase_strip;
#[cfg(feature = "witness")]
mod timeline;

#[cfg(feature = "witness")]
pub use app::WitnessApp;
