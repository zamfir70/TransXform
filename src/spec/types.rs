use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// The complete training specification (whitepaper §4).
/// Parsed from YAML at startup. Immutable after construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSpec {
    pub model: ModelDecl,
    #[serde(default)]
    pub roles: HashMap<String, RoleDecl>,
    pub invariants: InvariantBlock,
    pub phases: PhasesDecl,
    pub control: ControlConfig,
    #[serde(default)]
    pub metric_cadence: HashMap<String, u64>,
    #[serde(default)]
    pub profiles: Vec<ProfileRef>,
}

/// Model declaration — names the model and its components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDecl {
    pub name: String,
    #[serde(default)]
    pub layers: Option<u32>,
    #[serde(default)]
    pub hidden_dim: Option<u32>,
    #[serde(default)]
    pub attention_heads: Option<u32>,
    pub components: Vec<String>,
}

/// Declared role for a component — encodes intent so the supervisor can
/// distinguish intentional low entropy from pathological low entropy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleDecl {
    #[serde(default)]
    pub diversity_required: Option<bool>,
    #[serde(default)]
    pub min_active_heads: Option<u32>,
    #[serde(default)]
    pub must_preserve_variance: Option<bool>,
    #[serde(default)]
    pub must_maintain_gradient: Option<bool>,
    #[serde(default)]
    pub output_diversity_required: Option<bool>,
    /// Passive components are observed but not intervened on.
    /// Metrics are collected, violations appear in reports and diagnostics,
    /// but no interventions are generated and violations don't block phase
    /// transitions. Use for components with no loss term yet, auxiliary
    /// heads that can be toggled off, or anything you want to monitor
    /// without the supervisor acting on.
    #[serde(default)]
    pub passive: Option<bool>,
    /// Declares this component's upstream dependency. When this component
    /// collapses, the supervisor checks the upstream component's health
    /// before deciding where to intervene. If omitted, defaults to
    /// "backbone" (the v1.0 heuristic). Set to trace multi-level
    /// architectures like `backbone → compressor → emission_head`.
    #[serde(default)]
    pub upstream: Option<String>,
    /// Extensible: users can add custom role properties.
    #[serde(flatten)]
    pub custom: HashMap<String, serde_yaml::Value>,
}

impl TrainingSpec {
    /// Check if a component is marked as passive (observe only, no interventions).
    pub fn is_passive(&self, component: &str) -> bool {
        self.roles
            .get(component)
            .and_then(|r| r.passive)
            .unwrap_or(false)
    }

    /// Build a map of component → upstream component from role declarations.
    /// Components without an explicit upstream declaration are not included
    /// (the control law will fall back to the default "backbone" heuristic).
    pub fn upstream_map(&self) -> HashMap<String, String> {
        self.roles
            .iter()
            .filter_map(|(comp, role)| {
                role.upstream.as_ref().map(|u| (comp.clone(), u.clone()))
            })
            .collect()
    }
}

/// Hard and soft invariant declarations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantBlock {
    #[serde(default)]
    pub hard: HashMap<String, InvariantValue>,
    #[serde(default)]
    pub soft: HashMap<String, InvariantValue>,
}

/// An invariant value: either a single scalar threshold or a per-component map.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InvariantValue {
    Scalar(f64),
    PerComponent(HashMap<String, f64>),
}

/// Phase declarations — one optional block per phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasesDecl {
    #[serde(default)]
    pub bootstrap: Option<PhaseDecl>,
    #[serde(default)]
    pub representation_formation: Option<PhaseDecl>,
    #[serde(default)]
    pub stabilization: Option<PhaseDecl>,
    #[serde(default)]
    pub refinement: Option<PhaseDecl>,
}

/// Configuration for a single training phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseDecl {
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub thresholds: HashMap<String, f64>,
    #[serde(default)]
    pub max_duration_steps: Option<u64>,
    #[serde(default)]
    pub transition_guard: Option<TransitionGuard>,
    #[serde(default)]
    pub allowed_interventions: Option<Vec<String>>,
}

/// Condition for advancing to the next phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionGuard {
    pub all_hard_invariants_satisfied_for: u64,
}

/// Global control law configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlConfig {
    #[serde(default = "default_cooldown")]
    pub cooldown_steps: u64,
    #[serde(default = "default_max_hard")]
    pub max_hard_interventions: u32,
    /// Absolute hysteresis floor. When `hysteresis_pct > 0`, this is the
    /// minimum margin; otherwise it's the sole margin (backward-compatible).
    #[serde(default = "default_hysteresis")]
    pub hysteresis_margin: f64,
    /// Proportional hysteresis: margin = max(threshold * pct, hysteresis_margin).
    /// Set to 0.0 to disable (flat margin only). Default: 0.0.
    #[serde(default = "default_hysteresis_pct")]
    pub hysteresis_pct: f64,
    /// Catastrophic thresholds that fire with zero hysteresis, regardless of
    /// phase overrides. Maps invariant name (or metric_key) to absolute limit.
    #[serde(default)]
    pub catastrophic_overrides: HashMap<String, f64>,
    #[serde(default = "default_damping")]
    pub damping_factor: f64,
    #[serde(default)]
    pub base_lr: Option<f64>,
    #[serde(default = "default_regret_window")]
    pub regret_window_steps: u64,
    /// V1.3: Phase readiness gate. Before advancing to a new phase, check
    /// if current metrics already satisfy the next phase's thresholds.
    /// Prevents "cliff transitions" where the model enters a phase it can't
    /// survive. Default: true.
    #[serde(default = "default_readiness_gate")]
    pub readiness_gate: bool,
    /// V1.3: Steps to wait while the readiness gate blocks before relaxing
    /// thresholds. After this many steps of blockage, the supervisor begins
    /// adaptive threshold relaxation. Default: 200.
    #[serde(default = "default_readiness_patience")]
    pub readiness_patience_steps: u64,
    /// V1.3: Maximum percentage by which adaptive relaxation can widen a
    /// threshold per round. E.g., 0.02 means a Max-direction threshold of
    /// 0.98 can relax to at most 0.98 * (1 + 0.02) ≈ 0.9996.
    /// Default: 0.02 (2%).
    #[serde(default = "default_max_relaxation")]
    pub max_threshold_relaxation: f64,
}

impl ControlConfig {
    /// Compute the effective hysteresis margin for a given threshold.
    pub fn effective_margin(&self, threshold: f64) -> f64 {
        if self.hysteresis_pct > 0.0 {
            (threshold.abs() * self.hysteresis_pct).max(self.hysteresis_margin)
        } else {
            self.hysteresis_margin
        }
    }
}

fn default_cooldown() -> u64 { 50 }
fn default_max_hard() -> u32 { 3 }
fn default_hysteresis() -> f64 { 0.05 }
fn default_hysteresis_pct() -> f64 { 0.0 }
fn default_damping() -> f64 { 0.5 }
fn default_regret_window() -> u64 { 100 }
fn default_readiness_gate() -> bool { true }
fn default_readiness_patience() -> u64 { 200 }
fn default_max_relaxation() -> f64 { 0.02 }

/// Reference to a reusable architecture profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileRef {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub inherits: Option<String>,
    #[serde(default)]
    pub overrides: Option<serde_yaml::Value>,
}
