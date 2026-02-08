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
    /// Extensible: users can add custom role properties.
    #[serde(flatten)]
    pub custom: HashMap<String, serde_yaml::Value>,
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
    #[serde(default = "default_hysteresis")]
    pub hysteresis_margin: f64,
    #[serde(default = "default_damping")]
    pub damping_factor: f64,
    #[serde(default)]
    pub base_lr: Option<f64>,
    #[serde(default = "default_regret_window")]
    pub regret_window_steps: u64,
}

fn default_cooldown() -> u64 { 50 }
fn default_max_hard() -> u32 { 3 }
fn default_hysteresis() -> f64 { 0.05 }
fn default_damping() -> f64 { 0.5 }
fn default_regret_window() -> u64 { 100 }

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
