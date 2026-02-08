//! Merkle hashing for run integrity and reproducibility (spinning rims spec).
//!
//! This module is only available with the `merkle` feature enabled.

#[cfg(feature = "merkle")]
use sha2::{Digest, Sha256};

use serde::{Deserialize, Serialize};

use crate::types::{Action, MetricSnapshot};

/// Rolling Merkle state over the run.
///
/// Each step hashes `(previous_root, step, metric_snapshot, action_taken)`
/// into a new root, creating a tamper-evident chain of run state.
#[derive(Debug, Clone)]
pub struct MerkleState {
    /// Current Merkle root.
    root: [u8; 32],
    /// Number of steps hashed.
    step_count: u64,
}

/// A manifest capturing all inputs and the final Merkle root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub spec_hash: String,
    pub model_hash: String,
    pub optimizer_hash: String,
    pub initial_weights_hash: String,
    pub supervisor_version: String,
    pub final_merkle_root: String,
    pub step_count: u64,
}

/// A fork manifest recording which settings changed between runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForkManifest {
    pub parent_manifest: RunManifest,
    pub changes: Vec<String>,
    pub child_spec_hash: String,
}

/// Structural comparison between two run manifests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestDiff {
    pub spec_changed: bool,
    pub model_changed: bool,
    pub optimizer_changed: bool,
    pub weights_changed: bool,
    pub step_count_a: u64,
    pub step_count_b: u64,
    pub roots_match: bool,
}

impl MerkleState {
    /// Create a new Merkle state with an initial root derived from the spec.
    #[cfg(feature = "merkle")]
    pub fn new(spec_yaml: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"transxform-v0.1.0");
        hasher.update(spec_yaml.as_bytes());
        let result = hasher.finalize();
        let mut root = [0u8; 32];
        root.copy_from_slice(&result);
        Self {
            root,
            step_count: 0,
        }
    }

    /// Create a new Merkle state without the merkle feature (no-op).
    #[cfg(not(feature = "merkle"))]
    pub fn new(_spec_yaml: &str) -> Self {
        Self {
            root: [0u8; 32],
            step_count: 0,
        }
    }

    /// Update the Merkle root with a new step's data.
    #[cfg(feature = "merkle")]
    pub fn update(
        &mut self,
        step: u64,
        metrics: &MetricSnapshot,
        action: Option<&Action>,
    ) {
        let mut hasher = Sha256::new();
        // Previous root
        hasher.update(&self.root);
        // Step number
        hasher.update(step.to_le_bytes());
        // Metrics (sorted for determinism)
        let mut sorted_metrics: Vec<(&String, &f64)> = metrics.iter().collect();
        sorted_metrics.sort_by_key(|(k, _)| k.as_str());
        for (key, value) in sorted_metrics {
            hasher.update(key.as_bytes());
            hasher.update(value.to_le_bytes());
        }
        // Action taken (if any)
        if let Some(action) = action {
            hasher.update(format!("{}", action).as_bytes());
        }

        let result = hasher.finalize();
        self.root.copy_from_slice(&result);
        self.step_count += 1;
    }

    /// No-op update when merkle feature is disabled.
    #[cfg(not(feature = "merkle"))]
    pub fn update(
        &mut self,
        _step: u64,
        _metrics: &MetricSnapshot,
        _action: Option<&Action>,
    ) {
        self.step_count += 1;
    }

    /// Get the current root as a hex string.
    pub fn root_hex(&self) -> String {
        hex_encode(&self.root)
    }

    /// Get the raw root bytes.
    pub fn root(&self) -> &[u8; 32] {
        &self.root
    }

    /// Get the number of steps hashed.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }
}

/// Hash a string with SHA-256 and return the hex digest.
#[cfg(feature = "merkle")]
pub fn hash_string(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex_encode(&hasher.finalize())
}

#[cfg(not(feature = "merkle"))]
pub fn hash_string(_input: &str) -> String {
    "0".repeat(64)
}

/// Build a run manifest from component hashes.
pub fn build_manifest(
    spec_yaml: &str,
    model_description: &str,
    optimizer_description: &str,
    initial_weights_hash: &str,
    merkle_state: &MerkleState,
) -> RunManifest {
    RunManifest {
        spec_hash: hash_string(spec_yaml),
        model_hash: hash_string(model_description),
        optimizer_hash: hash_string(optimizer_description),
        initial_weights_hash: initial_weights_hash.to_string(),
        supervisor_version: "transxform-v0.1.0".to_string(),
        final_merkle_root: merkle_state.root_hex(),
        step_count: merkle_state.step_count(),
    }
}

/// Create a fork manifest recording changes from a parent run.
pub fn fork(parent: &RunManifest, child_spec_yaml: &str, changes: Vec<String>) -> ForkManifest {
    ForkManifest {
        parent_manifest: parent.clone(),
        changes,
        child_spec_hash: hash_string(child_spec_yaml),
    }
}

/// Diff two run manifests.
pub fn diff(a: &RunManifest, b: &RunManifest) -> ManifestDiff {
    ManifestDiff {
        spec_changed: a.spec_hash != b.spec_hash,
        model_changed: a.model_hash != b.model_hash,
        optimizer_changed: a.optimizer_hash != b.optimizer_hash,
        weights_changed: a.initial_weights_hash != b.initial_weights_hash,
        step_count_a: a.step_count,
        step_count_b: b.step_count,
        roots_match: a.final_merkle_root == b.final_merkle_root,
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn merkle_state_deterministic() {
        let mut state1 = MerkleState::new("test spec");
        let mut state2 = MerkleState::new("test spec");

        let mut metrics = HashMap::new();
        metrics.insert("loss".into(), 2.5);
        metrics.insert("head.grad_norm".into(), 0.05);

        state1.update(0, &metrics, None);
        state2.update(0, &metrics, None);

        assert_eq!(state1.root_hex(), state2.root_hex());
        assert_eq!(state1.step_count(), 1);
    }

    #[test]
    fn different_metrics_different_root() {
        let mut state1 = MerkleState::new("test spec");
        let mut state2 = MerkleState::new("test spec");

        let mut metrics1 = HashMap::new();
        metrics1.insert("loss".into(), 2.5);

        let mut metrics2 = HashMap::new();
        metrics2.insert("loss".into(), 3.0);

        state1.update(0, &metrics1, None);
        state2.update(0, &metrics2, None);

        // With merkle feature, roots should differ
        // Without, they'll both be zero
        #[cfg(feature = "merkle")]
        assert_ne!(state1.root_hex(), state2.root_hex());
    }

    #[test]
    fn manifest_diff_detects_changes() {
        let m1 = RunManifest {
            spec_hash: "abc".into(),
            model_hash: "def".into(),
            optimizer_hash: "ghi".into(),
            initial_weights_hash: "jkl".into(),
            supervisor_version: "v1".into(),
            final_merkle_root: "root1".into(),
            step_count: 100,
        };

        let m2 = RunManifest {
            spec_hash: "abc".into(),
            model_hash: "def".into(),
            optimizer_hash: "CHANGED".into(),
            initial_weights_hash: "jkl".into(),
            supervisor_version: "v1".into(),
            final_merkle_root: "root2".into(),
            step_count: 200,
        };

        let d = diff(&m1, &m2);
        assert!(!d.spec_changed);
        assert!(!d.model_changed);
        assert!(d.optimizer_changed);
        assert!(!d.weights_changed);
        assert_eq!(d.step_count_a, 100);
        assert_eq!(d.step_count_b, 200);
        assert!(!d.roots_match);
    }

    #[test]
    fn fork_records_parent() {
        let parent = RunManifest {
            spec_hash: "parent_spec".into(),
            model_hash: "model".into(),
            optimizer_hash: "opt".into(),
            initial_weights_hash: "weights".into(),
            supervisor_version: "v1".into(),
            final_merkle_root: "root".into(),
            step_count: 500,
        };

        let child = fork(
            &parent,
            "modified spec yaml",
            vec!["changed learning rate".into(), "added dropout".into()],
        );

        assert_eq!(child.parent_manifest.step_count, 500);
        assert_eq!(child.changes.len(), 2);
    }
}
