use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::*;

/// A known failure mode with detection pattern and fix (whitepaper §11).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSignature {
    pub id: String,
    pub name: String,
    pub description: String,
    pub detection: DetectionPattern,
    pub proven_fix: Action,
    pub architecture_tags: Vec<String>,
    pub version: String,
}

/// How to detect a failure signature from metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionPattern {
    pub conditions: Vec<MetricCondition>,
    #[serde(default)]
    pub sustained_steps: Option<u64>,
}

/// A single condition within a detection pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricCondition {
    pub metric_key: String,
    pub direction: ThresholdDirection,
    pub threshold: f64,
}

/// Registry of known failure signatures.
pub struct SignatureRegistry {
    signatures: Vec<FailureSignature>,
    /// Per-signature counter: consecutive steps where all conditions are met.
    match_state: HashMap<String, u64>,
}

impl SignatureRegistry {
    pub fn new() -> Self {
        Self {
            signatures: Vec::new(),
            match_state: HashMap::new(),
        }
    }

    /// Load the default built-in signatures (SIG-001 through SIG-007).
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        for sig in default_signatures() {
            registry.register(sig);
        }
        registry
    }

    /// Register a failure signature.
    pub fn register(&mut self, signature: FailureSignature) {
        self.match_state.insert(signature.id.clone(), 0);
        self.signatures.push(signature);
    }

    /// Check current metrics against all registered signatures.
    /// Returns matched signatures (those that have sustained for required steps).
    pub fn check(&mut self, metrics: &MetricSnapshot, _step: u64) -> Vec<&FailureSignature> {
        let mut matched = Vec::new();

        for sig in &self.signatures {
            let all_conditions_met = sig.detection.conditions.iter().all(|cond| {
                if let Some(&value) = metrics.get(&cond.metric_key) {
                    match cond.direction {
                        ThresholdDirection::Min => value < cond.threshold,
                        ThresholdDirection::Max => value > cond.threshold,
                    }
                } else {
                    false
                }
            });

            let count = self.match_state.entry(sig.id.clone()).or_insert(0);
            if all_conditions_met {
                *count += 1;
            } else {
                *count = 0;
            }

            let required = sig.detection.sustained_steps.unwrap_or(1);
            if *count >= required {
                matched.push(sig);
            }
        }

        // Borrow checker: re-collect from self.signatures based on matched IDs
        // (we need to return references to self.signatures, not to the loop variable)
        let matched_ids: Vec<String> = matched.iter().map(|s| s.id.clone()).collect();
        self.signatures
            .iter()
            .filter(|s| matched_ids.contains(&s.id))
            .collect()
    }

    /// Get a signature by ID.
    pub fn get(&self, id: &str) -> Option<&FailureSignature> {
        self.signatures.iter().find(|s| s.id == id)
    }

    /// Get all registered signatures.
    pub fn signatures(&self) -> &[FailureSignature] {
        &self.signatures
    }
}

impl Default for SignatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Default signatures from whitepaper §11.1
// ---------------------------------------------------------------------------

fn default_signatures() -> Vec<FailureSignature> {
    vec![
        FailureSignature {
            id: "SIG-001".into(),
            name: "Stability loss trivial convergence".into(),
            description: "Stability loss drops but representation variance collapses".into(),
            detection: DetectionPattern {
                conditions: vec![MetricCondition {
                    metric_key: "backbone.activation_variance".into(),
                    direction: ThresholdDirection::Min,
                    threshold: 1e-5,
                }],
                sustained_steps: Some(50),
            },
            proven_fix: Action::Abort {
                reason: "Reformulate stability loss".into(),
            },
            architecture_tags: vec!["recurrent_transformer".into()],
            version: "1.0".into(),
        },
        FailureSignature {
            id: "SIG-002".into(),
            name: "Compressor attention collapse".into(),
            description: "Compressor outputs converge to single direction".into(),
            detection: DetectionPattern {
                conditions: vec![MetricCondition {
                    metric_key: "compressor.pairwise_cosine".into(),
                    direction: ThresholdDirection::Max,
                    threshold: 0.7,
                }],
                sustained_steps: Some(100),
            },
            proven_fix: Action::Rescale {
                component: "compressor".into(),
                factor: 1.0,
            },
            architecture_tags: vec!["recurrent_transformer".into()],
            version: "1.0".into(),
        },
        FailureSignature {
            id: "SIG-003".into(),
            name: "Multi-task gradient conflict".into(),
            description: "One loss explodes on step 2 of training".into(),
            detection: DetectionPattern {
                conditions: vec![MetricCondition {
                    metric_key: "loss".into(),
                    direction: ThresholdDirection::Max,
                    threshold: 100.0,
                }],
                sustained_steps: Some(2),
            },
            proven_fix: Action::AdjustLr {
                component: "global".into(),
                factor: 0.1,
            },
            architecture_tags: vec!["multi_task".into()],
            version: "1.0".into(),
        },
        FailureSignature {
            id: "SIG-004".into(),
            name: "Recurrence erasing signal".into(),
            description: "H0→Hfinal cosine too high, recurrence overwrites input".into(),
            detection: DetectionPattern {
                conditions: vec![MetricCondition {
                    metric_key: "backbone.h0_hfinal_cosine".into(),
                    direction: ThresholdDirection::Max,
                    threshold: 0.5,
                }],
                sustained_steps: Some(50),
            },
            proven_fix: Action::Rescale {
                component: "backbone".into(),
                factor: 0.5,
            },
            architecture_tags: vec!["recurrent_transformer".into()],
            version: "1.0".into(),
        },
        FailureSignature {
            id: "SIG-005".into(),
            name: "Emission head frozen".into(),
            description: "Emission cosine ~1.0, contrastive gradient ~0".into(),
            detection: DetectionPattern {
                conditions: vec![
                    MetricCondition {
                        metric_key: "emission_head.pairwise_cosine".into(),
                        direction: ThresholdDirection::Max,
                        threshold: 0.99,
                    },
                    MetricCondition {
                        metric_key: "emission_head.grad_norm".into(),
                        direction: ThresholdDirection::Min,
                        threshold: 1e-4,
                    },
                ],
                sustained_steps: Some(30),
            },
            proven_fix: Action::Reinitialize {
                component: "emission_head".into(),
            },
            architecture_tags: vec!["recurrent_transformer".into()],
            version: "1.0".into(),
        },
        FailureSignature {
            id: "SIG-006".into(),
            name: "Dead attention heads".into(),
            description: "Attention entropy and head gradient norm at zero".into(),
            detection: DetectionPattern {
                conditions: vec![MetricCondition {
                    metric_key: "attention_head.grad_norm".into(),
                    direction: ThresholdDirection::Min,
                    threshold: 1e-6,
                }],
                sustained_steps: Some(50),
            },
            proven_fix: Action::Reinitialize {
                component: "attention_head".into(),
            },
            architecture_tags: vec!["transformer".into()],
            version: "1.0".into(),
        },
        FailureSignature {
            id: "SIG-007".into(),
            name: "Loss explosion".into(),
            description: "Loss exceeds 3x baseline within a few steps".into(),
            detection: DetectionPattern {
                conditions: vec![MetricCondition {
                    metric_key: "loss_delta_ratio".into(),
                    direction: ThresholdDirection::Max,
                    threshold: 3.0,
                }],
                sustained_steps: Some(1),
            },
            proven_fix: Action::AdjustLr {
                component: "global".into(),
                factor: 0.1,
            },
            architecture_tags: vec!["transformer".into()],
            version: "1.0".into(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_defaults() {
        let registry = SignatureRegistry::with_defaults();
        assert_eq!(registry.signatures().len(), 7);
        assert!(registry.get("SIG-001").is_some());
        assert!(registry.get("SIG-007").is_some());
    }

    #[test]
    fn matches_sig005_emission_frozen() {
        let mut registry = SignatureRegistry::with_defaults();

        let mut metrics = MetricSnapshot::new();
        metrics.insert("emission_head.pairwise_cosine".into(), 0.999);
        metrics.insert("emission_head.grad_norm".into(), 1e-5);

        // Need 30 sustained steps
        for step in 0..29 {
            let matched = registry.check(&metrics, step);
            assert!(matched.is_empty(), "Should not match before 30 steps");
        }

        let matched = registry.check(&metrics, 29);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].id, "SIG-005");
    }

    #[test]
    fn no_match_healthy_metrics() {
        let mut registry = SignatureRegistry::with_defaults();

        let mut metrics = MetricSnapshot::new();
        metrics.insert("emission_head.pairwise_cosine".into(), 0.5);
        metrics.insert("emission_head.grad_norm".into(), 0.05);
        metrics.insert("backbone.activation_variance".into(), 0.01);
        metrics.insert("loss".into(), 2.0);

        for step in 0..100 {
            let matched = registry.check(&metrics, step);
            assert!(matched.is_empty());
        }
    }

    #[test]
    fn sustained_counter_resets() {
        let mut registry = SignatureRegistry::with_defaults();

        let mut bad_metrics = MetricSnapshot::new();
        bad_metrics.insert("emission_head.pairwise_cosine".into(), 0.999);
        bad_metrics.insert("emission_head.grad_norm".into(), 1e-5);

        let mut good_metrics = MetricSnapshot::new();
        good_metrics.insert("emission_head.pairwise_cosine".into(), 0.5);
        good_metrics.insert("emission_head.grad_norm".into(), 0.05);

        // 20 bad steps
        for step in 0..20 {
            registry.check(&bad_metrics, step);
        }

        // 1 good step resets counter
        registry.check(&good_metrics, 20);

        // 20 more bad steps — not enough for 30 sustained
        for step in 21..41 {
            let matched = registry.check(&bad_metrics, step);
            if step < 50 {
                // Counter was reset, so need 30 more from step 21
                // At step 50 (30 steps after reset), it should match
            }
            assert!(matched.is_empty() || step >= 50);
        }
    }
}
