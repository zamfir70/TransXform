//! TransXform V1.5 — Bootstrap Threshold Discovery
//!
//! Observation-only mode: collect metric statistics during early training,
//! then propose empirical thresholds for hard and soft invariants.
//!
//! # Design Principles
//!
//! - **Standalone.** Takes metric history as input, produces proposals as output.
//!   No supervisor internals, no Model trait dependency.
//! - **Conservative.** Proposes thresholds that would have permitted the observed
//!   training trajectory. Hard thresholds are at p01/p99 (extreme tails),
//!   soft thresholds at p05/p95.
//! - **Direction-aware.** Infers whether each metric is a "higher is bad" (Max)
//!   or "lower is bad" (Min) based on naming conventions and observed trends.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::types::{MetricSnapshot, ThresholdDirection};

/// A proposed threshold for a single metric key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdProposal {
    /// The full metric key, e.g. "backbone.pairwise_cosine".
    pub metric_key: String,
    /// The component name parsed from the key, or "global" for keys without a dot.
    pub component: String,
    /// Inferred direction: Min (floor) or Max (ceiling).
    pub direction: ThresholdDirection,
    /// Proposed hard threshold (p01/p99 with safety margin).
    pub proposed_hard: f64,
    /// Proposed soft threshold (p05/p95 with safety margin).
    pub proposed_soft: f64,
    /// Observed minimum value.
    pub observed_min: f64,
    /// Observed maximum value.
    pub observed_max: f64,
    /// Observed mean value.
    pub observed_mean: f64,
    /// Observed standard deviation.
    pub observed_std: f64,
    /// 1st percentile.
    pub p01: f64,
    /// 5th percentile.
    pub p05: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
    /// Number of observations.
    pub sample_count: usize,
}

/// The complete discovery report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryReport {
    /// The step at which this report was generated.
    pub step: u64,
    /// How many steps of observation data were available.
    pub observation_steps: u64,
    /// Per-metric threshold proposals.
    pub proposals: Vec<ThresholdProposal>,
    /// Step at which a significant distributional shift was detected in loss,
    /// suggesting the bootstrap phase may have ended naturally.
    pub phase_shift_detected_at: Option<u64>,
}

/// Configuration for discovery analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Minimum samples before proposing thresholds. Default: 50.
    #[serde(default = "default_min_samples")]
    pub min_samples: usize,
    /// Hard threshold percentile for Max-direction metrics (use p99). Default: 0.99.
    #[serde(default = "default_hard_percentile")]
    pub hard_percentile: f64,
    /// Soft threshold percentile for Max-direction metrics (use p95). Default: 0.95.
    #[serde(default = "default_soft_percentile")]
    pub soft_percentile: f64,
    /// Safety margin multiplied onto proposed thresholds. Default: 1.05 (5% headroom).
    #[serde(default = "default_safety_margin")]
    pub safety_margin: f64,
}

fn default_min_samples() -> usize { 50 }
fn default_hard_percentile() -> f64 { 0.99 }
fn default_soft_percentile() -> f64 { 0.95 }
fn default_safety_margin() -> f64 { 1.05 }

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            min_samples: 50,
            hard_percentile: 0.99,
            soft_percentile: 0.95,
            safety_margin: 1.05,
        }
    }
}

/// Analyze metric history and propose thresholds from empirical distributions.
///
/// Returns a [`DiscoveryReport`] containing per-metric proposals. Metrics with
/// fewer than `config.min_samples` observations are excluded.
pub fn analyze(
    history: &[MetricSnapshot],
    components: &[String],
    config: &DiscoveryConfig,
    step: u64,
) -> DiscoveryReport {
    let component_set: HashSet<&str> = components.iter().map(|s| s.as_str()).collect();
    let mut proposals = Vec::new();

    // Collect all unique metric keys across the history
    let mut all_keys: HashSet<String> = HashSet::new();
    for snapshot in history {
        for key in snapshot.keys() {
            all_keys.insert(key.clone());
        }
    }

    for key in &all_keys {
        // Extract component from "component.metric_name" format
        let component = if let Some(dot_pos) = key.find('.') {
            let comp = &key[..dot_pos];
            // Only include metrics from declared components or "global" for dotless keys
            if !component_set.contains(comp) {
                continue;
            }
            comp.to_string()
        } else {
            "global".to_string()
        };

        // Collect time series for this key
        let values: Vec<f64> = history
            .iter()
            .filter_map(|m| m.get(key).copied())
            .filter(|v| v.is_finite())
            .collect();

        if values.len() < config.min_samples {
            continue;
        }

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = values.len();
        let observed_min = sorted[0];
        let observed_max = sorted[n - 1];
        let observed_mean = values.iter().sum::<f64>() / n as f64;
        let observed_std = if n > 1 {
            let variance = values.iter().map(|v| (v - observed_mean).powi(2)).sum::<f64>()
                / (n - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let p01 = percentile(&sorted, 0.01);
        let p05 = percentile(&sorted, 0.05);
        let p95 = percentile(&sorted, 0.95);
        let p99 = percentile(&sorted, 0.99);

        let direction = infer_direction(key);

        let (proposed_hard, proposed_soft) = match direction {
            ThresholdDirection::Min => {
                // Floor: use low percentiles, divide by safety margin to widen
                let hard = p01 / config.safety_margin;
                let soft = p05 / config.safety_margin;
                (hard, soft)
            }
            ThresholdDirection::Max => {
                // Ceiling: use high percentiles, multiply by safety margin to widen
                let hard = p99 * config.safety_margin;
                let soft = p95 * config.safety_margin;
                (hard, soft)
            }
        };

        proposals.push(ThresholdProposal {
            metric_key: key.clone(),
            component,
            direction,
            proposed_hard,
            proposed_soft,
            observed_min,
            observed_max,
            observed_mean,
            observed_std,
            p01,
            p05,
            p95,
            p99,
            sample_count: n,
        });
    }

    // Sort proposals by component then metric key for deterministic output
    proposals.sort_by(|a, b| (&a.component, &a.metric_key).cmp(&(&b.component, &b.metric_key)));

    // Detect phase shift in loss trajectory
    let phase_shift = detect_phase_shift(history, "loss");

    DiscoveryReport {
        step,
        observation_steps: history.len() as u64,
        proposals,
        phase_shift_detected_at: phase_shift,
    }
}

/// Infer threshold direction from metric key name (same convention as
/// `InvariantMonitor::infer_direction()`).
fn infer_direction(metric_key: &str) -> ThresholdDirection {
    let lower = metric_key.to_lowercase();
    if lower.contains("min") || lower.contains("floor") || lower.contains("liveliness") {
        ThresholdDirection::Min
    } else {
        ThresholdDirection::Max
    }
}

/// Compute percentile from a sorted slice using linear interpolation.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let rank = p * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;

    if lower == upper || upper >= sorted.len() {
        sorted[lower.min(sorted.len() - 1)]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Detect if a significant distributional shift occurred in a metric's trajectory.
///
/// Splits history into first and second half, computes mean and std of each,
/// and flags if the shift exceeds 2 standard deviations.
fn detect_phase_shift(history: &[MetricSnapshot], key: &str) -> Option<u64> {
    let values: Vec<f64> = history
        .iter()
        .filter_map(|m| m.get(key).copied())
        .filter(|v| v.is_finite())
        .collect();

    if values.len() < 20 {
        return None;
    }

    let mid = values.len() / 2;
    let first = &values[..mid];
    let second = &values[mid..];

    let first_mean = first.iter().sum::<f64>() / first.len() as f64;
    let second_mean = second.iter().sum::<f64>() / second.len() as f64;

    let first_std = if first.len() > 1 {
        let var = first.iter().map(|v| (v - first_mean).powi(2)).sum::<f64>()
            / (first.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    let shift = (second_mean - first_mean).abs();

    // Shift exceeds 2 standard deviations of the first half.
    // If first_std is near zero (constant metric), use absolute shift > 10% of mean.
    let significant = if first_std > 1e-12 {
        shift > 2.0 * first_std
    } else {
        let abs_mean = first_mean.abs().max(1e-12);
        shift / abs_mean > 0.1
    };

    if significant {
        // Return the approximate step where the shift occurs (midpoint)
        Some(mid as u64)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(pairs: &[(&str, f64)]) -> MetricSnapshot {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn empty_history_produces_empty_proposals() {
        let report = analyze(&[], &["backbone".into()], &DiscoveryConfig::default(), 0);
        assert!(report.proposals.is_empty());
        assert_eq!(report.observation_steps, 0);
    }

    #[test]
    fn proposes_thresholds_from_stable_metrics() {
        let history: Vec<MetricSnapshot> = (0..100)
            .map(|_| make_snapshot(&[
                ("backbone.pairwise_cosine", 0.5),
                ("backbone.grad_norm_min", 0.01),
                ("loss", 2.0),
            ]))
            .collect();

        let report = analyze(
            &history,
            &["backbone".into()],
            &DiscoveryConfig::default(),
            100,
        );

        assert_eq!(report.observation_steps, 100);

        // Should have proposals for backbone metrics (not "loss" since it has no dot + component)
        let cosine = report.proposals.iter()
            .find(|p| p.metric_key == "backbone.pairwise_cosine")
            .expect("Should propose threshold for pairwise_cosine");

        assert_eq!(cosine.direction, ThresholdDirection::Max);
        assert!((cosine.observed_mean - 0.5).abs() < 1e-6);
        assert!((cosine.observed_std).abs() < 1e-6); // constant → zero std
        // Hard threshold = p99 * 1.05 ≈ 0.5 * 1.05 = 0.525
        assert!((cosine.proposed_hard - 0.525).abs() < 1e-6);

        let grad = report.proposals.iter()
            .find(|p| p.metric_key == "backbone.grad_norm_min")
            .expect("Should propose threshold for grad_norm_min");

        assert_eq!(grad.direction, ThresholdDirection::Min);
        // Hard threshold = p01 / 1.05 ≈ 0.01 / 1.05 ≈ 0.00952
        assert!((grad.proposed_hard - 0.01 / 1.05).abs() < 1e-6);
    }

    #[test]
    fn max_direction_inferred_for_cosine() {
        assert_eq!(infer_direction("head.pairwise_cosine"), ThresholdDirection::Max);
        assert_eq!(infer_direction("backbone.attention_entropy"), ThresholdDirection::Max);
    }

    #[test]
    fn min_direction_inferred_for_floor_metrics() {
        assert_eq!(infer_direction("backbone.activation_variance_min"), ThresholdDirection::Min);
        assert_eq!(infer_direction("head.grad_norm_min"), ThresholdDirection::Min);
        assert_eq!(infer_direction("encoder.liveliness"), ThresholdDirection::Min);
    }

    #[test]
    fn safety_margin_applied() {
        let history: Vec<MetricSnapshot> = (0..100)
            .map(|i| make_snapshot(&[
                ("backbone.pairwise_cosine", 0.4 + (i as f64 / 500.0)), // 0.4 to 0.598
            ]))
            .collect();

        let config = DiscoveryConfig {
            safety_margin: 1.10, // 10% margin
            ..Default::default()
        };

        let report = analyze(&history, &["backbone".into()], &config, 100);
        let cosine = report.proposals.iter()
            .find(|p| p.metric_key == "backbone.pairwise_cosine")
            .unwrap();

        // p99 should be near the max (~0.598), hard = p99 * 1.10
        assert!(cosine.proposed_hard > cosine.p99 * 1.09);
        assert!(cosine.proposed_hard < cosine.p99 * 1.11);
    }

    #[test]
    fn phase_shift_detected_when_loss_drops() {
        let mut history = Vec::new();

        // First 50 steps: loss = 3.0 (constant)
        for _ in 0..50 {
            history.push(make_snapshot(&[("loss", 3.0)]));
        }
        // Next 50 steps: loss = 1.5 (constant)
        for _ in 0..50 {
            history.push(make_snapshot(&[("loss", 1.5)]));
        }

        let report = analyze(&history, &[], &DiscoveryConfig::default(), 100);

        assert!(
            report.phase_shift_detected_at.is_some(),
            "Should detect phase shift when loss drops from 3.0 to 1.5"
        );
        // Shift should be detected near the midpoint
        let shift_step = report.phase_shift_detected_at.unwrap();
        assert_eq!(shift_step, 50);
    }

    #[test]
    fn ignores_metrics_below_min_samples() {
        // Only 10 snapshots — below default min_samples of 50
        let history: Vec<MetricSnapshot> = (0..10)
            .map(|_| make_snapshot(&[
                ("backbone.pairwise_cosine", 0.5),
            ]))
            .collect();

        let report = analyze(&history, &["backbone".into()], &DiscoveryConfig::default(), 10);
        assert!(
            report.proposals.is_empty(),
            "Should not propose thresholds with fewer than min_samples"
        );
    }
}
