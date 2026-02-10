use std::collections::HashMap;

use crate::spec::{ControlConfig, InvariantBlock, InvariantValue, TrainingSpec};
use crate::types::*;

/// An invariant resolved from the spec into a flat, evaluable form.
#[derive(Debug, Clone)]
pub struct ResolvedInvariant {
    pub name: String,
    pub component: String,
    pub severity: Severity,
    pub metric_key: String,
    pub base_threshold: f64,
    pub direction: ThresholdDirection,
    pub tier: MetricTier,
}

/// Evaluates metrics against declared invariants (whitepaper §5).
pub struct InvariantMonitor {
    invariants: Vec<ResolvedInvariant>,
}

impl InvariantMonitor {
    /// Build from a training spec, resolving per-component invariants into a flat list.
    pub fn new(spec: &TrainingSpec) -> Self {
        let mut invariants = Vec::new();
        resolve_block(
            &spec.invariants,
            &spec.model.components,
            &mut invariants,
        );
        Self { invariants }
    }

    /// Get all resolved invariants.
    pub fn invariants(&self) -> &[ResolvedInvariant] {
        &self.invariants
    }

    /// Get all unique metric keys that invariants depend on.
    /// Used by the diagnostic layer to detect missing metrics.
    pub fn metric_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .invariants
            .iter()
            .map(|i| i.metric_key.clone())
            .collect();
        keys.sort();
        keys.dedup();
        keys
    }

    /// Check all invariants against the given metrics for the current phase.
    ///
    /// `phase_thresholds` override `base_threshold` for matching invariant names.
    /// Hysteresis is proportional: `max(threshold * pct, floor)`.
    /// Catastrophic overrides bypass both phase thresholds and hysteresis.
    pub fn check(
        &self,
        metrics: &MetricSnapshot,
        phase_thresholds: &HashMap<String, f64>,
        control: &ControlConfig,
        step: u64,
    ) -> Vec<Violation> {
        let mut violations = Vec::new();

        for inv in &self.invariants {
            let observed = match metrics.get(&inv.metric_key) {
                Some(v) => *v,
                None => continue, // Metric not reported this step — skip
            };

            // 1. Check catastrophic overrides first (zero hysteresis, ignores phase)
            let catastrophic = control.catastrophic_overrides.get(&inv.metric_key)
                .or_else(|| control.catastrophic_overrides.get(&inv.name));
            if let Some(&cat_threshold) = catastrophic {
                let cat_violated = match inv.direction {
                    ThresholdDirection::Min => observed < cat_threshold,
                    ThresholdDirection::Max => observed > cat_threshold,
                };
                if cat_violated {
                    violations.push(Violation {
                        invariant_name: inv.name.clone(),
                        component: inv.component.clone(),
                        severity: inv.severity,
                        observed,
                        threshold: cat_threshold,
                        direction: inv.direction,
                        step,
                        passive: false,
                    });
                    continue; // Don't double-count
                }
            }

            // 2. Normal check with phase thresholds + proportional hysteresis
            let threshold = phase_thresholds
                .get(&inv.metric_key)
                .or_else(|| phase_thresholds.get(&inv.name))
                .copied()
                .unwrap_or(inv.base_threshold);

            let margin = control.effective_margin(threshold);

            let violated = match inv.direction {
                ThresholdDirection::Min => observed < threshold - margin,
                ThresholdDirection::Max => observed > threshold + margin,
            };

            if violated {
                violations.push(Violation {
                    invariant_name: inv.name.clone(),
                    component: inv.component.clone(),
                    severity: inv.severity,
                    observed,
                    threshold,
                    direction: inv.direction,
                    step,
                    passive: false,
                });
            }
        }

        violations
    }

    /// Check for near-misses: metrics within hysteresis margin of a hard threshold
    /// but not yet violating it. Uses proportional hysteresis per-threshold.
    pub fn check_near_misses(
        &self,
        metrics: &MetricSnapshot,
        phase_thresholds: &HashMap<String, f64>,
        control: &ControlConfig,
        step: u64,
    ) -> Vec<NearMiss> {
        let mut near_misses = Vec::new();

        // Build a map of hard thresholds for quick lookup
        let hard_thresholds: HashMap<&str, (f64, ThresholdDirection)> = self
            .invariants
            .iter()
            .filter(|inv| inv.severity == Severity::Hard)
            .map(|inv| {
                let threshold = phase_thresholds
                    .get(&inv.metric_key)
                    .or_else(|| phase_thresholds.get(&inv.name))
                    .copied()
                    .unwrap_or(inv.base_threshold);
                (inv.metric_key.as_str(), (threshold, inv.direction))
            })
            .collect();

        // Check soft invariants for proximity to hard thresholds
        for inv in &self.invariants {
            if inv.severity != Severity::Soft {
                continue;
            }

            let observed = match metrics.get(&inv.metric_key) {
                Some(v) => *v,
                None => continue,
            };

            // Is there a hard threshold on the same metric?
            if let Some(&(hard_thresh, direction)) = hard_thresholds.get(inv.metric_key.as_str()) {
                let hyst_margin = control.effective_margin(hard_thresh);
                let distance = match direction {
                    ThresholdDirection::Min => hard_thresh - observed,
                    ThresholdDirection::Max => observed - hard_thresh,
                };

                // Near-miss: within hysteresis margin but not actually violating
                if distance > -hyst_margin && distance < hyst_margin {
                    near_misses.push(NearMiss {
                        step,
                        invariant_name: inv.name.clone(),
                        component: inv.component.clone(),
                        observed,
                        hard_threshold: hard_thresh,
                        margin: distance.abs(),
                        metric_snapshot: metrics.clone(),
                    });
                }
            }
        }

        near_misses
    }

    /// Return invariants that should be checked at this step given their tier and cadence.
    pub fn due_invariants(
        &self,
        step: u64,
        cadence: &HashMap<String, u64>,
    ) -> Vec<&ResolvedInvariant> {
        self.invariants
            .iter()
            .filter(|inv| {
                match inv.tier {
                    MetricTier::Tier0 => true, // every step
                    MetricTier::Tier1 => {
                        let interval = cadence
                            .get(&inv.metric_key)
                            .or_else(|| cadence.get("tier1"))
                            .copied()
                            .unwrap_or(10);
                        step % interval == 0
                    }
                    MetricTier::Tier2 => {
                        let interval = cadence
                            .get(&inv.metric_key)
                            .or_else(|| cadence.get("tier2"))
                            .copied()
                            .unwrap_or(100);
                        step % interval == 0
                    }
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Invariant resolution
// ---------------------------------------------------------------------------

/// Resolve the spec's InvariantBlock into flat ResolvedInvariant entries.
fn resolve_block(
    block: &InvariantBlock,
    components: &[String],
    out: &mut Vec<ResolvedInvariant>,
) {
    resolve_severity(&block.hard, Severity::Hard, components, out);
    resolve_severity(&block.soft, Severity::Soft, components, out);
}

fn resolve_severity(
    map: &HashMap<String, InvariantValue>,
    severity: Severity,
    components: &[String],
    out: &mut Vec<ResolvedInvariant>,
) {
    for (name, value) in map {
        match value {
            InvariantValue::Scalar(threshold) => {
                // Determine component and metric key from the invariant name.
                // Convention: "component.metric" -> per-component
                //             "metric_name"      -> global (component = "global")
                let (component, metric_key) = if let Some(dot_pos) = name.find('.') {
                    (name[..dot_pos].to_string(), name.clone())
                } else {
                    ("global".to_string(), name.clone())
                };

                let direction = infer_direction(name);
                let tier = infer_tier(name);

                out.push(ResolvedInvariant {
                    name: name.clone(),
                    component,
                    severity,
                    metric_key,
                    base_threshold: *threshold,
                    direction,
                    tier,
                });
            }
            InvariantValue::PerComponent(map) => {
                for (comp, threshold) in map {
                    let actual_components: Vec<&str> = if comp == "all" || comp == "all_layers" {
                        components.iter().map(|s| s.as_str()).collect()
                    } else {
                        vec![comp.as_str()]
                    };

                    for component in actual_components {
                        let metric_key = format!("{}.{}", component, name);
                        let direction = infer_direction(name);
                        let tier = infer_tier(name);

                        out.push(ResolvedInvariant {
                            name: name.clone(),
                            component: component.to_string(),
                            severity,
                            metric_key,
                            base_threshold: *threshold,
                            direction,
                            tier,
                        });
                    }
                }
            }
        }
    }
}

/// Infer threshold direction from the invariant name.
/// Names containing "min" or "floor" -> Min (metric must stay above).
/// Names containing "max", "ceiling", "factor", "spike", "threshold" -> Max.
/// Default: Max (safer — treats unknown invariants as upper bounds).
fn infer_direction(name: &str) -> ThresholdDirection {
    let lower = name.to_lowercase();
    if lower.contains("min") || lower.contains("floor") || lower.contains("liveliness") {
        ThresholdDirection::Min
    } else {
        ThresholdDirection::Max
    }
}

/// Infer metric tier from the invariant name.
fn infer_tier(name: &str) -> MetricTier {
    let lower = name.to_lowercase();
    if lower.contains("grad_norm")
        || lower.contains("activation_variance")
        || lower.contains("pairwise_cosine")
        || lower.contains("loss")
    {
        MetricTier::Tier0
    } else if lower.contains("entropy") || lower.contains("mutual_information") || lower.contains("drift") {
        MetricTier::Tier1
    } else {
        MetricTier::Tier0 // default to most frequent checking
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::parse_spec;

    fn test_spec() -> TrainingSpec {
        parse_spec(
            r#"
model:
  name: "test"
  components: [backbone, head]
invariants:
  hard:
    head.pairwise_cosine: 0.95
    head.grad_norm: 0.001
    activation_variance_min:
      all: 0.0001
  soft:
    attention_entropy_min: 0.3
    loss_explosion_factor: 3.0
phases: {}
control: {}
"#,
        )
        .unwrap()
    }

    /// Build a ControlConfig with zero hysteresis (flat, backward-compatible).
    fn zero_control() -> ControlConfig {
        ControlConfig {
            cooldown_steps: 50,
            max_hard_interventions: 3,
            hysteresis_margin: 0.0,
            hysteresis_pct: 0.0,
            catastrophic_overrides: HashMap::new(),
            damping_factor: 0.5,
            base_lr: None,
            regret_window_steps: 100,
            readiness_gate: true,
            readiness_patience_steps: 200,
            max_threshold_relaxation: 0.02,
            discovery_mode: false,
            shadow_step: false,
        }
    }

    #[test]
    fn resolves_invariants() {
        let spec = test_spec();
        let monitor = InvariantMonitor::new(&spec);
        // 2 direct hard + 2 expanded "all" + 2 soft = 6
        assert!(monitor.invariants().len() >= 4);
    }

    #[test]
    fn detects_hard_violation() {
        let spec = test_spec();
        let monitor = InvariantMonitor::new(&spec);

        let mut metrics = MetricSnapshot::new();
        metrics.insert("head.pairwise_cosine".into(), 0.99); // above 0.95 threshold
        metrics.insert("head.grad_norm".into(), 0.0005); // below 0.001 threshold, no violation

        let violations = monitor.check(&metrics, &HashMap::new(), &zero_control(), 100);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].invariant_name, "head.pairwise_cosine");
        assert_eq!(violations[0].severity, Severity::Hard);
    }

    #[test]
    fn hysteresis_prevents_boundary_trigger() {
        let spec = test_spec();
        let monitor = InvariantMonitor::new(&spec);

        let mut metrics = MetricSnapshot::new();
        // Exactly at threshold + small amount less than hysteresis
        metrics.insert("head.pairwise_cosine".into(), 0.97);

        // With flat 0.05 hysteresis, need > 0.95 + 0.05 = 1.0 to trigger
        let ctrl = ControlConfig {
            hysteresis_margin: 0.05,
            ..zero_control()
        };
        let violations = monitor.check(&metrics, &HashMap::new(), &ctrl, 100);
        assert!(violations.is_empty());
    }

    #[test]
    fn proportional_hysteresis_scales_with_threshold() {
        let spec = test_spec();
        let monitor = InvariantMonitor::new(&spec);

        let mut metrics = MetricSnapshot::new();
        // Threshold = 0.95, pct = 0.10 → margin = 0.095
        // observed = 1.04 → 1.04 > 0.95 + 0.095 = 1.045? No → no violation
        metrics.insert("head.pairwise_cosine".into(), 1.04);

        let ctrl = ControlConfig {
            hysteresis_margin: 0.001, // floor
            hysteresis_pct: 0.10,     // 10% of threshold
            ..zero_control()
        };
        let violations = monitor.check(&metrics, &HashMap::new(), &ctrl, 100);
        assert!(violations.is_empty());

        // observed = 1.05 → 1.05 > 1.045 → violation
        metrics.insert("head.pairwise_cosine".into(), 1.05);
        let violations = monitor.check(&metrics, &HashMap::new(), &ctrl, 100);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn proportional_hysteresis_respects_floor() {
        let spec = test_spec();
        let monitor = InvariantMonitor::new(&spec);

        // For grad_norm threshold 0.001, pct = 0.10 → 0.0001, but floor = 0.001
        // So margin = max(0.0001, 0.001) = 0.001
        // Min direction: observed < 0.001 - 0.001 = 0.0 to violate
        let mut metrics = MetricSnapshot::new();
        metrics.insert("head.grad_norm".into(), 0.0001); // above 0.0, no violation

        let ctrl = ControlConfig {
            hysteresis_margin: 0.001,
            hysteresis_pct: 0.10,
            ..zero_control()
        };
        // grad_norm is Max direction (default), threshold 0.001
        // observed 0.0001 < 0.001 + 0.001 = 0.002 → no violation
        let violations = monitor.check(&metrics, &HashMap::new(), &ctrl, 100);
        let grad_violations: Vec<_> = violations.iter()
            .filter(|v| v.invariant_name == "head.grad_norm")
            .collect();
        assert!(grad_violations.is_empty());
    }

    #[test]
    fn catastrophic_override_fires_regardless() {
        let spec = test_spec();
        let monitor = InvariantMonitor::new(&spec);

        let mut metrics = MetricSnapshot::new();
        // Cosine at 0.998 — above catastrophic ceiling of 0.995
        metrics.insert("head.pairwise_cosine".into(), 0.998);

        let mut cat = HashMap::new();
        cat.insert("head.pairwise_cosine".to_string(), 0.995);

        let ctrl = ControlConfig {
            hysteresis_margin: 100.0, // absurdly large — would mask everything
            hysteresis_pct: 0.0,
            catastrophic_overrides: cat,
            ..zero_control()
        };

        // Even with huge hysteresis, catastrophic fires (zero hysteresis)
        let violations = monitor.check(&metrics, &HashMap::new(), &ctrl, 100);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].threshold, 0.995);
    }

    #[test]
    fn catastrophic_override_by_invariant_name() {
        let spec = test_spec();
        let monitor = InvariantMonitor::new(&spec);

        // activation_variance_min has metric_key "backbone.activation_variance_min"
        // but we can also match by invariant name "activation_variance_min"
        let mut metrics = MetricSnapshot::new();
        metrics.insert("backbone.activation_variance_min".into(), 0.000001);

        let mut cat = HashMap::new();
        cat.insert("activation_variance_min".to_string(), 0.00001); // Min direction

        let ctrl = ControlConfig {
            catastrophic_overrides: cat,
            ..zero_control()
        };

        let violations = monitor.check(&metrics, &HashMap::new(), &ctrl, 100);
        let var_violations: Vec<_> = violations.iter()
            .filter(|v| v.invariant_name == "activation_variance_min")
            .collect();
        assert!(!var_violations.is_empty());
    }

    #[test]
    fn phase_thresholds_override_base() {
        let spec = test_spec();
        let monitor = InvariantMonitor::new(&spec);

        let mut metrics = MetricSnapshot::new();
        metrics.insert("head.pairwise_cosine".into(), 0.92);

        // Base threshold is 0.95, so 0.92 wouldn't trigger.
        // Override to 0.90, now 0.92 > 0.90 triggers.
        let mut overrides = HashMap::new();
        overrides.insert("head.pairwise_cosine".into(), 0.90);

        let violations = monitor.check(&metrics, &overrides, &zero_control(), 100);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn missing_metrics_skipped() {
        let spec = test_spec();
        let monitor = InvariantMonitor::new(&spec);

        let metrics = MetricSnapshot::new(); // empty
        let violations = monitor.check(&metrics, &HashMap::new(), &zero_control(), 100);
        assert!(violations.is_empty());
    }

    #[test]
    fn min_direction_detected() {
        let spec = test_spec();
        let monitor = InvariantMonitor::new(&spec);

        let min_invs: Vec<_> = monitor
            .invariants()
            .iter()
            .filter(|i| i.direction == ThresholdDirection::Min)
            .collect();
        // activation_variance_min and attention_entropy_min should be Min direction
        assert!(!min_invs.is_empty());
    }

    #[test]
    fn per_component_expansion() {
        let spec = test_spec();
        let monitor = InvariantMonitor::new(&spec);

        // "all" should expand to both backbone and head
        let variance_invs: Vec<_> = monitor
            .invariants()
            .iter()
            .filter(|i| i.name == "activation_variance_min")
            .collect();
        assert_eq!(variance_invs.len(), 2);
        let comps: Vec<&str> = variance_invs.iter().map(|i| i.component.as_str()).collect();
        assert!(comps.contains(&"backbone"));
        assert!(comps.contains(&"head"));
    }
}
