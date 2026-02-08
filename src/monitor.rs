use std::collections::HashMap;

use crate::spec::{InvariantBlock, InvariantValue, TrainingSpec};
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

    /// Check all invariants against the given metrics for the current phase.
    ///
    /// `phase_thresholds` override `base_threshold` for matching invariant names.
    /// `hysteresis_margin` prevents triggering on boundary touches.
    pub fn check(
        &self,
        metrics: &MetricSnapshot,
        phase_thresholds: &HashMap<String, f64>,
        hysteresis_margin: f64,
        step: u64,
    ) -> Vec<Violation> {
        let mut violations = Vec::new();

        for inv in &self.invariants {
            let threshold = phase_thresholds
                .get(&inv.metric_key)
                .or_else(|| phase_thresholds.get(&inv.name))
                .copied()
                .unwrap_or(inv.base_threshold);

            let observed = match metrics.get(&inv.metric_key) {
                Some(v) => *v,
                None => continue, // Metric not reported this step — skip
            };

            let violated = match inv.direction {
                ThresholdDirection::Min => observed < threshold - hysteresis_margin,
                ThresholdDirection::Max => observed > threshold + hysteresis_margin,
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
                });
            }
        }

        violations
    }

    /// Check for near-misses: metrics within hysteresis margin of a hard threshold
    /// but not yet violating it.
    pub fn check_near_misses(
        &self,
        metrics: &MetricSnapshot,
        phase_thresholds: &HashMap<String, f64>,
        hysteresis_margin: f64,
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
                let margin = match direction {
                    ThresholdDirection::Min => hard_thresh - observed,
                    ThresholdDirection::Max => observed - hard_thresh,
                };

                // Near-miss: within hysteresis margin but not actually violating
                if margin > -hysteresis_margin && margin < hysteresis_margin {
                    near_misses.push(NearMiss {
                        step,
                        invariant_name: inv.name.clone(),
                        component: inv.component.clone(),
                        observed,
                        hard_threshold: hard_thresh,
                        margin: margin.abs(),
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

        let violations = monitor.check(&metrics, &HashMap::new(), 0.0, 100);
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

        // With 0.05 hysteresis, need > 0.95 + 0.05 = 1.0 to trigger
        let violations = monitor.check(&metrics, &HashMap::new(), 0.05, 100);
        assert!(violations.is_empty());
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

        let violations = monitor.check(&metrics, &overrides, 0.0, 100);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn missing_metrics_skipped() {
        let spec = test_spec();
        let monitor = InvariantMonitor::new(&spec);

        let metrics = MetricSnapshot::new(); // empty
        let violations = monitor.check(&metrics, &HashMap::new(), 0.0, 100);
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
