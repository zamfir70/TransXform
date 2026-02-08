use std::collections::HashMap;

use crate::error::TransXformError;
use crate::model::Model;
use crate::types::{MetricSnapshot, MetricTier};

/// Trait for collecting metrics from a model (whitepaper §3.2).
pub trait MetricCollector<M: Model> {
    /// Collect metrics from the model for the current step.
    fn collect(
        &self,
        model: &M,
        step: u64,
        tier: MetricTier,
    ) -> Result<MetricSnapshot, TransXformError>;
}

/// A basic metric collector that delegates to `Model::component_metrics`
/// and `Model::global_metrics`. Works with any Model implementation.
pub struct BasicMetricCollector {
    component_names: Vec<String>,
    cadence: HashMap<String, u64>,
}

impl BasicMetricCollector {
    pub fn new(component_names: Vec<String>, cadence: HashMap<String, u64>) -> Self {
        Self {
            component_names,
            cadence,
        }
    }

    /// Determine which tier of metrics to collect at this step.
    pub fn tier_for_step(&self, step: u64) -> MetricTier {
        let tier2_interval = self.cadence.get("tier2").copied().unwrap_or(100);
        let tier1_interval = self.cadence.get("tier1").copied().unwrap_or(10);

        if step % tier2_interval == 0 {
            MetricTier::Tier2
        } else if step % tier1_interval == 0 {
            MetricTier::Tier1
        } else {
            MetricTier::Tier0
        }
    }
}

impl<M: Model> MetricCollector<M> for BasicMetricCollector {
    fn collect(
        &self,
        model: &M,
        _step: u64,
        _tier: MetricTier,
    ) -> Result<MetricSnapshot, TransXformError> {
        let mut snapshot = MetricSnapshot::new();

        // Collect per-component metrics
        for name in &self.component_names {
            if model.has_component(name) {
                let component_metrics = model.component_metrics(name)?;
                snapshot.extend(component_metrics);
            }
        }

        // Collect global metrics
        let global = model.global_metrics()?;
        snapshot.extend(global);

        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MockModel;

    #[test]
    fn basic_collector_gathers_all_metrics() {
        let mut model = MockModel::new(&["backbone", "head"]);
        model.set_metric("backbone", "grad_norm", 0.05);
        model.set_metric("head", "grad_norm", 0.01);
        model.set_global_metric("loss", 2.5);

        let collector = BasicMetricCollector::new(
            vec!["backbone".into(), "head".into()],
            HashMap::new(),
        );

        let snapshot = collector.collect(&model, 0, MetricTier::Tier0).unwrap();
        assert!((snapshot["backbone.grad_norm"] - 0.05).abs() < f64::EPSILON);
        assert!((snapshot["head.grad_norm"] - 0.01).abs() < f64::EPSILON);
        assert!((snapshot["loss"] - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn tier_selection() {
        let collector = BasicMetricCollector::new(vec![], HashMap::new());
        assert_eq!(collector.tier_for_step(0), MetricTier::Tier2); // 0 % 100 == 0
        assert_eq!(collector.tier_for_step(1), MetricTier::Tier0);
        assert_eq!(collector.tier_for_step(10), MetricTier::Tier1);
        assert_eq!(collector.tier_for_step(50), MetricTier::Tier1);
        assert_eq!(collector.tier_for_step(100), MetricTier::Tier2);
    }
}
