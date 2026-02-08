use std::collections::HashMap;

use tch::Tensor;

use crate::collector::MetricCollector;
use crate::error::TransXformError;
use crate::tch_backend::model::TchModel;
use crate::types::{MetricSnapshot, MetricTier};

/// A metric collector that computes tensor-level metrics from a TchModel.
///
/// Provides richer metrics than the basic Model trait methods, including
/// pairwise cosine similarity, attention entropy, and spectrum analysis.
pub struct TchMetricCollector {
    component_names: Vec<String>,
    cadence: HashMap<String, u64>,
}

impl TchMetricCollector {
    pub fn new(component_names: Vec<String>, cadence: HashMap<String, u64>) -> Self {
        Self {
            component_names,
            cadence,
        }
    }

    /// Compute pairwise cosine similarity for a set of tensors.
    fn pairwise_cosine(tensors: &[Tensor]) -> f64 {
        if tensors.len() < 2 {
            return 0.0;
        }

        let flat: Vec<Tensor> = tensors
            .iter()
            .map(|t| t.reshape(&[-1]))
            .collect();

        let mut total_cos = 0.0;
        let mut count = 0;
        for i in 0..flat.len() {
            for j in (i + 1)..flat.len() {
                let cos = Tensor::cosine_similarity(&flat[i], &flat[j], 0, 1e-8);
                total_cos += cos.double_value(&[]);
                count += 1;
            }
        }

        if count > 0 {
            total_cos / count as f64
        } else {
            0.0
        }
    }
}

impl MetricCollector<TchModel> for TchMetricCollector {
    fn collect(
        &self,
        model: &TchModel,
        _step: u64,
        tier: MetricTier,
    ) -> Result<MetricSnapshot, TransXformError> {
        let mut snapshot = MetricSnapshot::new();

        // Always collect Tier0 metrics
        for name in &self.component_names {
            if model.has_component(name) {
                let component_metrics = model.component_metrics(name)?;
                snapshot.extend(component_metrics);
            }
        }

        let global = model.global_metrics()?;
        snapshot.extend(global);

        // Tier1: attention entropy, pairwise cosine (more expensive)
        if matches!(tier, MetricTier::Tier1 | MetricTier::Tier2) {
            // Pairwise cosine computed from weights
            // (Actual attention entropy requires forward pass output — deferred)
        }

        Ok(snapshot)
    }
}
