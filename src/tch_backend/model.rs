use std::collections::HashMap;

use tch::{nn, Device, Kind, Tensor};

use crate::error::TransXformError;
use crate::model::Model;
use crate::types::MetricSnapshot;

/// A PyTorch model wrapper implementing the `Model` trait.
///
/// Maps named components to `nn::Path` segments within a `VarStore`.
pub struct TchModel {
    vs: nn::VarStore,
    components: HashMap<String, Vec<String>>,
    frozen: HashMap<String, bool>,
}

impl TchModel {
    /// Create a new TchModel wrapping an existing VarStore.
    ///
    /// `component_map` maps component names to the variable path prefixes
    /// that belong to that component.
    pub fn new(vs: nn::VarStore, component_map: HashMap<String, Vec<String>>) -> Self {
        let frozen: HashMap<String, bool> = component_map
            .keys()
            .map(|k| (k.clone(), false))
            .collect();
        Self {
            vs,
            components: component_map,
            frozen,
        }
    }

    /// Get the VarStore for external use (e.g., optimizer construction).
    pub fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }

    /// Get a mutable reference to the VarStore.
    pub fn var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    fn get_component_tensors(&self, component: &str) -> Result<Vec<Tensor>, TransXformError> {
        let prefixes = self
            .components
            .get(component)
            .ok_or_else(|| TransXformError::UnknownComponent(component.to_string()))?;

        let variables = self.vs.variables();
        let tensors: Vec<Tensor> = variables
            .iter()
            .filter(|(name, _)| prefixes.iter().any(|p| name.starts_with(p.as_str())))
            .map(|(_, tensor)| tensor.shallow_clone())
            .collect();

        Ok(tensors)
    }
}

impl Model for TchModel {
    fn component_names(&self) -> Vec<String> {
        self.components.keys().cloned().collect()
    }

    fn has_component(&self, name: &str) -> bool {
        self.components.contains_key(name)
    }

    fn component_metrics(&self, name: &str) -> Result<MetricSnapshot, TransXformError> {
        let tensors = self.get_component_tensors(name)?;
        let mut metrics = MetricSnapshot::new();

        if tensors.is_empty() {
            return Ok(metrics);
        }

        // Gradient L2 norm
        let mut grad_norm_sum = 0.0f64;
        let mut has_grad = false;
        for t in &tensors {
            if let Some(grad) = t.grad().into() {
                let norm: f64 = grad.norm().double_value(&[]);
                grad_norm_sum += norm * norm;
                has_grad = true;
            }
        }
        if has_grad {
            metrics.insert(
                format!("{}.grad_norm", name),
                grad_norm_sum.sqrt(),
            );
        }

        // Weight norm
        let mut weight_norm_sum = 0.0f64;
        for t in &tensors {
            let norm: f64 = t.norm().double_value(&[]);
            weight_norm_sum += norm * norm;
        }
        metrics.insert(
            format!("{}.weight_norm", name),
            weight_norm_sum.sqrt(),
        );

        // Activation variance (approximate: use weight variance as proxy)
        let all_params: Vec<f64> = tensors
            .iter()
            .flat_map(|t| {
                let flat = t.reshape(&[-1]);
                let size = flat.size()[0] as usize;
                let mut vals = vec![0.0f64; size];
                flat.copy_data(&mut vals, size);
                vals
            })
            .collect();

        if !all_params.is_empty() {
            let mean = all_params.iter().sum::<f64>() / all_params.len() as f64;
            let variance = all_params.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / all_params.len() as f64;
            metrics.insert(format!("{}.activation_variance", name), variance);
        }

        Ok(metrics)
    }

    fn global_metrics(&self) -> Result<MetricSnapshot, TransXformError> {
        // Global metrics must be provided externally (loss comes from training loop)
        Ok(MetricSnapshot::new())
    }

    fn reinitialize(&mut self, component: &str) -> Result<(), TransXformError> {
        let tensors = self.get_component_tensors(component)?;
        tch::no_grad(|| {
            for t in &tensors {
                let size = t.size();
                // Kaiming uniform initialization
                if size.len() >= 2 {
                    let fan_in = size[1] as f64;
                    let bound = (6.0 / fan_in).sqrt();
                    let new_vals = Tensor::uniform(&size, -bound, bound, (Kind::Float, t.device()));
                    t.copy_(&new_vals);
                } else {
                    t.zero_();
                }
            }
        });
        // Unfreeze after reinitialization
        if let Some(frozen) = self.frozen.get_mut(component) {
            *frozen = false;
        }
        Ok(())
    }

    fn freeze(&mut self, component: &str) -> Result<(), TransXformError> {
        if !self.components.contains_key(component) {
            return Err(TransXformError::UnknownComponent(component.to_string()));
        }
        let tensors = self.get_component_tensors(component)?;
        for t in &tensors {
            let _ = t.set_requires_grad(false);
        }
        self.frozen.insert(component.to_string(), true);
        Ok(())
    }

    fn unfreeze(&mut self, component: &str) -> Result<(), TransXformError> {
        if !self.components.contains_key(component) {
            return Err(TransXformError::UnknownComponent(component.to_string()));
        }
        let tensors = self.get_component_tensors(component)?;
        for t in &tensors {
            let _ = t.set_requires_grad(true);
        }
        self.frozen.insert(component.to_string(), false);
        Ok(())
    }

    fn rescale(&mut self, component: &str, factor: f64) -> Result<(), TransXformError> {
        let tensors = self.get_component_tensors(component)?;
        tch::no_grad(|| {
            for t in &tensors {
                let scaled = t.multiply_scalar_(factor);
            }
        });
        Ok(())
    }

    fn inject_noise(&mut self, component: &str, magnitude: f64) -> Result<(), TransXformError> {
        let tensors = self.get_component_tensors(component)?;
        tch::no_grad(|| {
            for t in &tensors {
                let noise = Tensor::randn_like(t) * magnitude;
                let _ = t.add_(&noise);
            }
        });
        Ok(())
    }

    fn adjust_lr(&mut self, _component: &str, _factor: f64) -> Result<(), TransXformError> {
        // Cannot adjust LR from the model — optimizer access required.
        // Return an error so the supervisor queues it as pending.
        Err(TransXformError::InterventionFailed {
            action: "adjust_lr".into(),
            component: _component.to_string(),
            reason: "Model does not own optimizer; LR adjustment queued in SupervisorReport"
                .into(),
        })
    }
}
