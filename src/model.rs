use std::collections::HashMap;

use crate::error::TransXformError;
use crate::types::MetricSnapshot;

/// The central abstraction bridging TransXform to any ML framework.
///
/// # Design Principles
///
/// 1. **No tensor types in the signature.** Metrics are `f64`, component access
///    is by string name. The implementation decides how to compute metrics from
///    its internal tensor representation.
///
/// 2. **Pull-based metrics.** The supervisor asks for metrics; the model provides
///    them. This avoids requiring framework-specific hook registration in core.
///
/// 3. **Fallible interventions.** Not all models support all interventions.
///    `adjust_lr` in particular may not be possible without optimizer access.
///
/// 4. **Component-oriented.** Every method takes a component name. The model
///    implementation maps names to its internal module/parameter structure.
///
/// # Implementing for a new framework
///
/// 1. Create a new feature in Cargo.toml
/// 2. Implement `Model` for your framework's model type
/// 3. Implement `MetricCollector<YourModel>` for tensor-level metric computation
/// 4. Place implementations in `src/{framework}/`
pub trait Model {
    /// List all named components in the model.
    fn component_names(&self) -> Vec<String>;

    /// Check if a named component exists.
    fn has_component(&self, name: &str) -> bool {
        self.component_names().iter().any(|n| n == name)
    }

    /// Collect metrics for a specific component.
    ///
    /// Must return at minimum:
    /// - `"{component}.grad_norm"` — gradient L2 norm
    /// - `"{component}.activation_variance"` — activation variance
    ///
    /// May return additional framework-specific metrics.
    fn component_metrics(&self, name: &str) -> Result<MetricSnapshot, TransXformError>;

    /// Collect global metrics (loss values, overall statistics).
    fn global_metrics(&self) -> Result<MetricSnapshot, TransXformError>;

    /// Reinitialize a component's parameters to break degenerate states.
    fn reinitialize(&mut self, component: &str) -> Result<(), TransXformError>;

    /// Freeze a component (stop gradient flow).
    fn freeze(&mut self, component: &str) -> Result<(), TransXformError>;

    /// Unfreeze a component (resume gradient flow).
    fn unfreeze(&mut self, component: &str) -> Result<(), TransXformError>;

    /// Rescale a component's weights/activations by a factor.
    fn rescale(&mut self, component: &str, factor: f64) -> Result<(), TransXformError>;

    /// Inject noise into a component's parameters to break symmetry.
    fn inject_noise(&mut self, component: &str, magnitude: f64) -> Result<(), TransXformError>;

    /// Adjust learning rate for a component.
    ///
    /// Implementations that do not have optimizer access should return
    /// `Err(TransXformError::InterventionFailed)` with a descriptive message.
    /// The supervisor will include the adjustment in
    /// `SupervisorReport.pending_lr_adjustments` for the training loop to apply.
    fn adjust_lr(&mut self, component: &str, factor: f64) -> Result<(), TransXformError>;

    /// Reset optimizer state (momentum, variance accumulators) for a component.
    ///
    /// Called automatically after [`reinitialize`](Model::reinitialize) to prevent
    /// momentum corpses — Adam's exponential moving averages remember the
    /// pre-reinit gradient trajectory for ~β₂⁻¹ steps (typically ~999), causing
    /// the freshly initialized parameters to drift back toward the degenerate
    /// state (whitepaper §2.1).
    ///
    /// The default implementation is a no-op for backward compatibility.
    /// Implementations with optimizer access should zero the first-moment (m)
    /// and second-moment (v) accumulators for the named component's parameters.
    fn reset_optimizer_state(&mut self, component: &str) -> Result<(), TransXformError> {
        let _ = component;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MockModel — always available for testing
// ---------------------------------------------------------------------------

/// A mock model for testing. Stores metrics in-memory and logs interventions.
#[derive(Debug, Clone)]
pub struct MockModel {
    components: HashMap<String, MockComponent>,
    global: MetricSnapshot,
    interventions: Vec<(String, String)>, // (component, action_name)
}

#[derive(Debug, Clone)]
struct MockComponent {
    metrics: MetricSnapshot,
    frozen: bool,
}

impl MockModel {
    pub fn new(component_names: &[&str]) -> Self {
        let mut components = HashMap::new();
        for name in component_names {
            components.insert(
                name.to_string(),
                MockComponent {
                    metrics: HashMap::new(),
                    frozen: false,
                },
            );
        }
        Self {
            components,
            global: HashMap::new(),
            interventions: Vec::new(),
        }
    }

    /// Set a metric value for a specific component.
    pub fn set_metric(&mut self, component: &str, key: &str, value: f64) {
        if let Some(comp) = self.components.get_mut(component) {
            let full_key = format!("{}.{}", component, key);
            comp.metrics.insert(full_key, value);
        }
    }

    /// Set a global metric value.
    pub fn set_global_metric(&mut self, key: &str, value: f64) {
        self.global.insert(key.to_string(), value);
    }

    /// Get the log of all interventions executed.
    pub fn interventions(&self) -> &[(String, String)] {
        &self.interventions
    }

    /// Check if a component is currently frozen.
    pub fn is_frozen(&self, component: &str) -> bool {
        self.components
            .get(component)
            .map_or(false, |c| c.frozen)
    }

    fn check_component(&self, component: &str) -> Result<(), TransXformError> {
        if self.components.contains_key(component) {
            Ok(())
        } else {
            Err(TransXformError::UnknownComponent(component.to_string()))
        }
    }
}

impl Model for MockModel {
    fn component_names(&self) -> Vec<String> {
        self.components.keys().cloned().collect()
    }

    fn has_component(&self, name: &str) -> bool {
        self.components.contains_key(name)
    }

    fn component_metrics(&self, name: &str) -> Result<MetricSnapshot, TransXformError> {
        self.check_component(name)?;
        Ok(self.components[name].metrics.clone())
    }

    fn global_metrics(&self) -> Result<MetricSnapshot, TransXformError> {
        Ok(self.global.clone())
    }

    fn reinitialize(&mut self, component: &str) -> Result<(), TransXformError> {
        self.check_component(component)?;
        self.interventions
            .push((component.to_string(), "reinitialize".to_string()));
        // Reset metrics to simulate reinitialization
        if let Some(comp) = self.components.get_mut(component) {
            comp.frozen = false;
        }
        Ok(())
    }

    fn freeze(&mut self, component: &str) -> Result<(), TransXformError> {
        self.check_component(component)?;
        self.interventions
            .push((component.to_string(), "freeze".to_string()));
        if let Some(comp) = self.components.get_mut(component) {
            comp.frozen = true;
        }
        Ok(())
    }

    fn unfreeze(&mut self, component: &str) -> Result<(), TransXformError> {
        self.check_component(component)?;
        self.interventions
            .push((component.to_string(), "unfreeze".to_string()));
        if let Some(comp) = self.components.get_mut(component) {
            comp.frozen = false;
        }
        Ok(())
    }

    fn rescale(&mut self, component: &str, factor: f64) -> Result<(), TransXformError> {
        self.check_component(component)?;
        self.interventions
            .push((component.to_string(), format!("rescale({:.4})", factor)));
        Ok(())
    }

    fn inject_noise(&mut self, component: &str, magnitude: f64) -> Result<(), TransXformError> {
        self.check_component(component)?;
        self.interventions.push((
            component.to_string(),
            format!("inject_noise({:.6})", magnitude),
        ));
        Ok(())
    }

    fn adjust_lr(&mut self, component: &str, factor: f64) -> Result<(), TransXformError> {
        self.check_component(component)?;
        self.interventions
            .push((component.to_string(), format!("adjust_lr({:.4})", factor)));
        Ok(())
    }

    fn reset_optimizer_state(&mut self, component: &str) -> Result<(), TransXformError> {
        self.check_component(component)?;
        self.interventions
            .push((component.to_string(), "reset_optimizer_state".to_string()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_model_basics() {
        let mut model = MockModel::new(&["backbone", "head"]);
        assert!(model.has_component("backbone"));
        assert!(!model.has_component("nonexistent"));
        assert_eq!(model.component_names().len(), 2);

        model.set_metric("backbone", "grad_norm", 0.05);
        let metrics = model.component_metrics("backbone").unwrap();
        assert!((metrics["backbone.grad_norm"] - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn mock_model_interventions() {
        let mut model = MockModel::new(&["head"]);
        model.reinitialize("head").unwrap();
        model.freeze("head").unwrap();
        assert!(model.is_frozen("head"));
        model.unfreeze("head").unwrap();
        assert!(!model.is_frozen("head"));
        assert_eq!(model.interventions().len(), 3);
    }

    #[test]
    fn mock_model_unknown_component() {
        let mut model = MockModel::new(&["head"]);
        assert!(model.reinitialize("nonexistent").is_err());
        assert!(model.component_metrics("nonexistent").is_err());
    }
}
