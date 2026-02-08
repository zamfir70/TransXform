use std::path::Path;

use crate::error::TransXformError;
use super::types::TrainingSpec;
use super::validate::validate_spec;

/// Parse a training spec from a YAML string.
pub fn parse_spec(yaml: &str) -> Result<TrainingSpec, TransXformError> {
    let spec: TrainingSpec = serde_yaml::from_str(yaml)?;
    validate_spec(&spec)?;
    Ok(spec)
}

/// Parse a training spec from a YAML file.
pub fn parse_spec_from_file(path: &Path) -> Result<TrainingSpec, TransXformError> {
    let contents = std::fs::read_to_string(path)?;
    parse_spec(&contents)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_SPEC: &str = r#"
model:
  name: "test_model"
  components:
    - backbone
    - head

invariants:
  hard:
    head.grad_norm: 0.001
  soft:
    loss_explosion_factor: 3.0

phases:
  bootstrap:
    description: "Initial learning"
    thresholds:
      head.grad_norm: 0.0001

control:
  cooldown_steps: 50
  max_hard_interventions: 3
  hysteresis_margin: 0.05
  damping_factor: 0.5
"#;

    #[test]
    fn parse_minimal_spec() {
        let spec = parse_spec(MINIMAL_SPEC).unwrap();
        assert_eq!(spec.model.name, "test_model");
        assert_eq!(spec.model.components.len(), 2);
        assert!(spec.invariants.hard.contains_key("head.grad_norm"));
        assert!(spec.invariants.soft.contains_key("loss_explosion_factor"));
        assert_eq!(spec.control.cooldown_steps, 50);
    }

    #[test]
    fn parse_defaults_applied() {
        let yaml = r#"
model:
  name: "test"
  components: [a]
invariants:
  hard: {}
  soft: {}
phases: {}
control: {}
"#;
        let spec = parse_spec(yaml).unwrap();
        assert_eq!(spec.control.cooldown_steps, 50);
        assert_eq!(spec.control.max_hard_interventions, 3);
        assert!((spec.control.hysteresis_margin - 0.05).abs() < f64::EPSILON);
        assert!((spec.control.damping_factor - 0.5).abs() < f64::EPSILON);
        assert_eq!(spec.control.regret_window_steps, 100);
    }

    #[test]
    fn parse_per_component_invariant() {
        let yaml = r#"
model:
  name: "test"
  components: [backbone, head]
invariants:
  hard:
    grad_norm_min:
      backbone: 0.0001
      head: 0.001
  soft: {}
phases: {}
control: {}
"#;
        let spec = parse_spec(yaml).unwrap();
        match &spec.invariants.hard["grad_norm_min"] {
            super::super::types::InvariantValue::PerComponent(map) => {
                assert!((map["backbone"] - 0.0001).abs() < f64::EPSILON);
                assert!((map["head"] - 0.001).abs() < f64::EPSILON);
            }
            _ => panic!("Expected PerComponent variant"),
        }
    }

    #[test]
    fn reject_invalid_yaml() {
        let result = parse_spec("not: [valid: yaml: {{");
        assert!(result.is_err());
    }
}
