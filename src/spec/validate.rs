use crate::error::TransXformError;
use super::types::{InvariantValue, TrainingSpec};

const VALID_ACTIONS: &[&str] = &[
    "reinitialize",
    "freeze",
    "unfreeze",
    "rescale",
    "inject_noise",
    "adjust_lr",
    "abort",
];

/// Validate a parsed training spec for internal consistency.
pub fn validate_spec(spec: &TrainingSpec) -> Result<(), TransXformError> {
    validate_components(spec)?;
    validate_invariant_names(spec)?;
    validate_control_config(spec)?;
    validate_phases(spec)?;
    Ok(())
}

/// All components referenced in invariants must exist in model.components.
fn validate_components(spec: &TrainingSpec) -> Result<(), TransXformError> {
    let components: std::collections::HashSet<&str> =
        spec.model.components.iter().map(|s| s.as_str()).collect();

    // Check per-component invariants reference valid components
    for (name, value) in spec.invariants.hard.iter().chain(spec.invariants.soft.iter()) {
        if let InvariantValue::PerComponent(map) = value {
            for comp in map.keys() {
                if comp != "all_layers" && comp != "all" && !components.contains(comp.as_str()) {
                    return Err(TransXformError::SpecValidation(format!(
                        "Invariant '{}' references unknown component '{}'",
                        name, comp
                    )));
                }
            }
        }
        // For dot-notation keys like "emission_head.grad_norm", check the component prefix
        if let Some(dot_pos) = name.find('.') {
            let comp = &name[..dot_pos];
            if !components.contains(comp) && comp != "all" && comp != "all_layers" {
                return Err(TransXformError::SpecValidation(format!(
                    "Invariant '{}' references unknown component '{}'",
                    name, comp
                )));
            }
        }
    }

    Ok(())
}

/// Hard and soft invariants cannot share the same name.
fn validate_invariant_names(spec: &TrainingSpec) -> Result<(), TransXformError> {
    for name in spec.invariants.hard.keys() {
        if spec.invariants.soft.contains_key(name) {
            return Err(TransXformError::SpecValidation(format!(
                "Invariant '{}' declared as both hard and soft",
                name
            )));
        }
    }
    Ok(())
}

/// Control config values must be within valid ranges.
fn validate_control_config(spec: &TrainingSpec) -> Result<(), TransXformError> {
    let c = &spec.control;

    if c.cooldown_steps == 0 {
        return Err(TransXformError::SpecValidation(
            "cooldown_steps must be > 0".into(),
        ));
    }

    if c.max_hard_interventions == 0 {
        return Err(TransXformError::SpecValidation(
            "max_hard_interventions must be >= 1".into(),
        ));
    }

    if c.hysteresis_margin < 0.0 {
        return Err(TransXformError::SpecValidation(
            "hysteresis_margin must be >= 0.0".into(),
        ));
    }

    if c.damping_factor <= 0.0 || c.damping_factor > 1.0 {
        return Err(TransXformError::SpecValidation(
            "damping_factor must be in (0.0, 1.0]".into(),
        ));
    }

    Ok(())
}

/// Phase-specific validation: transition guards > 0, allowed interventions are valid.
fn validate_phases(spec: &TrainingSpec) -> Result<(), TransXformError> {
    let phase_decls = [
        ("bootstrap", &spec.phases.bootstrap),
        ("representation_formation", &spec.phases.representation_formation),
        ("stabilization", &spec.phases.stabilization),
        ("refinement", &spec.phases.refinement),
    ];

    for (phase_name, decl_opt) in &phase_decls {
        if let Some(decl) = decl_opt {
            // Validate transition guard
            if let Some(guard) = &decl.transition_guard {
                if guard.all_hard_invariants_satisfied_for == 0 {
                    return Err(TransXformError::SpecValidation(format!(
                        "Phase '{}': transition_guard.all_hard_invariants_satisfied_for must be > 0",
                        phase_name
                    )));
                }
            }

            // Validate allowed interventions
            if let Some(allowed) = &decl.allowed_interventions {
                for action_name in allowed {
                    if !VALID_ACTIONS.contains(&action_name.as_str()) {
                        return Err(TransXformError::SpecValidation(format!(
                            "Phase '{}': unknown intervention '{}'",
                            phase_name, action_name
                        )));
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::spec::parse::parse_spec;

    #[test]
    fn reject_unknown_component_in_invariant() {
        let yaml = r#"
model:
  name: "test"
  components: [backbone]
invariants:
  hard:
    nonexistent.grad_norm: 0.001
  soft: {}
phases: {}
control: {}
"#;
        let err = parse_spec(yaml).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("nonexistent"), "Error: {}", msg);
    }

    #[test]
    fn reject_duplicate_hard_soft_name() {
        let yaml = r#"
model:
  name: "test"
  components: [a]
invariants:
  hard:
    shared_name: 0.5
  soft:
    shared_name: 0.3
phases: {}
control: {}
"#;
        let err = parse_spec(yaml).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("both hard and soft"), "Error: {}", msg);
    }

    #[test]
    fn reject_zero_cooldown() {
        let yaml = r#"
model:
  name: "test"
  components: [a]
invariants:
  hard: {}
  soft: {}
phases: {}
control:
  cooldown_steps: 0
"#;
        let err = parse_spec(yaml).unwrap_err();
        assert!(format!("{}", err).contains("cooldown_steps"));
    }

    #[test]
    fn reject_invalid_damping() {
        let yaml = r#"
model:
  name: "test"
  components: [a]
invariants:
  hard: {}
  soft: {}
phases: {}
control:
  damping_factor: 1.5
"#;
        let err = parse_spec(yaml).unwrap_err();
        assert!(format!("{}", err).contains("damping_factor"));
    }

    #[test]
    fn reject_unknown_intervention() {
        let yaml = r#"
model:
  name: "test"
  components: [a]
invariants:
  hard: {}
  soft: {}
phases:
  bootstrap:
    allowed_interventions:
      - "magic_wand"
control: {}
"#;
        let err = parse_spec(yaml).unwrap_err();
        assert!(format!("{}", err).contains("magic_wand"));
    }

    #[test]
    fn accept_valid_spec() {
        let yaml = r#"
model:
  name: "loom"
  layers: 24
  hidden_dim: 1024
  attention_heads: 16
  components:
    - backbone
    - emission_head

invariants:
  hard:
    emission_head.pairwise_cosine: 0.95
    emission_head.grad_norm: 0.001
  soft:
    attention_entropy_min: 0.3
    loss_explosion_factor: 3.0

phases:
  bootstrap:
    description: "Initial learning"
    thresholds:
      emission_head.grad_norm: 0.0001
    max_duration_steps: 500
  representation_formation:
    transition_guard:
      all_hard_invariants_satisfied_for: 100
  refinement:
    allowed_interventions:
      - adjust_lr
      - freeze

control:
  cooldown_steps: 50
  max_hard_interventions: 3
  hysteresis_margin: 0.05
  damping_factor: 0.5
"#;
        let spec = parse_spec(yaml).unwrap();
        assert_eq!(spec.model.name, "loom");
        assert_eq!(spec.model.components.len(), 2);
    }
}
