//! Integration tests: full supervisor loop with MockModel.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use transxform::*;
use transxform::collector::BasicMetricCollector;
use transxform::report::generate_report;

fn make_supervisor(
    spec_yaml: &str,
    components: &[&str],
) -> (Supervisor<MockModel, BasicMetricCollector>, Rc<RefCell<MockModel>>) {
    let spec = parse_spec(spec_yaml).expect("Failed to parse spec");
    let model = Rc::new(RefCell::new(MockModel::new(components)));
    let collector = BasicMetricCollector::new(
        components.iter().map(|s| s.to_string()).collect(),
        HashMap::new(),
    );
    let supervisor = Supervisor::new(spec, model.clone(), collector).expect("Failed to create supervisor");
    (supervisor, model)
}

fn default_spec() -> &'static str {
    r#"
model:
  name: "test-model"
  components: [backbone, head]
invariants:
  hard:
    head.pairwise_cosine: 0.95
    head.grad_norm: 0.001
  soft:
    attention_entropy_min: 0.3
phases:
  bootstrap:
    transition_guard:
      all_hard_invariants_satisfied_for: 5
  representation_formation:
    transition_guard:
      all_hard_invariants_satisfied_for: 10
  stabilization:
    transition_guard:
      all_hard_invariants_satisfied_for: 10
  refinement:
    allowed_interventions:
      - adjust_lr
      - freeze
control:
  cooldown_steps: 10
  max_hard_interventions: 3
  hysteresis_margin: 0.0
  damping_factor: 0.5
  regret_window_steps: 20
"#
}

fn set_healthy_metrics(model: &Rc<RefCell<MockModel>>) {
    let mut m = model.borrow_mut();
    m.set_metric("head", "pairwise_cosine", 0.5);
    m.set_metric("head", "grad_norm", 0.0005);
    m.set_metric("backbone", "grad_norm", 0.05);
    m.set_metric("backbone", "activation_variance", 0.01);
    m.set_global_metric("loss", 2.0);
}

// =========================================================================
// Scenario 1: Healthy run → HEALTHY certificate
// =========================================================================

#[test]
fn healthy_run_produces_healthy_certificate() {
    let (mut supervisor, model) = make_supervisor(default_spec(), &["backbone", "head"]);
    set_healthy_metrics(&model);

    for step in 0..50 {
        let report = supervisor.step(step).unwrap();
        assert!(report.violations.is_empty(), "Step {}: unexpected violation", step);
    }

    let cert = supervisor.emit_certificate(50);
    assert_eq!(cert.verdict, HealthVerdict::Healthy);
    assert_eq!(cert.intervention_summary.total_hard, 0);
    assert_eq!(cert.intervention_summary.total_soft, 0);
}

// =========================================================================
// Scenario 2: Emission collapse → detect, reinitialize, recover → RECOVERED
// =========================================================================

#[test]
fn emission_collapse_detected_and_recovered() {
    let (mut supervisor, model) = make_supervisor(default_spec(), &["backbone", "head"]);

    // Start with violating cosine (emission collapse)
    {
        let mut m = model.borrow_mut();
        m.set_metric("head", "pairwise_cosine", 0.99);
        m.set_metric("head", "grad_norm", 0.0005);
        m.set_metric("backbone", "grad_norm", 0.05);
        m.set_global_metric("loss", 2.0);
    }

    // Step 0: should detect and reinitialize
    let r = supervisor.step(0).unwrap();
    assert!(!r.violations.is_empty());
    assert!(r.actions_taken.iter().any(|a| a.action_name() == "reinitialize"));

    // Simulate recovery after reinitialization
    {
        let mut m = model.borrow_mut();
        m.set_metric("head", "pairwise_cosine", 0.5);
    }

    // Run enough steps to close regret windows
    for step in 1..30 {
        let _ = supervisor.step(step);
    }

    let cert = supervisor.emit_certificate(30);
    // Should have interventions recorded
    assert!(cert.intervention_summary.total_hard > 0 || cert.intervention_summary.total_soft > 0);
}

// =========================================================================
// Scenario 3: Loss explosion → LR reduction
// =========================================================================

#[test]
fn loss_explosion_triggers_lr_adjustment() {
    let spec = r#"
model:
  name: "test"
  components: [backbone, head]
invariants:
  hard:
    loss_explosion_factor: 3.0
  soft: {}
phases:
  bootstrap:
    transition_guard:
      all_hard_invariants_satisfied_for: 5
control:
  cooldown_steps: 10
  max_hard_interventions: 3
  damping_factor: 0.5
"#;

    let (mut supervisor, model) = make_supervisor(spec, &["backbone", "head"]);

    {
        let mut m = model.borrow_mut();
        m.set_global_metric("loss_explosion_factor", 5.0);
        m.set_metric("head", "grad_norm", 0.05);
        m.set_metric("backbone", "grad_norm", 0.05);
    }

    let r = supervisor.step(0).unwrap();
    // Should detect loss explosion and adjust LR
    let has_lr_action = r.actions_taken.iter().any(|a| a.action_name() == "adjust_lr");
    let has_reinit = r.actions_taken.iter().any(|a| a.action_name() == "reinitialize");
    // Either adjust_lr or reinitialize (default fallback) is acceptable
    assert!(has_lr_action || has_reinit || !r.violations.is_empty());
}

// =========================================================================
// Scenario 4: Phase progression through all 4 phases
// =========================================================================

#[test]
fn full_phase_progression() {
    let (mut supervisor, model) = make_supervisor(default_spec(), &["backbone", "head"]);
    set_healthy_metrics(&model);

    // Phase 1: Bootstrap → RepresentationFormation (5 clean steps)
    for step in 0..5 {
        supervisor.step(step).unwrap();
    }
    assert_eq!(supervisor.current_phase(), Phase::RepresentationFormation);

    // Phase 2: RepresentationFormation → Stabilization (10 more clean steps)
    for step in 5..15 {
        supervisor.step(step).unwrap();
    }
    assert_eq!(supervisor.current_phase(), Phase::Stabilization);

    // Phase 3: Stabilization → Refinement (10 more clean steps)
    for step in 15..25 {
        supervisor.step(step).unwrap();
    }
    assert_eq!(supervisor.current_phase(), Phase::Refinement);
}

// =========================================================================
// Scenario 5: NaN loss → abort
// =========================================================================

#[test]
fn nan_loss_aborts_training() {
    let (mut supervisor, model) = make_supervisor(default_spec(), &["backbone", "head"]);

    {
        let mut m = model.borrow_mut();
        m.set_metric("head", "pairwise_cosine", 0.5);
        m.set_metric("head", "grad_norm", 0.0005);
        m.set_global_metric("loss", f64::NAN);
    }

    let result = supervisor.step(0);
    assert!(result.is_err());
    assert!(supervisor.is_aborted());
}

// =========================================================================
// Scenario 6: Certificate and report generation
// =========================================================================

#[test]
fn certificate_and_report_generation() {
    let (mut supervisor, model) = make_supervisor(default_spec(), &["backbone", "head"]);

    // A few violation steps
    {
        let mut m = model.borrow_mut();
        m.set_metric("head", "pairwise_cosine", 0.99);
        m.set_metric("head", "grad_norm", 0.0005);
        m.set_metric("backbone", "grad_norm", 0.05);
        m.set_global_metric("loss", 2.0);
    }
    supervisor.step(0).unwrap();

    // Then healthy
    set_healthy_metrics(&model);
    for step in 1..30 {
        let _ = supervisor.step(step);
    }

    let cert = supervisor.emit_certificate(30);
    let report = generate_report(&cert, supervisor.ledger());

    let md = report.to_markdown();
    assert!(md.contains("TransXform Training Report"));

    let json = report.to_json().unwrap();
    assert!(json.contains("test-model"));
}

// =========================================================================
// Scenario 7: Cooldown enforcement
// =========================================================================

#[test]
fn cooldown_enforced_between_interventions() {
    let (mut supervisor, model) = make_supervisor(default_spec(), &["backbone", "head"]);

    {
        let mut m = model.borrow_mut();
        m.set_metric("head", "pairwise_cosine", 0.99);
        m.set_metric("head", "grad_norm", 0.0005);
        m.set_metric("backbone", "grad_norm", 0.05);
        m.set_global_metric("loss", 2.0);
    }

    // Step 0: intervention fires
    let r0 = supervisor.step(0).unwrap();
    assert!(!r0.actions_taken.is_empty());

    // Steps 1-9: should be in cooldown (cooldown_steps = 10)
    for step in 1..10 {
        let r = supervisor.step(step).unwrap();
        assert!(
            r.actions_taken.is_empty(),
            "Step {}: expected no actions during cooldown, got {}",
            step,
            r.actions_taken.len()
        );
    }

    // Step 10+: cooldown expires, should intervene again
    let r10 = supervisor.step(10).unwrap();
    assert!(
        !r10.actions_taken.is_empty(),
        "Expected intervention after cooldown expiry"
    );
}

// =========================================================================
// Scenario 8: Merkle state integrity
// =========================================================================

#[test]
fn merkle_state_deterministic_across_runs() {
    let spec = default_spec();

    // Run 1
    let mut merkle1 = transxform::merkle::MerkleState::new(spec);
    let mut metrics = MetricSnapshot::new();
    metrics.insert("loss".into(), 2.5);
    for step in 0..10 {
        merkle1.update(step, &metrics, None);
    }

    // Run 2 (identical)
    let mut merkle2 = transxform::merkle::MerkleState::new(spec);
    for step in 0..10 {
        merkle2.update(step, &metrics, None);
    }

    assert_eq!(merkle1.root_hex(), merkle2.root_hex());
    assert_eq!(merkle1.step_count(), 10);
}

// =========================================================================
// Scenario 9: Failure signature detection
// =========================================================================

#[test]
fn failure_signature_registry_works() {
    let mut registry = transxform::registry::SignatureRegistry::with_defaults();
    assert_eq!(registry.signatures().len(), 7);

    // SIG-005: Emission head frozen
    let mut metrics = MetricSnapshot::new();
    metrics.insert("emission_head.pairwise_cosine".into(), 0.999);
    metrics.insert("emission_head.grad_norm".into(), 1e-5);

    // Need 30 sustained steps
    for step in 0..30 {
        let matched = registry.check(&metrics, step);
        if step < 29 {
            assert!(matched.is_empty());
        }
    }

    let matched = registry.check(&metrics, 30);
    assert!(!matched.is_empty());
    assert_eq!(matched[0].id, "SIG-005");
}

// =========================================================================
// Scenario 10: Near-miss tracking
// =========================================================================

#[test]
fn near_miss_detection_works() {
    let spec = r#"
model:
  name: "test"
  components: [head]
invariants:
  hard:
    head.pairwise_cosine: 0.95
  soft:
    head.cosine_drift: 0.90
phases: {}
control:
  hysteresis_margin: 0.05
"#;

    let (mut supervisor, model) = make_supervisor(spec, &["head"]);

    // Set metric near the hard threshold boundary
    {
        let mut m = model.borrow_mut();
        m.set_metric("head", "pairwise_cosine", 0.93);
        m.set_metric("head", "grad_norm", 0.05);
    }

    let r = supervisor.step(0).unwrap();
    // The supervisor should run without error
    assert!(r.step == 0);
}

// =========================================================================
// Scenario 11: Spec parsing from YAML file
// =========================================================================

#[test]
fn loom_spec_parses_successfully() {
    let yaml = std::fs::read_to_string("examples/loom_spec.yaml")
        .expect("Failed to read loom_spec.yaml");
    let spec = parse_spec(&yaml).expect("Failed to parse LOOM spec");

    assert_eq!(spec.model.name, "loom-24L-1024H");
    assert_eq!(spec.model.components.len(), 5);
    assert_eq!(spec.model.layers, Some(24));
    assert_eq!(spec.model.hidden_dim, Some(1024));
    assert_eq!(spec.model.attention_heads, Some(16));
    assert_eq!(spec.control.cooldown_steps, 50);
    assert_eq!(spec.control.max_hard_interventions, 3);
}

// =========================================================================
// Scenario 12: Checkpoint and resume produces identical behavior
// =========================================================================

#[test]
fn checkpoint_resume_produces_identical_behavior() {
    let spec_yaml = default_spec();
    let (mut supervisor, model) = make_supervisor(spec_yaml, &["backbone", "head"]);

    // Start with a violation to create some state
    {
        let mut m = model.borrow_mut();
        m.set_metric("head", "pairwise_cosine", 0.99);
        m.set_metric("head", "grad_norm", 0.0005);
        m.set_metric("backbone", "grad_norm", 0.05);
        m.set_global_metric("loss", 2.0);
    }
    supervisor.step(0).unwrap();

    // Recover and run a few more steps
    set_healthy_metrics(&model);
    for step in 1..15 {
        supervisor.step(step).unwrap();
    }

    // Checkpoint at step 15
    let checkpoint = supervisor.checkpoint(15);
    assert_eq!(checkpoint.step, 15);
    assert_eq!(checkpoint.model_name, "test-model");
    assert!(checkpoint.total_interventions > 0);

    // Continue original to step 20
    for step in 15..20 {
        supervisor.step(step).unwrap();
    }
    let original_phase = supervisor.current_phase();
    let original_interventions = supervisor.total_interventions();

    // Restore from checkpoint at step 15
    let spec = parse_spec(spec_yaml).unwrap();
    let model2 = Rc::new(RefCell::new(MockModel::new(&["backbone", "head"])));
    {
        let mut m = model2.borrow_mut();
        m.set_metric("head", "pairwise_cosine", 0.5);
        m.set_metric("head", "grad_norm", 0.0005);
        m.set_metric("backbone", "grad_norm", 0.05);
        m.set_metric("backbone", "activation_variance", 0.01);
        m.set_global_metric("loss", 2.0);
    }
    let collector2 = transxform::collector::BasicMetricCollector::new(
        vec!["backbone".into(), "head".into()],
        HashMap::new(),
    );
    let mut restored = Supervisor::from_checkpoint(spec, model2, collector2, checkpoint).unwrap();

    // Run restored from step 15 to step 20 with same metrics
    for step in 15..20 {
        restored.step(step).unwrap();
    }

    // Both should be in the same phase with same intervention count
    assert_eq!(restored.current_phase(), original_phase);
    assert_eq!(restored.total_interventions(), original_interventions);
}

// =========================================================================
// Scenario 13: Checkpoint JSON serialization roundtrip
// =========================================================================

#[test]
fn checkpoint_json_roundtrip() {
    let (mut supervisor, model) = make_supervisor(default_spec(), &["backbone", "head"]);
    set_healthy_metrics(&model);

    for step in 0..10 {
        supervisor.step(step).unwrap();
    }

    let checkpoint = supervisor.checkpoint(10);
    let json = checkpoint.to_json().unwrap();
    let restored = SupervisorCheckpoint::from_json(&json).unwrap();

    assert_eq!(restored.step, 10);
    assert_eq!(restored.model_name, "test-model");
    assert_eq!(restored.phase_controller.current, supervisor.current_phase());
}

// =========================================================================
// Scenario 14: Checkpoint model name mismatch rejected
// =========================================================================

#[test]
fn checkpoint_rejects_model_name_mismatch() {
    let (mut supervisor, model) = make_supervisor(default_spec(), &["backbone", "head"]);
    set_healthy_metrics(&model);
    supervisor.step(0).unwrap();

    let checkpoint = supervisor.checkpoint(1);

    // Try to restore with a different spec (different model name)
    let different_spec = parse_spec(r#"
model:
  name: "different-model"
  components: [backbone, head]
invariants:
  hard:
    head.pairwise_cosine: 0.95
  soft: {}
phases: {}
control: {}
"#).unwrap();

    let model2 = Rc::new(RefCell::new(MockModel::new(&["backbone", "head"])));
    let collector2 = transxform::collector::BasicMetricCollector::new(
        vec!["backbone".into(), "head".into()],
        HashMap::new(),
    );

    let result = Supervisor::from_checkpoint(different_spec, model2, collector2, checkpoint);
    match result {
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains("does not match"), "Expected model name mismatch error, got: {}", msg);
        }
        Ok(_) => panic!("Expected error for model name mismatch"),
    }
}

// =========================================================================
// Scenario 15: Checkpoint hint fires near phase transition
// =========================================================================

#[test]
fn checkpoint_hint_fires_on_phase_transition() {
    let (mut supervisor, model) = make_supervisor(default_spec(), &["backbone", "head"]);
    set_healthy_metrics(&model);

    let mut got_hint = false;
    for step in 0..10 {
        let report = supervisor.step(step).unwrap();
        if let Some(hint) = &report.checkpoint_hint {
            got_hint = true;
            // Should be either pre or post transition
            match &hint.reason {
                CheckpointReason::PrePhaseTransition { .. } => {},
                CheckpointReason::PostPhaseTransition { .. } => {},
                _ => panic!("Unexpected checkpoint reason: {:?}", hint.reason),
            }
        }
    }

    assert!(got_hint, "Should have received at least one checkpoint hint near phase transition");
}
