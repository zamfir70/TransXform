//! Basic training loop example demonstrating TransXform supervisor integration.
//!
//! This example shows how to:
//! 1. Parse a training spec from YAML
//! 2. Create a supervisor with a MockModel
//! 3. Run a training loop with supervisor feedback
//! 4. Generate a certificate and report at the end
//!
//! Run with: `cargo run --example basic_training`

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use transxform::{
    BasicMetricCollector, MockModel, Supervisor,
    merkle::MerkleState,
    report::generate_report,
};

fn main() {
    let spec_yaml = r#"
model:
  name: "example-model"
  components:
    - backbone
    - head

invariants:
  hard:
    head.pairwise_cosine: 0.95
    head.grad_norm: 0.001
    backbone.activation_variance_min: 0.0001
  soft:
    attention_entropy_min: 0.3
    loss_explosion_factor: 3.0

phases:
  bootstrap:
    transition_guard:
      all_hard_invariants_satisfied_for: 20
  representation_formation:
    transition_guard:
      all_hard_invariants_satisfied_for: 30
  stabilization:
    transition_guard:
      all_hard_invariants_satisfied_for: 30
  refinement:
    allowed_interventions:
      - adjust_lr
      - freeze

control:
  cooldown_steps: 10
  max_hard_interventions: 3
  hysteresis_margin: 0.02
  damping_factor: 0.5
  regret_window_steps: 50
"#;

    // 1. Create model and supervisor
    let model = Rc::new(RefCell::new(MockModel::new(&["backbone", "head"])));
    let collector = BasicMetricCollector::new(
        vec!["backbone".into(), "head".into()],
        HashMap::new(),
    );

    let spec = transxform::parse_spec(spec_yaml).expect("Failed to parse spec");
    let mut supervisor =
        Supervisor::new(spec, model.clone(), collector).expect("Failed to create supervisor");

    // Optional: Merkle state for run integrity
    let mut merkle = MerkleState::new(spec_yaml);

    println!("=== TransXform Basic Training Example ===\n");
    println!("Starting training with supervisor...\n");

    let total_steps = 200;

    for step in 0..total_steps {
        // Simulate training: update model metrics based on step
        {
            let mut m = model.borrow_mut();

            // Simulate gradually improving metrics
            let progress = step as f64 / total_steps as f64;

            // Cosine similarity: starts high (bad), improves over time
            let cosine = 0.99 - progress * 0.6;
            m.set_metric("head", "pairwise_cosine", cosine);

            // Gradient norm: healthy throughout
            m.set_metric("head", "grad_norm", 0.005 * (1.0 + progress));
            m.set_metric("backbone", "grad_norm", 0.01);

            // Activation variance: healthy
            m.set_metric("backbone", "activation_variance", 0.01);

            // Loss: decreasing
            let loss = 5.0 * (1.0 - progress * 0.8);
            m.set_global_metric("loss", loss);
        }

        // Run supervisor step
        match supervisor.step(step) {
            Ok(report) => {
                // Update Merkle state
                let action = report.actions_taken.first();
                merkle.update(step, &report.metrics, action);

                // Print interesting events
                if !report.violations.is_empty() {
                    println!(
                        "Step {:4} [{}] {} violations, {} actions",
                        step,
                        report.phase,
                        report.violations.len(),
                        report.actions_taken.len(),
                    );
                    for action in &report.actions_taken {
                        println!("  -> {}", action);
                    }
                }

                if let Some(ref transition) = report.phase_transition {
                    println!(
                        "\n*** Phase transition: {} ***\n",
                        transition,
                    );
                }

                // V1.4: Checkpoint hint
                if let Some(ref hint) = report.checkpoint_hint {
                    println!(
                        "Step {:4}: Checkpoint hint: {:?}",
                        step, hint.reason,
                    );
                    // In a real training loop, you'd save here:
                    // let checkpoint = supervisor.checkpoint(step);
                    // checkpoint.save_json(Path::new("checkpoint.json")).unwrap();
                }

                if !report.signature_matches.is_empty() {
                    println!(
                        "Step {:4}: Signature matches: {:?}",
                        step, report.signature_matches,
                    );
                }
            }
            Err(e) => {
                println!("Step {:4}: Training error: {}", step, e);
                break;
            }
        }
    }

    // 3. Generate certificate and report
    println!("\n=== Training Complete ===\n");

    let certificate = supervisor.emit_certificate(total_steps);
    println!("Certificate:");
    println!("  Model:     {}", certificate.model_name);
    println!("  Verdict:   {}", certificate.verdict);
    println!("  Steps:     {}", certificate.total_steps);
    println!(
        "  Interventions: {} hard, {} soft",
        certificate.intervention_summary.total_hard,
        certificate.intervention_summary.total_soft,
    );
    println!(
        "  Regret: {} assessed, {} confident, {} low-confidence",
        certificate.regret_summary.total_assessed,
        certificate.regret_summary.confident,
        certificate.regret_summary.low_confidence,
    );

    // Generate markdown report
    let report = generate_report(&certificate, supervisor.ledger());
    let markdown = report.to_markdown();
    println!("\n--- Report ---\n");
    println!("{}", markdown);

    // Merkle root
    println!("Merkle root: {}", merkle.root_hex());
    println!("Steps hashed: {}", merkle.step_count());
}
