use std::collections::HashMap;

use crate::ledger::{
    BoundaryLedger, InterventionOutcome, LedgerEntryType, TrainingCertificate,
};
use crate::diagnostic::DiagnosticLayer;
use crate::types::*;

/// A structured end-of-run report (spinning rims spec).
pub struct Report {
    pub executive_summary: ExecutiveSummary,
    pub intervention_table: Vec<InterventionRow>,
    pub regret_analysis: RegretAnalysis,
    pub final_health: FinalHealthSnapshot,
    /// V2: Diagnostic advisories (non-authoritative).
    pub diagnostic_advisories: DiagnosticAdvisories,
}

/// V2: Advisory diagnostic section — collected from the diagnostic layer.
pub struct DiagnosticAdvisories {
    pub warnings: Vec<DiagnosticAdvisoryRow>,
    pub total: u64,
    pub acknowledged: u64,
    pub unacknowledged: u64,
}

/// One row in the diagnostic advisory section.
pub struct DiagnosticAdvisoryRow {
    pub step: u64,
    pub signal: String,
    pub summary: String,
    pub evidence: Vec<String>,
    pub confidence: f64,
    pub acknowledged: bool,
}

/// High-level summary of training outcome.
pub struct ExecutiveSummary {
    pub model_name: String,
    pub verdict: HealthVerdict,
    pub total_steps: u64,
    pub total_interventions: u64,
    pub phases_completed: Vec<Phase>,
    pub near_abort_conditions: Vec<String>,
    pub phase_regressions: u64,
}

/// One row in the intervention table.
pub struct InterventionRow {
    pub step: u64,
    pub phase: Phase,
    pub component: String,
    pub invariant: String,
    pub observed: Option<f64>,
    pub threshold: Option<f64>,
    pub action: String,
    pub outcome: String,
    pub recovery_steps: Option<u64>,
}

/// Regret analysis section.
pub struct RegretAnalysis {
    pub likely_necessary: Vec<RegretRow>,
    pub possibly_preemptive: Vec<RegretRow>,
    pub cooldown_near_misses: u64,
}

pub struct RegretRow {
    pub step: u64,
    pub component: String,
    pub action: String,
    pub tag: RegretTag,
    pub post_improvement: Option<f64>,
    pub was_recovering: bool,
}

/// Final per-component health snapshot.
pub struct FinalHealthSnapshot {
    pub components: Vec<ComponentHealth>,
    pub overall_loss: Option<f64>,
}

pub struct ComponentHealth {
    pub name: String,
    pub grad_norm: Option<f64>,
    pub activation_variance: Option<f64>,
    pub pairwise_cosine: Option<f64>,
    pub status: String,
}

/// Generate a full report from the certificate and ledger.
pub fn generate_report(
    certificate: &TrainingCertificate,
    ledger: &BoundaryLedger,
) -> Report {
    generate_report_with_diagnostics(certificate, ledger, None)
}

/// Generate a full report with V2 diagnostic advisories.
pub fn generate_report_with_diagnostics(
    certificate: &TrainingCertificate,
    ledger: &BoundaryLedger,
    diagnostic: Option<&DiagnosticLayer>,
) -> Report {
    Report {
        executive_summary: build_executive_summary(certificate),
        intervention_table: build_intervention_table(ledger),
        regret_analysis: build_regret_analysis(certificate, ledger),
        final_health: build_final_health(certificate),
        diagnostic_advisories: build_diagnostic_advisories(diagnostic),
    }
}

fn build_diagnostic_advisories(diagnostic: Option<&DiagnosticLayer>) -> DiagnosticAdvisories {
    let Some(diag) = diagnostic else {
        return DiagnosticAdvisories {
            warnings: Vec::new(),
            total: 0,
            acknowledged: 0,
            unacknowledged: 0,
        };
    };

    let warnings: Vec<DiagnosticAdvisoryRow> = diag
        .warnings()
        .iter()
        .map(|w| DiagnosticAdvisoryRow {
            step: w.step,
            signal: w.signal.to_string(),
            summary: w.summary.clone(),
            evidence: w.evidence.clone(),
            confidence: w.confidence,
            acknowledged: w.acknowledged,
        })
        .collect();

    let total = warnings.len() as u64;
    let acknowledged = warnings.iter().filter(|w| w.acknowledged).count() as u64;

    DiagnosticAdvisories {
        warnings,
        total,
        acknowledged,
        unacknowledged: total - acknowledged,
    }
}

fn build_executive_summary(cert: &TrainingCertificate) -> ExecutiveSummary {
    let phases_completed: Vec<Phase> = cert
        .phase_trace
        .iter()
        .map(|t| t.to)
        .filter(|p| !p.is_terminal())
        .collect();

    let regressions = cert
        .phase_trace
        .windows(2)
        .filter(|w| {
            // A regression is when we go "backwards"
            matches!(
                (&w[0].to, &w[1].to),
                (Phase::RepresentationFormation, Phase::Bootstrap)
                    | (Phase::Stabilization, Phase::RepresentationFormation)
                    | (Phase::Refinement, Phase::Stabilization)
            )
        })
        .count() as u64;

    let near_abort: Vec<String> = cert
        .phase_trace
        .iter()
        .filter(|t| t.to == Phase::Aborted || t.reason.contains("exhausted"))
        .map(|t| format!("Step {}: {}", t.step, t.reason))
        .collect();

    ExecutiveSummary {
        model_name: cert.model_name.clone(),
        verdict: cert.verdict.clone(),
        total_steps: cert.total_steps,
        total_interventions: cert.intervention_summary.total_hard
            + cert.intervention_summary.total_soft,
        phases_completed,
        near_abort_conditions: near_abort,
        phase_regressions: regressions,
    }
}

fn build_intervention_table(ledger: &BoundaryLedger) -> Vec<InterventionRow> {
    ledger
        .entries()
        .iter()
        .filter(|e| matches!(e.entry_type, LedgerEntryType::Violation))
        .map(|e| {
            let observed = e.metric_snapshot.get(&e.invariant).copied();
            InterventionRow {
                step: e.step,
                phase: e.phase,
                component: e.component.clone(),
                invariant: e.invariant.clone(),
                observed,
                threshold: None,
                action: format!("{}", e.action),
                outcome: match &e.outcome {
                    InterventionOutcome::Pending => "pending".into(),
                    InterventionOutcome::Recovered => "recovered".into(),
                    InterventionOutcome::Persisted => "persisted".into(),
                    InterventionOutcome::Worsened => "worsened".into(),
                },
                recovery_steps: None,
            }
        })
        .collect()
}

fn build_regret_analysis(
    cert: &TrainingCertificate,
    ledger: &BoundaryLedger,
) -> RegretAnalysis {
    let mut likely_necessary = Vec::new();
    let mut possibly_preemptive = Vec::new();

    for entry in ledger.entries() {
        if !matches!(entry.entry_type, LedgerEntryType::Violation) {
            continue;
        }
        if let Some(tag) = &entry.regret_tag {
            let row = RegretRow {
                step: entry.step,
                component: entry.component.clone(),
                action: format!("{}", entry.action),
                tag: *tag,
                post_improvement: None,
                was_recovering: false,
            };
            match tag {
                RegretTag::Confident => likely_necessary.push(row),
                RegretTag::LowConfidence => possibly_preemptive.push(row),
                RegretTag::Pending => {}
            }
        }
    }

    RegretAnalysis {
        likely_necessary,
        possibly_preemptive,
        cooldown_near_misses: cert.regret_summary.near_misses,
    }
}

fn build_final_health(cert: &TrainingCertificate) -> FinalHealthSnapshot {
    // Group metrics by component
    let mut components_map: HashMap<String, ComponentHealth> = HashMap::new();

    for (key, &value) in &cert.final_health {
        if let Some(dot_pos) = key.find('.') {
            let comp = &key[..dot_pos];
            let metric = &key[dot_pos + 1..];

            let health = components_map
                .entry(comp.to_string())
                .or_insert_with(|| ComponentHealth {
                    name: comp.to_string(),
                    grad_norm: None,
                    activation_variance: None,
                    pairwise_cosine: None,
                    status: "ok".into(),
                });

            match metric {
                "grad_norm" => health.grad_norm = Some(value),
                "activation_variance" => health.activation_variance = Some(value),
                "pairwise_cosine" => health.pairwise_cosine = Some(value),
                _ => {}
            }
        }
    }

    // Determine status for each component
    for health in components_map.values_mut() {
        if let Some(gn) = health.grad_norm {
            if gn < 1e-6 {
                health.status = "warning: near-zero gradient".into();
            }
        }
        if let Some(cos) = health.pairwise_cosine {
            if cos > 0.95 {
                health.status = "warning: high cosine similarity".into();
            }
        }
    }

    let overall_loss = cert.final_health.get("loss").copied();

    FinalHealthSnapshot {
        components: components_map.into_values().collect(),
        overall_loss,
    }
}

impl Report {
    /// Render the report as Markdown.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();

        // Executive Summary
        out.push_str("# TransXform Training Report\n\n");
        out.push_str("## Executive Summary\n\n");
        out.push_str(&format!("- **Model:** {}\n", self.executive_summary.model_name));
        out.push_str(&format!("- **Verdict:** {}\n", self.executive_summary.verdict));
        out.push_str(&format!("- **Total Steps:** {}\n", self.executive_summary.total_steps));
        out.push_str(&format!(
            "- **Total Interventions:** {}\n",
            self.executive_summary.total_interventions
        ));
        out.push_str(&format!(
            "- **Phase Regressions:** {}\n",
            self.executive_summary.phase_regressions
        ));

        if !self.executive_summary.near_abort_conditions.is_empty() {
            out.push_str("\n### Near-Abort Conditions\n\n");
            for cond in &self.executive_summary.near_abort_conditions {
                out.push_str(&format!("- {}\n", cond));
            }
        }

        // Intervention Table
        out.push_str("\n## Intervention Table\n\n");
        if self.intervention_table.is_empty() {
            out.push_str("No interventions were necessary.\n");
        } else {
            out.push_str("| Step | Phase | Component | Invariant | Action | Outcome |\n");
            out.push_str("|------|-------|-----------|-----------|--------|---------|\n");
            for row in &self.intervention_table {
                out.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} |\n",
                    row.step, row.phase, row.component, row.invariant, row.action, row.outcome,
                ));
            }
        }

        // Regret Analysis
        out.push_str("\n## Regret Analysis\n\n");
        out.push_str(&format!(
            "- **Cooldown near-misses:** {}\n\n",
            self.regret_analysis.cooldown_near_misses
        ));

        if !self.regret_analysis.likely_necessary.is_empty() {
            out.push_str("### Likely Necessary Interventions\n\n");
            for row in &self.regret_analysis.likely_necessary {
                out.push_str(&format!(
                    "- Step {}: {} on {} (confident)\n",
                    row.step, row.action, row.component
                ));
            }
        }

        if !self.regret_analysis.possibly_preemptive.is_empty() {
            out.push_str("\n### Possibly Preemptive Interventions\n\n");
            for row in &self.regret_analysis.possibly_preemptive {
                out.push_str(&format!(
                    "- Step {}: {} on {} (low confidence)\n",
                    row.step, row.action, row.component
                ));
            }
        }

        // Diagnostic Advisories (V2)
        if self.diagnostic_advisories.total > 0 {
            out.push_str("\n## Diagnostic Advisories\n\n");
            out.push_str(&format!(
                "- **Total:** {} ({} acknowledged, {} unacknowledged)\n\n",
                self.diagnostic_advisories.total,
                self.diagnostic_advisories.acknowledged,
                self.diagnostic_advisories.unacknowledged,
            ));

            for row in &self.diagnostic_advisories.warnings {
                let ack = if row.acknowledged { " [acknowledged]" } else { "" };
                out.push_str(&format!(
                    "### Step {} — {}{}\n\n",
                    row.step, row.signal, ack,
                ));
                out.push_str(&format!(
                    "> {}\n\n",
                    row.summary,
                ));
                out.push_str(&format!("Confidence: {:.0}%\n\n", row.confidence * 100.0));
                if !row.evidence.is_empty() {
                    out.push_str("Evidence:\n");
                    for ev in &row.evidence {
                        out.push_str(&format!("- {}\n", ev));
                    }
                    out.push_str("\n");
                }
            }
        }

        // Final Health
        out.push_str("\n## Final Health Snapshot\n\n");
        if let Some(loss) = self.final_health.overall_loss {
            out.push_str(&format!("- **Final Loss:** {:.6}\n\n", loss));
        }

        if !self.final_health.components.is_empty() {
            out.push_str("| Component | Grad Norm | Variance | Cosine | Status |\n");
            out.push_str("|-----------|-----------|----------|--------|--------|\n");
            for comp in &self.final_health.components {
                out.push_str(&format!(
                    "| {} | {} | {} | {} | {} |\n",
                    comp.name,
                    comp.grad_norm
                        .map(|v| format!("{:.6}", v))
                        .unwrap_or_else(|| "-".into()),
                    comp.activation_variance
                        .map(|v| format!("{:.6}", v))
                        .unwrap_or_else(|| "-".into()),
                    comp.pairwise_cosine
                        .map(|v| format!("{:.6}", v))
                        .unwrap_or_else(|| "-".into()),
                    comp.status,
                ));
            }
        }

        out
    }

    /// Serialize the report as JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        // Build a serializable representation
        let json = serde_json::json!({
            "executive_summary": {
                "model_name": self.executive_summary.model_name,
                "verdict": format!("{}", self.executive_summary.verdict),
                "total_steps": self.executive_summary.total_steps,
                "total_interventions": self.executive_summary.total_interventions,
                "phase_regressions": self.executive_summary.phase_regressions,
                "near_abort_conditions": self.executive_summary.near_abort_conditions,
            },
            "intervention_table": self.intervention_table.iter().map(|row| {
                serde_json::json!({
                    "step": row.step,
                    "phase": format!("{}", row.phase),
                    "component": row.component,
                    "invariant": row.invariant,
                    "observed": row.observed,
                    "threshold": row.threshold,
                    "action": row.action,
                    "outcome": row.outcome,
                    "recovery_steps": row.recovery_steps,
                })
            }).collect::<Vec<_>>(),
            "regret_analysis": {
                "likely_necessary": self.regret_analysis.likely_necessary.len(),
                "possibly_preemptive": self.regret_analysis.possibly_preemptive.len(),
                "cooldown_near_misses": self.regret_analysis.cooldown_near_misses,
            },
            "final_health": {
                "overall_loss": self.final_health.overall_loss,
                "components": self.final_health.components.iter().map(|c| {
                    serde_json::json!({
                        "name": c.name,
                        "grad_norm": c.grad_norm,
                        "activation_variance": c.activation_variance,
                        "pairwise_cosine": c.pairwise_cosine,
                        "status": c.status,
                    })
                }).collect::<Vec<_>>(),
            },
            "diagnostic_advisories": {
                "total": self.diagnostic_advisories.total,
                "acknowledged": self.diagnostic_advisories.acknowledged,
                "unacknowledged": self.diagnostic_advisories.unacknowledged,
                "warnings": self.diagnostic_advisories.warnings.iter().map(|w| {
                    serde_json::json!({
                        "step": w.step,
                        "signal": w.signal,
                        "summary": w.summary,
                        "evidence": w.evidence,
                        "confidence": w.confidence,
                        "acknowledged": w.acknowledged,
                    })
                }).collect::<Vec<_>>(),
            },
        });

        serde_json::to_string_pretty(&json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::BoundaryLedger;

    #[test]
    fn generates_report_for_healthy_run() {
        let ledger = BoundaryLedger::new();
        let mut final_metrics = MetricSnapshot::new();
        final_metrics.insert("loss".into(), 1.5);
        final_metrics.insert("head.grad_norm".into(), 0.05);

        let cert = ledger.emit_certificate("test_model", 1000, &final_metrics, &[]);
        let report = generate_report(&cert, &ledger);

        assert_eq!(report.executive_summary.model_name, "test_model");
        assert_eq!(report.executive_summary.verdict, HealthVerdict::Healthy);
        assert!(report.intervention_table.is_empty());

        let md = report.to_markdown();
        assert!(md.contains("TransXform Training Report"));
        assert!(md.contains("HEALTHY"));

        let json = report.to_json().unwrap();
        assert!(json.contains("test_model"));
    }

    #[test]
    fn generates_report_with_interventions() {
        let mut ledger = BoundaryLedger::new();
        let v = Violation {
            invariant_name: "pairwise_cosine".into(),
            component: "head".into(),
            severity: Severity::Hard,
            observed: 0.99,
            threshold: 0.95,
            direction: ThresholdDirection::Max,
            step: 100,
            passive: false,
        };
        let action = Action::Reinitialize {
            component: "head".into(),
        };
        ledger.record(100, Phase::RepresentationFormation, &v, &action, "test".into());

        let cert = ledger.emit_certificate("model", 500, &HashMap::new(), &[]);
        let report = generate_report(&cert, &ledger);

        assert_eq!(report.intervention_table.len(), 1);
        assert_eq!(report.intervention_table[0].step, 100);

        let md = report.to_markdown();
        assert!(md.contains("head"));
        assert!(md.contains("reinitialize"));
    }
}
