use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::error::TransXformError;
use crate::types::*;

/// A single ledger entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub step: u64,
    pub timestamp: chrono::DateTime<Utc>,
    pub phase: Phase,
    pub component: String,
    pub invariant: String,
    pub metric_snapshot: MetricSnapshot,
    pub action: Action,
    pub justification: String,
    pub outcome: InterventionOutcome,
    pub regret_tag: Option<RegretTag>,
    pub entry_type: LedgerEntryType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LedgerEntryType {
    Violation,
    NearMiss,
    PhaseTransition,
    Abort,
    /// V2: Advisory diagnostic warning (non-authoritative).
    Advisory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionOutcome {
    Pending,
    Recovered,
    Persisted,
    Worsened,
}

/// Training health certificate (whitepaper §10.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCertificate {
    pub model_name: String,
    pub total_steps: u64,
    pub start_time: chrono::DateTime<Utc>,
    pub end_time: chrono::DateTime<Utc>,
    pub verdict: HealthVerdict,
    pub invariant_compliance: HashMap<String, InvariantCompliance>,
    pub intervention_summary: InterventionSummary,
    pub phase_trace: Vec<PhaseTransition>,
    pub final_health: MetricSnapshot,
    pub regret_summary: RegretSummary,
    /// V2: Diagnostic warning summary (advisory, non-authoritative).
    #[serde(default)]
    pub diagnostic_summary: DiagnosticSummary,
}

/// V2: Summary of diagnostic advisories for the training certificate.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagnosticSummary {
    pub total_warnings: u64,
    pub acknowledged: u64,
    pub unacknowledged: u64,
    pub by_signal: HashMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantCompliance {
    pub invariant_name: String,
    pub total_checks: u64,
    pub violations: u64,
    pub compliance_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionSummary {
    pub total_hard: u64,
    pub total_soft: u64,
    pub by_component: HashMap<String, u64>,
    pub by_action: HashMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegretSummary {
    pub total_assessed: u64,
    pub confident: u64,
    pub low_confidence: u64,
    pub near_misses: u64,
}

/// Append-only audit log (whitepaper §10).
pub struct BoundaryLedger {
    entries: Vec<LedgerEntry>,
    start_time: chrono::DateTime<Utc>,
    invariant_check_counts: HashMap<String, u64>,
    invariant_violation_counts: HashMap<String, u64>,
}

impl BoundaryLedger {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            start_time: Utc::now(),
            invariant_check_counts: HashMap::new(),
            invariant_violation_counts: HashMap::new(),
        }
    }

    /// Record a violation and its intervention.
    pub fn record(
        &mut self,
        step: u64,
        phase: Phase,
        violation: &Violation,
        action: &Action,
        justification: String,
    ) {
        *self
            .invariant_violation_counts
            .entry(violation.invariant_name.clone())
            .or_insert(0) += 1;

        self.entries.push(LedgerEntry {
            step,
            timestamp: Utc::now(),
            phase,
            component: violation.component.clone(),
            invariant: violation.invariant_name.clone(),
            metric_snapshot: HashMap::new(), // caller can enrich
            action: action.clone(),
            justification,
            outcome: InterventionOutcome::Pending,
            regret_tag: None,
            entry_type: LedgerEntryType::Violation,
        });
    }

    /// Record a violation with a full metric snapshot.
    pub fn record_with_snapshot(
        &mut self,
        step: u64,
        phase: Phase,
        violation: &Violation,
        action: &Action,
        justification: String,
        snapshot: MetricSnapshot,
    ) {
        *self
            .invariant_violation_counts
            .entry(violation.invariant_name.clone())
            .or_insert(0) += 1;

        self.entries.push(LedgerEntry {
            step,
            timestamp: Utc::now(),
            phase,
            component: violation.component.clone(),
            invariant: violation.invariant_name.clone(),
            metric_snapshot: snapshot,
            action: action.clone(),
            justification,
            outcome: InterventionOutcome::Pending,
            regret_tag: None,
            entry_type: LedgerEntryType::Violation,
        });
    }

    /// Record a near-miss.
    pub fn record_near_miss(&mut self, step: u64, phase: Phase, near_miss: &NearMiss) {
        self.entries.push(LedgerEntry {
            step,
            timestamp: Utc::now(),
            phase,
            component: near_miss.component.clone(),
            invariant: near_miss.invariant_name.clone(),
            metric_snapshot: near_miss.metric_snapshot.clone(),
            action: Action::Abort {
                reason: "near_miss (no action taken)".into(),
            },
            justification: format!(
                "Near-miss: observed={:.6}, hard_threshold={:.6}, margin={:.6}",
                near_miss.observed, near_miss.hard_threshold, near_miss.margin
            ),
            outcome: InterventionOutcome::Pending,
            regret_tag: None,
            entry_type: LedgerEntryType::NearMiss,
        });
    }

    /// Record an advisory diagnostic warning (V2). No intervention, no authority.
    pub fn record_advisory(
        &mut self,
        step: u64,
        phase: Phase,
        signal_name: &str,
        summary: &str,
    ) {
        self.entries.push(LedgerEntry {
            step,
            timestamp: Utc::now(),
            phase,
            component: "diagnostic".into(),
            invariant: signal_name.to_string(),
            metric_snapshot: HashMap::new(),
            action: Action::Abort {
                reason: format!("advisory: {}", signal_name),
            },
            justification: summary.to_string(),
            outcome: InterventionOutcome::Pending,
            regret_tag: None,
            entry_type: LedgerEntryType::Advisory,
        });
    }

    /// Record a phase transition.
    pub fn record_phase_transition(&mut self, step: u64, transition: &PhaseTransition) {
        self.entries.push(LedgerEntry {
            step,
            timestamp: Utc::now(),
            phase: transition.to,
            component: "system".into(),
            invariant: "phase_transition".into(),
            metric_snapshot: HashMap::new(),
            action: Action::Abort {
                reason: format!("{} → {}", transition.from, transition.to),
            },
            justification: transition.reason.clone(),
            outcome: InterventionOutcome::Pending,
            regret_tag: None,
            entry_type: LedgerEntryType::PhaseTransition,
        });
    }

    /// Update an entry's outcome and regret tag.
    pub fn update_outcome(
        &mut self,
        intervention_step: u64,
        component: &str,
        outcome: InterventionOutcome,
        regret_tag: RegretTag,
    ) {
        for entry in self.entries.iter_mut().rev() {
            if entry.step == intervention_step
                && entry.component == component
                && matches!(entry.entry_type, LedgerEntryType::Violation)
            {
                entry.outcome = outcome;
                entry.regret_tag = Some(regret_tag);
                return;
            }
        }
    }

    /// Record that an invariant was checked (for compliance rate calculation).
    pub fn record_check(&mut self, invariant_name: &str) {
        *self
            .invariant_check_counts
            .entry(invariant_name.to_string())
            .or_insert(0) += 1;
    }

    /// Emit the training certificate.
    pub fn emit_certificate(
        &self,
        model_name: &str,
        total_steps: u64,
        final_metrics: &MetricSnapshot,
        phase_trace: &[PhaseTransition],
    ) -> TrainingCertificate {
        self.emit_certificate_with_diagnostics(
            model_name,
            total_steps,
            final_metrics,
            phase_trace,
            DiagnosticSummary::default(),
        )
    }

    /// Emit the training certificate with V2 diagnostic summary.
    pub fn emit_certificate_with_diagnostics(
        &self,
        model_name: &str,
        total_steps: u64,
        final_metrics: &MetricSnapshot,
        phase_trace: &[PhaseTransition],
        diagnostic_summary: DiagnosticSummary,
    ) -> TrainingCertificate {
        let violation_entries: Vec<&LedgerEntry> = self
            .entries
            .iter()
            .filter(|e| matches!(e.entry_type, LedgerEntryType::Violation))
            .collect();

        let total_hard = violation_entries
            .iter()
            .filter(|e| matches!(e.action, Action::Reinitialize { .. } | Action::InjectNoise { .. }))
            .count() as u64;

        let total_soft = violation_entries.len() as u64 - total_hard;

        // Build intervention summary
        let mut by_component: HashMap<String, u64> = HashMap::new();
        let mut by_action: HashMap<String, u64> = HashMap::new();
        for entry in &violation_entries {
            *by_component.entry(entry.component.clone()).or_insert(0) += 1;
            *by_action
                .entry(entry.action.action_name().to_string())
                .or_insert(0) += 1;
        }

        // Build invariant compliance
        let mut compliance = HashMap::new();
        for (name, &checks) in &self.invariant_check_counts {
            let violations = self.invariant_violation_counts.get(name).copied().unwrap_or(0);
            let rate = if checks > 0 {
                1.0 - (violations as f64 / checks as f64)
            } else {
                1.0
            };
            compliance.insert(
                name.clone(),
                InvariantCompliance {
                    invariant_name: name.clone(),
                    total_checks: checks,
                    violations,
                    compliance_rate: rate,
                },
            );
        }

        // Build regret summary
        let regret_assessed: Vec<&LedgerEntry> = self
            .entries
            .iter()
            .filter(|e| e.regret_tag.is_some())
            .collect();
        let confident = regret_assessed
            .iter()
            .filter(|e| e.regret_tag == Some(RegretTag::Confident))
            .count() as u64;
        let low_confidence = regret_assessed
            .iter()
            .filter(|e| e.regret_tag == Some(RegretTag::LowConfidence))
            .count() as u64;
        let near_miss_count = self
            .entries
            .iter()
            .filter(|e| matches!(e.entry_type, LedgerEntryType::NearMiss))
            .count() as u64;

        // Determine verdict
        let has_unresolved = violation_entries
            .iter()
            .any(|e| matches!(e.outcome, InterventionOutcome::Persisted | InterventionOutcome::Worsened));

        let verdict = if violation_entries.is_empty() {
            HealthVerdict::Healthy
        } else if has_unresolved {
            HealthVerdict::Compromised {
                details: "Unresolved violations at training end".into(),
            }
        } else {
            HealthVerdict::Recovered {
                intervention_count: violation_entries.len() as u64,
            }
        };

        TrainingCertificate {
            model_name: model_name.to_string(),
            total_steps,
            start_time: self.start_time,
            end_time: Utc::now(),
            verdict,
            invariant_compliance: compliance,
            intervention_summary: InterventionSummary {
                total_hard,
                total_soft,
                by_component,
                by_action,
            },
            phase_trace: phase_trace.to_vec(),
            final_health: final_metrics.clone(),
            regret_summary: RegretSummary {
                total_assessed: regret_assessed.len() as u64,
                confident,
                low_confidence,
                near_misses: near_miss_count,
            },
            diagnostic_summary,
        }
    }

    /// Serialize the full ledger to JSON.
    pub fn to_json(&self) -> Result<String, TransXformError> {
        Ok(serde_json::to_string_pretty(&self.entries)?)
    }

    /// Get the last entry.
    pub fn last_entry(&self) -> Option<&LedgerEntry> {
        self.entries.last()
    }

    /// Get all entries.
    pub fn entries(&self) -> &[LedgerEntry] {
        &self.entries
    }

    /// Get entries for a specific component.
    pub fn entries_for_component(&self, component: &str) -> Vec<&LedgerEntry> {
        self.entries
            .iter()
            .filter(|e| e.component == component)
            .collect()
    }

    /// Count violation entries for a component in a phase.
    pub fn violation_count(&self, component: &str, phase: Phase) -> u64 {
        self.entries
            .iter()
            .filter(|e| {
                e.component == component
                    && e.phase == phase
                    && matches!(e.entry_type, LedgerEntryType::Violation)
            })
            .count() as u64
    }
}

impl Default for BoundaryLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_violation() -> Violation {
        Violation {
            invariant_name: "pairwise_cosine".into(),
            component: "head".into(),
            severity: Severity::Hard,
            observed: 0.99,
            threshold: 0.95,
            direction: ThresholdDirection::Max,
            step: 100,
        }
    }

    #[test]
    fn record_and_retrieve() {
        let mut ledger = BoundaryLedger::new();
        let v = test_violation();
        let action = Action::Reinitialize {
            component: "head".into(),
        };

        ledger.record(100, Phase::RepresentationFormation, &v, &action, "test".into());

        assert_eq!(ledger.entries().len(), 1);
        assert_eq!(ledger.last_entry().unwrap().step, 100);
    }

    #[test]
    fn filter_by_component() {
        let mut ledger = BoundaryLedger::new();
        let v1 = test_violation();
        let v2 = Violation {
            component: "backbone".into(),
            ..test_violation()
        };

        let action = Action::Reinitialize {
            component: "head".into(),
        };
        ledger.record(100, Phase::Bootstrap, &v1, &action, "a".into());
        ledger.record(200, Phase::Bootstrap, &v2, &action, "b".into());

        assert_eq!(ledger.entries_for_component("head").len(), 1);
        assert_eq!(ledger.entries_for_component("backbone").len(), 1);
    }

    #[test]
    fn healthy_certificate() {
        let ledger = BoundaryLedger::new();
        let cert = ledger.emit_certificate("test", 1000, &HashMap::new(), &[]);
        assert_eq!(cert.verdict, HealthVerdict::Healthy);
    }

    #[test]
    fn recovered_certificate() {
        let mut ledger = BoundaryLedger::new();
        let v = test_violation();
        let action = Action::Reinitialize {
            component: "head".into(),
        };
        ledger.record(100, Phase::Bootstrap, &v, &action, "test".into());
        // Mark as recovered
        ledger.update_outcome(100, "head", InterventionOutcome::Recovered, RegretTag::Confident);

        let cert = ledger.emit_certificate("test", 1000, &HashMap::new(), &[]);
        assert!(matches!(cert.verdict, HealthVerdict::Recovered { .. }));
    }

    #[test]
    fn json_roundtrip() {
        let mut ledger = BoundaryLedger::new();
        let v = test_violation();
        let action = Action::Reinitialize {
            component: "head".into(),
        };
        ledger.record(100, Phase::Bootstrap, &v, &action, "test".into());

        let json = ledger.to_json().unwrap();
        let parsed: Vec<LedgerEntry> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].step, 100);
    }
}
