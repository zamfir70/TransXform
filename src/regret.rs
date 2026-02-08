use std::collections::VecDeque;

use serde::Serialize;

use crate::types::*;

/// Tracks intervention regret and near-misses (whitepaper §6.4).
pub struct RegretTracker {
    windows: Vec<RegretWindow>,
    near_misses: Vec<NearMiss>,
    regret_window_length: u64,
}

/// An open regret window for a specific intervention.
#[derive(Debug, Clone)]
pub struct RegretWindow {
    pub intervention_step: u64,
    pub component: String,
    pub action: Action,
    pub invariant_name: String,
    pub pre_metric_value: f64,
    pub post_metric_values: Vec<(u64, f64)>,
    pub pre_trajectory: VecDeque<f64>,
    pub tag: RegretTag,
    pub closed: bool,
}

/// The outcome of a regret window evaluation.
#[derive(Debug, Clone, Serialize)]
pub struct RegretAssessment {
    pub intervention_step: u64,
    pub component: String,
    pub action: Action,
    pub recovery_steps: Option<u64>,
    pub post_improvement: f64,
    pub was_recovering: bool,
    pub tag: RegretTag,
}

impl RegretTracker {
    pub fn new(regret_window_length: u64) -> Self {
        Self {
            windows: Vec::new(),
            near_misses: Vec::new(),
            regret_window_length,
        }
    }

    /// Open a new regret window for a hard intervention.
    pub fn open_window(
        &mut self,
        step: u64,
        component: &str,
        action: &Action,
        invariant_name: &str,
        pre_metric_value: f64,
        pre_trajectory: VecDeque<f64>,
    ) {
        self.windows.push(RegretWindow {
            intervention_step: step,
            component: component.to_string(),
            action: action.clone(),
            invariant_name: invariant_name.to_string(),
            pre_metric_value,
            post_metric_values: Vec::new(),
            pre_trajectory,
            tag: RegretTag::Pending,
            closed: false,
        });
    }

    /// Feed new metrics into all open windows. Close and assess expired ones.
    pub fn update(
        &mut self,
        step: u64,
        metrics: &MetricSnapshot,
    ) -> Vec<RegretAssessment> {
        let mut assessments = Vec::new();

        for window in &mut self.windows {
            if window.closed {
                continue;
            }

            // Record the metric value for this window's invariant
            if let Some(&value) = metrics.get(&format!(
                "{}.{}",
                window.component,
                window.invariant_name.split('.').last().unwrap_or(&window.invariant_name)
            )).or_else(|| metrics.get(&window.invariant_name)) {
                window.post_metric_values.push((step, value));
            }

            // Check if window has expired
            if step >= window.intervention_step + self.regret_window_length {
                window.closed = true;
                window.tag = assess_window(window);
                assessments.push(RegretAssessment {
                    intervention_step: window.intervention_step,
                    component: window.component.clone(),
                    action: window.action.clone(),
                    recovery_steps: compute_recovery_steps(window),
                    post_improvement: compute_improvement(window),
                    was_recovering: was_pre_trajectory_recovering(window),
                    tag: window.tag,
                });
            }
        }

        assessments
    }

    /// Record a near-miss.
    pub fn record_near_miss(&mut self, near_miss: NearMiss) {
        self.near_misses.push(near_miss);
    }

    /// Get all near-misses.
    pub fn near_misses(&self) -> &[NearMiss] {
        &self.near_misses
    }

    /// Get all open (pending) windows.
    pub fn open_windows(&self) -> Vec<&RegretWindow> {
        self.windows.iter().filter(|w| !w.closed).collect()
    }

    /// Get all completed assessments from closed windows.
    pub fn completed_assessments(&self) -> Vec<RegretAssessment> {
        self.windows
            .iter()
            .filter(|w| w.closed)
            .map(|w| RegretAssessment {
                intervention_step: w.intervention_step,
                component: w.component.clone(),
                action: w.action.clone(),
                recovery_steps: compute_recovery_steps(w),
                post_improvement: compute_improvement(w),
                was_recovering: was_pre_trajectory_recovering(w),
                tag: w.tag,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Assessment logic
// ---------------------------------------------------------------------------

fn assess_window(window: &RegretWindow) -> RegretTag {
    let improvement = compute_improvement(window);
    let was_recovering = was_pre_trajectory_recovering(window);

    if improvement > 0.1 && !was_recovering {
        RegretTag::Confident
    } else if improvement <= 0.0 && was_recovering {
        RegretTag::LowConfidence
    } else if was_recovering && improvement < 0.05 {
        RegretTag::LowConfidence
    } else {
        RegretTag::Confident
    }
}

/// Count steps until the post-intervention metric returned to a "healthy" level.
fn compute_recovery_steps(window: &RegretWindow) -> Option<u64> {
    // For now, define "recovered" as metric returning to pre-intervention level
    for &(step, value) in &window.post_metric_values {
        // Simple heuristic: if the metric moved back toward the pre-intervention
        // value (or past it), count as recovered.
        let delta_pre = (window.pre_metric_value - value).abs();
        if delta_pre < 0.01 || value.is_sign_positive() != window.pre_metric_value.is_sign_positive() {
            return Some(step - window.intervention_step);
        }
    }
    None
}

/// Compute the improvement delta: average post-metric vs pre-metric.
fn compute_improvement(window: &RegretWindow) -> f64 {
    if window.post_metric_values.is_empty() {
        return 0.0;
    }
    let post_avg: f64 = window.post_metric_values.iter().map(|(_, v)| v).sum::<f64>()
        / window.post_metric_values.len() as f64;
    // Improvement means moving away from the violation threshold
    // This is a simplified heuristic; real implementation would consider direction
    (post_avg - window.pre_metric_value).abs()
}

/// Check if the pre-intervention trajectory was already trending toward recovery.
fn was_pre_trajectory_recovering(window: &RegretWindow) -> bool {
    let traj = &window.pre_trajectory;
    if traj.len() < 3 {
        return false;
    }

    // Simple linear trend: compare first half average to second half average
    let mid = traj.len() / 2;
    let first_half: f64 = traj.iter().take(mid).sum::<f64>() / mid as f64;
    let second_half: f64 = traj.iter().skip(mid).sum::<f64>() / (traj.len() - mid) as f64;

    // If the metric was already moving away from the violation, it was recovering
    let delta = (second_half - first_half).abs();
    delta > 0.01 && second_half.abs() < first_half.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn open_and_close_window() {
        let mut tracker = RegretTracker::new(10);

        tracker.open_window(
            100,
            "head",
            &Action::Reinitialize { component: "head".into() },
            "pairwise_cosine",
            0.99,
            VecDeque::new(),
        );

        assert_eq!(tracker.open_windows().len(), 1);

        // Feed metrics for 10 steps
        for step in 101..=110 {
            let mut metrics = HashMap::new();
            metrics.insert("head.pairwise_cosine".into(), 0.80);
            tracker.update(step, &metrics);
        }

        // Window should be closed now
        assert_eq!(tracker.open_windows().len(), 0);
        assert_eq!(tracker.completed_assessments().len(), 1);
    }

    #[test]
    fn near_miss_recording() {
        let mut tracker = RegretTracker::new(100);
        tracker.record_near_miss(NearMiss {
            step: 50,
            invariant_name: "test".into(),
            component: "head".into(),
            observed: 0.94,
            hard_threshold: 0.95,
            margin: 0.01,
            metric_snapshot: HashMap::new(),
        });
        assert_eq!(tracker.near_misses().len(), 1);
    }

    #[test]
    fn recovering_trajectory_detected() {
        let mut traj = VecDeque::new();
        // Values trending downward (away from a max threshold)
        traj.push_back(0.98);
        traj.push_back(0.96);
        traj.push_back(0.94);
        traj.push_back(0.92);
        traj.push_back(0.90);
        traj.push_back(0.88);

        let window = RegretWindow {
            intervention_step: 100,
            component: "head".into(),
            action: Action::Reinitialize { component: "head".into() },
            invariant_name: "cosine".into(),
            pre_metric_value: 0.88,
            post_metric_values: vec![],
            pre_trajectory: traj,
            tag: RegretTag::Pending,
            closed: false,
        };

        assert!(was_pre_trajectory_recovering(&window));
    }
}
