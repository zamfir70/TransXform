use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::supervisor::SupervisorReport;

/// Render intervention markers and optional detail pane.
pub fn render_markers(
    frame: &mut Frame,
    area: Rect,
    reports: &[SupervisorReport],
    selected: Option<usize>,
) {
    let block = Block::default()
        .title(" Interventions ")
        .borders(Borders::ALL);

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Collect all reports that have interventions
    let intervention_reports: Vec<&SupervisorReport> = reports
        .iter()
        .filter(|r| !r.actions_taken.is_empty())
        .collect();

    if intervention_reports.is_empty() {
        let msg = Paragraph::new("No interventions taken")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);
        frame.render_widget(msg, inner);
        return;
    }

    // If a marker is selected, show detail
    if let Some(idx) = selected {
        if let Some(report) = intervention_reports.get(idx) {
            let detail = format!(
                "Step {} | Phase: {} | Actions: {}\nViolations: {}",
                report.step,
                report.phase,
                report
                    .actions_taken
                    .iter()
                    .map(|a| format!("{}", a))
                    .collect::<Vec<_>>()
                    .join(", "),
                report
                    .violations
                    .iter()
                    .map(|v| format!("{}", v))
                    .collect::<Vec<_>>()
                    .join("; "),
            );
            let para = Paragraph::new(detail)
                .style(Style::default().fg(Color::White))
                .wrap(ratatui::widgets::Wrap { trim: true });
            frame.render_widget(para, inner);
            return;
        }
    }

    // Summary view: show counts
    let summary = format!(
        "{} interventions across {} steps | Arrow keys to browse, Esc to deselect",
        intervention_reports
            .iter()
            .map(|r| r.actions_taken.len())
            .sum::<usize>(),
        intervention_reports.len(),
    );
    let para = Paragraph::new(summary)
        .style(Style::default().fg(Color::Cyan))
        .alignment(Alignment::Center);
    frame.render_widget(para, inner);
}
