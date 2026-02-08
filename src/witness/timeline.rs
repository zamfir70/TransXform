use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Sparkline};

use crate::supervisor::SupervisorReport;

/// Render the health timeline: X=steps, Y=normalized health bands.
pub fn render_timeline(frame: &mut Frame, area: Rect, reports: &[SupervisorReport]) {
    let block = Block::default()
        .title(" Health Timeline ")
        .borders(Borders::ALL);

    if reports.is_empty() {
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let msg = ratatui::widgets::Paragraph::new("Waiting for data...")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center);
        frame.render_widget(msg, inner);
        return;
    }

    let inner = block.inner(area);
    frame.render_widget(block, inner_block_area(area));

    // Split inner area into 4 bands (one per metric category)
    let bands = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
        ])
        .split(inner);

    // Collect metric time series from reports
    let loss_data = extract_metric(reports, "loss");
    let grad_data = extract_metric(reports, "head.grad_norm");
    let cosine_data = extract_metric(reports, "head.pairwise_cosine");
    let variance_data = extract_metric(reports, "backbone.activation_variance");

    render_band(frame, bands[0], "Loss", &loss_data, Color::Yellow);
    render_band(frame, bands[1], "Grad Norm", &grad_data, Color::Cyan);
    render_band(frame, bands[2], "Cosine Sim", &cosine_data, Color::Red);
    render_band(frame, bands[3], "Variance", &variance_data, Color::Green);
}

fn render_band(frame: &mut Frame, area: Rect, label: &str, data: &[u64], color: Color) {
    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .title(format!(" {} ", label))
                .borders(Borders::LEFT),
        )
        .data(data)
        .style(Style::default().fg(color));
    frame.render_widget(sparkline, area);
}

fn extract_metric(reports: &[SupervisorReport], key: &str) -> Vec<u64> {
    reports
        .iter()
        .map(|r| {
            let val = r.metrics.get(key).copied().unwrap_or(0.0);
            // Normalize to 0-100 for sparkline (which expects u64)
            (val.abs().min(100.0) * 100.0) as u64
        })
        .collect()
}

fn inner_block_area(area: Rect) -> Rect {
    area
}
