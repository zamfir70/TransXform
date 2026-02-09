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

    // Auto-discover metric keys from the first report
    let (loss_key, grad_key, cosine_key, var_key) = discover_metric_keys(reports);

    let loss_data = extract_metric(reports, &loss_key);
    let grad_data = extract_metric(reports, &grad_key);
    let cosine_data = extract_metric(reports, &cosine_key);
    let variance_data = extract_metric(reports, &var_key);

    render_band(frame, bands[0], "Loss", &loss_data, Color::Yellow);
    render_band(frame, bands[1], "Grad Norm", &grad_data, Color::Cyan);
    render_band(frame, bands[2], "Cosine Sim", &cosine_data, Color::Red);
    render_band(frame, bands[3], "Variance", &variance_data, Color::Green);
}

/// Discover metric keys from reports by pattern matching.
///
/// Looks for keys containing known substrings (loss, grad, cosine, variance).
/// Falls back to alphabetically sorted keys if patterns don't match.
fn discover_metric_keys(reports: &[SupervisorReport]) -> (String, String, String, String) {
    let first = &reports[0].metrics;
    let keys: Vec<&String> = first.keys().collect();

    let find = |patterns: &[&str], fallback: &str| -> String {
        for pat in patterns {
            if let Some(key) = keys.iter().find(|k| k.to_lowercase().contains(pat)) {
                return key.to_string();
            }
        }
        fallback.to_string()
    };

    let loss_key = find(&["loss_explosion", "loss"], "loss_explosion_factor");
    let grad_key = find(
        &["backbone.grad_norm", "grad_norm"],
        "backbone.grad_norm_min",
    );
    let cosine_key = find(
        &["pairwise_cosine", "cosine"],
        "structure_head.pairwise_cosine",
    );
    let var_key = find(
        &["backbone.activation_variance", "activation_variance", "variance"],
        "backbone.activation_variance_min",
    );

    (loss_key, grad_key, cosine_key, var_key)
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
    // Find the max value for normalization (sparklines look better with full range)
    let max_val = reports
        .iter()
        .map(|r| r.metrics.get(key).copied().unwrap_or(0.0).abs())
        .fold(0.0_f64, f64::max)
        .max(1e-8); // avoid division by zero

    reports
        .iter()
        .map(|r| {
            let val = r.metrics.get(key).copied().unwrap_or(0.0).abs();
            // Normalize to 0-100 range relative to observed max
            ((val / max_val) * 100.0) as u64
        })
        .collect()
}

fn inner_block_area(area: Rect) -> Rect {
    area
}
