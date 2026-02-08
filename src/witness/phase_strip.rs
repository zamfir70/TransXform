use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::supervisor::SupervisorReport;
use crate::types::Phase;

/// Render the phase strip — a thin horizontal band showing phase progression.
pub fn render_phase_strip(
    frame: &mut Frame,
    area: Rect,
    reports: &[SupervisorReport],
) {
    let block = Block::default()
        .title(" TransXform Witness Console ")
        .borders(Borders::ALL);

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if reports.is_empty() {
        let msg = Paragraph::new("Awaiting first step...")
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(msg, inner);
        return;
    }

    let current = reports.last().unwrap();
    let phase_color = match current.phase {
        Phase::Bootstrap => Color::Blue,
        Phase::RepresentationFormation => Color::Cyan,
        Phase::Stabilization => Color::Green,
        Phase::Refinement => Color::Yellow,
        Phase::Aborted => Color::Red,
    };

    let violations_count: usize = current.violations.len();
    let status = if violations_count > 0 {
        format!(
            "Step {} | {} | {} violations | {} actions",
            current.step,
            current.phase,
            violations_count,
            current.actions_taken.len(),
        )
    } else {
        format!(
            "Step {} | {} | healthy",
            current.step, current.phase,
        )
    };

    let para = Paragraph::new(status)
        .style(Style::default().fg(phase_color).bold());
    frame.render_widget(para, inner);
}
