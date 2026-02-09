use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;

use crate::supervisor::SupervisorReport;

use super::markers::render_markers;
use super::phase_strip::render_phase_strip;
use super::timeline::render_timeline;

/// State for the Witness Console TUI.
pub struct WitnessApp {
    rx: mpsc::Receiver<SupervisorReport>,
    reports: Vec<SupervisorReport>,
    selected_marker: Option<usize>,
    scroll_offset: usize,
}

impl WitnessApp {
    /// Create a new WitnessApp and return the sender for feeding reports.
    pub fn new() -> (Self, mpsc::Sender<SupervisorReport>) {
        let (tx, rx) = mpsc::channel();
        (
            Self {
                rx,
                reports: Vec::new(),
                selected_marker: None,
                scroll_offset: 0,
            },
            tx,
        )
    }

    /// Spawn the TUI on a background thread.
    ///
    /// Returns a `JoinHandle` that resolves when the user quits (pressing 'q').
    pub fn spawn(mut self) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            if let Err(e) = self.run() {
                eprintln!("Witness console error: {}", e);
            }
        })
    }

    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        loop {
            // Drain any pending reports
            while let Ok(report) = self.rx.try_recv() {
                self.reports.push(report);
            }

            // Draw
            terminal.draw(|frame| self.render(frame))?;

            // Handle input (non-blocking with 100ms timeout)
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            break
                        }
                        KeyCode::Left => {
                            if let Some(ref mut idx) = self.selected_marker {
                                *idx = idx.saturating_sub(1);
                            }
                        }
                        KeyCode::Right => {
                            if let Some(ref mut idx) = self.selected_marker {
                                let max = self
                                    .reports
                                    .iter()
                                    .filter(|r| !r.actions_taken.is_empty())
                                    .count();
                                if *idx + 1 < max {
                                    *idx += 1;
                                }
                            } else if self.reports.iter().any(|r| !r.actions_taken.is_empty()) {
                                self.selected_marker = Some(0);
                            }
                        }
                        KeyCode::Esc => {
                            self.selected_marker = None;
                        }
                        _ => {}
                    }
                }
            }
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        Ok(())
    }

    fn render(&self, frame: &mut Frame) {
        let area = frame.area();

        // Layout: 3 rows
        // [Phase Strip - 3 lines]
        // [Health Timeline - flex]
        // [Intervention Markers / Detail - 8 lines]
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(8),
            ])
            .split(area);

        // Header: phase strip
        render_phase_strip(frame, chunks[0], &self.reports);

        // Main area: health timeline
        render_timeline(frame, chunks[1], &self.reports);

        // Footer: intervention markers
        render_markers(frame, chunks[2], &self.reports, self.selected_marker);
    }
}
