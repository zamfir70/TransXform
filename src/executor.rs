use std::cell::RefCell;
use std::rc::Rc;

use crate::error::TransXformError;
use crate::model::Model;
use crate::types::Action;

/// Dispatches intervention actions to the model (whitepaper §7).
///
/// The executor is stateless — all decision logic lives in `ControlLaws`.
/// It borrows the model mutably only for the duration of each intervention.
pub struct InterventionExecutor<M: Model> {
    model: Rc<RefCell<M>>,
}

impl<M: Model> InterventionExecutor<M> {
    pub fn new(model: Rc<RefCell<M>>) -> Self {
        Self { model }
    }

    /// Execute an action against the model.
    ///
    /// For `Action::Abort`, this does not touch the model — it returns an error
    /// that the supervisor handles.
    pub fn execute(&self, action: &Action) -> Result<(), TransXformError> {
        match action {
            Action::Reinitialize { component } => {
                log::info!("Executing: reinitialize({})", component);
                self.model.borrow_mut().reinitialize(component)?;
                // Zero optimizer momentum/variance to prevent corpses (§2.1).
                // Best-effort: if the model doesn't support this, the reinit
                // still succeeded — just log and continue.
                if let Err(e) = self.model.borrow_mut().reset_optimizer_state(component) {
                    log::warn!(
                        "Optimizer state reset failed for {}: {} (reinit still applied)",
                        component, e
                    );
                }
                Ok(())
            }
            Action::Freeze { component } => {
                log::info!("Executing: freeze({})", component);
                self.model.borrow_mut().freeze(component)
            }
            Action::Unfreeze { component } => {
                log::info!("Executing: unfreeze({})", component);
                self.model.borrow_mut().unfreeze(component)
            }
            Action::Rescale { component, factor } => {
                log::info!("Executing: rescale({}, {:.4})", component, factor);
                self.model.borrow_mut().rescale(component, *factor)
            }
            Action::InjectNoise {
                component,
                magnitude,
            } => {
                log::info!("Executing: inject_noise({}, {:.6})", component, magnitude);
                self.model.borrow_mut().inject_noise(component, *magnitude)
            }
            Action::AdjustLr { component, factor } => {
                log::info!("Executing: adjust_lr({}, {:.4})", component, factor);
                self.model.borrow_mut().adjust_lr(component, *factor)
            }
            Action::Abort { reason } => {
                log::warn!("Training abort requested: {}", reason);
                Err(TransXformError::TrainingAborted {
                    verdict: crate::types::NegativeVerdict::UnstableArchitecture,
                    details: reason.clone(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MockModel;

    #[test]
    fn execute_reinitialize() {
        let model = Rc::new(RefCell::new(MockModel::new(&["head"])));
        let executor = InterventionExecutor::new(model.clone());

        let action = Action::Reinitialize {
            component: "head".into(),
        };
        executor.execute(&action).unwrap();

        let interventions = model.borrow().interventions().to_vec();
        assert_eq!(interventions.len(), 2);
        assert_eq!(interventions[0], ("head".into(), "reinitialize".into()));
        assert_eq!(interventions[1], ("head".into(), "reset_optimizer_state".into()));
    }

    #[test]
    fn execute_freeze_unfreeze() {
        let model = Rc::new(RefCell::new(MockModel::new(&["backbone"])));
        let executor = InterventionExecutor::new(model.clone());

        executor
            .execute(&Action::Freeze {
                component: "backbone".into(),
            })
            .unwrap();
        assert!(model.borrow().is_frozen("backbone"));

        executor
            .execute(&Action::Unfreeze {
                component: "backbone".into(),
            })
            .unwrap();
        assert!(!model.borrow().is_frozen("backbone"));
    }

    #[test]
    fn execute_abort_returns_error() {
        let model = Rc::new(RefCell::new(MockModel::new(&["head"])));
        let executor = InterventionExecutor::new(model);

        let result = executor.execute(&Action::Abort {
            reason: "test abort".into(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn execute_unknown_component_fails() {
        let model = Rc::new(RefCell::new(MockModel::new(&["head"])));
        let executor = InterventionExecutor::new(model);

        let result = executor.execute(&Action::Reinitialize {
            component: "nonexistent".into(),
        });
        assert!(result.is_err());
    }
}
