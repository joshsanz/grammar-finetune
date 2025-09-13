"""
Optuna callback for hyperparameter optimization with pruning support.
"""

import optuna
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OptunaCallback(TrainerCallback):
    """
    Callback for reporting intermediate results to Optuna for pruning decisions.
    
    This callback integrates with Optuna trials to:
    1. Report intermediate evaluation metrics (loss and accuracy) 
    2. Check for pruning decisions after each evaluation
    3. Raise TrialPruned exception if trial should be stopped early
    """
    
    def __init__(self, trial: optuna.Trial, metric_for_best_model: str = "eval_loss"):
        """
        Initialize the Optuna callback.
        
        Args:
            trial: Optuna trial object for reporting and pruning
            metric_for_best_model: Primary metric to report to Optuna ("eval_loss" or "eval_accuracy")
        """
        self.trial = trial
        self.metric_for_best_model = metric_for_best_model
        self.eval_step = 0
        
        # Store training history for later analysis
        self.history = {
            "eval_loss": [],
            "eval_accuracy": [],
            "train_loss": []
        }
        
        logger.info(f"OptunaCallback initialized with metric: {metric_for_best_model}")
    
    def on_evaluate(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        model=None, 
        tokenizer=None, 
        **kwargs
    ):
        """
        Called after each evaluation phase.
        Reports intermediate results to Optuna and checks for pruning.
        """
        if state.log_history:
            # Get the latest evaluation metrics
            latest_log = state.log_history[-1]
            
            eval_loss = latest_log.get("eval_loss")
            eval_accuracy = latest_log.get("eval_accuracy")
            
            # Store in history
            if eval_loss is not None:
                self.history["eval_loss"].append(eval_loss)
            if eval_accuracy is not None:
                self.history["eval_accuracy"].append(eval_accuracy)
            
            # Determine which metric to report based on configuration
            if self.metric_for_best_model == "eval_loss" and eval_loss is not None:
                metric_value = eval_loss
                logger.info(f"Reporting eval_loss={metric_value} to Optuna at step {self.eval_step}")
            elif self.metric_for_best_model == "eval_accuracy" and eval_accuracy is not None:
                metric_value = eval_accuracy
                logger.info(f"Reporting eval_accuracy={metric_value} to Optuna at step {self.eval_step}")
            else:
                logger.warning(f"Metric {self.metric_for_best_model} not found in evaluation results")
                return
            
            # Report to Optuna
            try:
                self.trial.report(metric_value, self.eval_step)
                
                # Check if trial should be pruned
                if self.trial.should_prune():
                    logger.info(f"Trial {self.trial.number} pruned at step {self.eval_step}")
                    # Set control to stop training
                    control.should_training_stop = True
                    # Raise the pruning exception that Optuna expects
                    raise optuna.TrialPruned()
                    
            except optuna.TrialPruned:
                # Re-raise pruning exception
                raise
            except Exception as e:
                logger.error(f"Error reporting to Optuna: {e}")
            
            self.eval_step += 1
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState, 
        control: TrainerControl,
        model=None,
        tokenizer=None,
        **kwargs
    ):
        """
        Called when logging training metrics.
        Store training loss for history tracking.
        """
        if state.log_history:
            latest_log = state.log_history[-1]
            train_loss = latest_log.get("train_loss")
            if train_loss is not None:
                self.history["train_loss"].append(train_loss)
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Get the complete training history for this trial.
        
        Returns:
            Dictionary containing training metrics history
        """
        return {
            "eval_loss_history": self.history["eval_loss"],
            "eval_accuracy_history": self.history["eval_accuracy"], 
            "train_loss_history": self.history["train_loss"],
            "total_eval_steps": self.eval_step,
            "best_eval_loss": min(self.history["eval_loss"]) if self.history["eval_loss"] else None,
            "best_eval_accuracy": max(self.history["eval_accuracy"]) if self.history["eval_accuracy"] else None
        }