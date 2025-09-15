"""
Training result types and data structures for grammar correction fine-tuning.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from transformers.trainer_utils import TrainOutput
from unsloth import FastModel
from transformers import PreTrainedTokenizer


@dataclass
class TrainingResults:
    """Results returned from the training function."""
    model: FastModel
    tokenizer: PreTrainedTokenizer
    trainer_stats: TrainOutput
    best_eval_loss: float
    best_eval_accuracy: Optional[float]
    training_history: Dict[str, List[float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excluding model/tokenizer)."""
        return {
            "trainer_stats": self.trainer_stats.__dict__ if hasattr(self.trainer_stats, '__dict__') else str(self.trainer_stats),
            "best_eval_loss": self.best_eval_loss,
            "best_eval_accuracy": self.best_eval_accuracy,
            "training_history": self.training_history
        }