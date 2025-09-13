"""
Custom evaluation metrics for grammar correction hyperparameter tuning.
"""

import numpy as np
from transformers import EvalPrediction
import logging

logger = logging.getLogger(__name__)

def compute_accuracy(eval_pred: EvalPrediction, tokenizer=None) -> dict:
    """
    Compute exact match accuracy for grammar correction task.

    Args:
        eval_pred: Evaluation prediction from transformers trainer
        tokenizer: Tokenizer for decoding (if needed)

    Returns:
        dict: Dictionary with accuracy metric
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # For generation tasks, predictions are token IDs
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Decode predictions and labels
    decoded_preds = []
    decoded_labels = []

    for pred, label in zip(predictions, labels):
        # Filter out padding tokens (-100)
        pred = pred[label != -100]
        label = label[label != -100]

        # Decode to text
        if tokenizer:
            pred_text = tokenizer.decode(pred, skip_special_tokens=True)
            label_text = tokenizer.decode(label, skip_special_tokens=True)
        else:
            # Fallback: assume predictions are already text
            pred_text = str(pred)
            label_text = str(label)

        decoded_preds.append(pred_text.strip())
        decoded_labels.append(label_text.strip())

    # Calculate exact match accuracy
    correct = 0
    total = len(decoded_preds)

    for pred, label in zip(decoded_preds, decoded_labels):
        if pred == label:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"Evaluation Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "eval_accuracy": accuracy,
        "eval_correct_predictions": correct,
        "eval_total_predictions": total
    }

def compute_metrics(eval_pred: EvalPrediction, tokenizer=None) -> dict:
    """
    Combined metrics computation for hyperparameter tuning.
    Includes both accuracy and loss metrics.

    Args:
        eval_pred: Evaluation prediction from transformers trainer
        tokenizer: Tokenizer for decoding

    Returns:
        dict: Combined metrics dictionary
    """
    # Get accuracy metrics
    metrics = compute_accuracy(eval_pred, tokenizer)
    loss = np.mean(eval_pred.losses) # type: ignore
    metrics["eval_loss"] = loss

    # Return combined metrics
    return metrics
