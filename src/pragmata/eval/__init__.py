"""Public evaluation namespace."""

from pragmata.api.eval import predict_labels as predict_labels
from pragmata.api.eval import train_evaluator as train_evaluator

__all__ = ["predict_labels", "train_evaluator"]
