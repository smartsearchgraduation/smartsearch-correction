# Metrics functions

import textdistance

def sentence_accuracy(preds: list[str], targets: list[str]) -> float:
    """
    Percentage of samples where predicted == target.
    """
    if len(preds) != len(targets):
        raise ValueError("Preds and targets must have the same length")
    correct = sum(1 for p, t in zip(preds, targets) if p == t)
    return correct / len(preds) if preds else 0.0

def avg_levenshtein(preds: list[str], targets: list[str]) -> float:
    """
    Average Levenshtein distance between predicted and target.
    """
    if len(preds) != len(targets):
        raise ValueError("Preds and targets must have the same length")
    distances = [textdistance.levenshtein(p, t) for p, t in zip(preds, targets)]
    return sum(distances) / len(distances) if distances else 0.0