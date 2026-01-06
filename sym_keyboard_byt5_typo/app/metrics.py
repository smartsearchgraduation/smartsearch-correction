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

def token_level_accuracy(preds: list[str], targets: list[str]) -> float:
    """
    Token-level accuracy: fraction of tokens that match exactly.
    """
    if len(preds) != len(targets):
        raise ValueError("Preds and targets must have the same length")
    
    total_tokens = 0
    correct_tokens = 0
    
    for p, t in zip(preds, targets):
        p_tokens = p.split()
        t_tokens = t.split()
        min_len = min(len(p_tokens), len(t_tokens))
        
        for i in range(min_len):
            if p_tokens[i] == t_tokens[i]:
                correct_tokens += 1
        total_tokens += max(len(p_tokens), len(t_tokens))
    
    return correct_tokens / total_tokens if total_tokens > 0 else 0.0

def avg_jaccard_similarity(preds: list[str], targets: list[str]) -> float:
    """
    Average Jaccard similarity between predicted and target token sets.
    """
    if len(preds) != len(targets):
        raise ValueError("Preds and targets must have the same length")
    
    similarities = []
    for p, t in zip(preds, targets):
        p_tokens = set(p.split())
        t_tokens = set(t.split())
        if not p_tokens and not t_tokens:
            similarities.append(1.0)
        else:
            intersection = len(p_tokens & t_tokens)
            union = len(p_tokens | t_tokens)
            similarities.append(intersection / union if union > 0 else 0.0)
    
    return sum(similarities) / len(similarities) if similarities else 0.0