# Evaluation script

import pandas as pd
import sys
import os

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.typo_corrector import TypoCorrector
from app.metrics import sentence_accuracy, avg_levenshtein

def main():
    # Load dataset
    dataset_path = "data/typo_dataset.csv"
    df = pd.read_csv(dataset_path)
    
    # Initialize corrector
    corrector = TypoCorrector()
    
    preds = []
    targets = []
    total_tokens = 0
    correct_tokens = 0
    
    for _, row in df.iterrows():
        noisy = row['noisy']
        clean = row['clean']
        result = corrector.correct(noisy)
        pred = result['normalized_query']
        preds.append(pred)
        targets.append(clean)
        
        # Token-level accuracy
        pred_tokens = pred.split()
        clean_tokens = clean.split()
        min_len = min(len(pred_tokens), len(clean_tokens))
        for i in range(min_len):
            if pred_tokens[i] == clean_tokens[i]:
                correct_tokens += 1
        total_tokens += max(len(pred_tokens), len(clean_tokens))
    
    acc = sentence_accuracy(preds, targets)
    avg_lev = avg_levenshtein(preds, targets)
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    print(f"Samples evaluated: {len(preds)}")
    print(f"Sentence accuracy: {acc:.2f}")
    print(f"Token accuracy: {token_acc:.2f}")
    print(f"Average Levenshtein distance: {avg_lev:.2f}")

if __name__ == "__main__":
    main()