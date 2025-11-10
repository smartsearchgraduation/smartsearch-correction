# SmartSearch Typo Correction Module

This module provides typo correction for e-commerce search queries using a hybrid approach of dictionary-based correction and edit distance.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows: `.\venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Backend Integration

Your backend team can integrate this module as follows:

### Python Backend Example

```python
from app.typo_corrector import TypoCorrector

# Initialize once (e.g., at startup)
corrector = TypoCorrector()

# For each search query
query = "iphnoe 15 512 gb under 500 dolars"  # from user input
result = corrector.correct(query)

# Use the corrected query for search
normalized_query = result['normalized_query']  # "iphone 15 512 gb under 500 dollars"

# For batch processing
queries = ["iphnoe", "samsng", "lptop"]
results = corrector.correct_batch(queries)
corrected_queries = [r['normalized_query'] for r in results]
```

### Output Format

The `correct()` method returns a JSON-serializable dictionary:

```json
{
  "original_query": "iphnoe 15 512 gb under 500 dolars",
  "normalized_query": "iphone 15 512 gb under 500 dollars",
  "changed": true,
  "tokens": [
    {
      "original": "iphnoe",
      "corrected": "iphone",
      "confidence": 0.83
    },
    {
      "original": "dolars",
      "corrected": "dollars",
      "confidence": 0.83
    }
  ]
}
```

- `original_query`: User's original input
- `normalized_query`: Corrected query (ready for search)
- `changed`: Boolean indicating if any corrections were made
- `tokens`: List of individual token corrections (optional, for debugging)

### Integration with SmartSearch Backend

Based on your Flask API, integrate as follows:

1. **Add to imports** (top of your `app.py`):
   ```python
   import sys
   sys.path.append('path/to/typo_correction')  # Adjust path
   from app.typo_corrector import TypoCorrector
   ```

2. **Initialize at startup** (after `CORS(app)`):
   ```python
   corrector = TypoCorrector()
   ```

3. **Modify `/api/search` endpoint** (after getting `raw_text`):
   ```python
   # Apply typo correction
   correction_result = corrector.correct(raw_text)
   corrected_text = correction_result['normalized_query']
   
   # Log if changed
   if correction_result['changed']:
       print(f"Corrected: '{raw_text}' -> '{corrected_text}'")
   
   # Use corrected_text for FAISS call instead of raw_text
   faiss_response = call_faiss_api(corrected_text, pipeline_hint)
   
   # Store correction details in query record
   queries[query_id] = {
       # ... existing fields ...
       'correction_details': correction_result
   }
   ```

This ensures queries are corrected before reaching FAISS, improving search accuracy.

See `scripts/backend_integration_example.py` for a complete example.

### Evaluation

Run the evaluation script:
```bash
python scripts/eval_typo.py
```

### Demo CLI

Run the demo CLI:
```bash
python scripts/demo_cli.py
```

### Evaluation
Run the evaluation script:
```bash
python scripts/eval_typo.py
```

Example output:
```
Samples evaluated: 113
Sentence accuracy: 0.45
Token accuracy: 0.53
Average Levenshtein distance: 2.61
```

- **Sentence accuracy**: Percentage of fully correct sentences.
- **Token accuracy**: Percentage of correctly corrected words (token-level).
- **Average Levenshtein distance**: Average character-level edit distance.

Enter queries to test the correction interactively.

## Code Structure

### app/typo_corrector.py
- **TypoCorrector class**: Main correction logic
  - `__init__()`: Loads vocabularies and initializes SymSpell
  - `correct(query)`: Processes a single query, returns correction result
  - `correct_batch(queries)`: Processes multiple queries
  - `_find_best_candidate(token)`: Finds best correction using fuzzy matching
  - Helper methods for vocab loading and token checking

### app/domain_vocab.py
- `load_domain_vocab()`: Loads domain-specific words from file

### app/metrics.py
- Evaluation functions: sentence accuracy, token accuracy, Levenshtein distance

### scripts/
- `demo_cli.py`: Interactive demo
- `eval_typo.py`: Batch evaluation on dataset
- `backend_integration_example.py`: Flask integration example

### data/
- `typo_dataset.csv`: Test examples (noisy → clean pairs)
- `domain_vocab.txt`: E-commerce domain words
- `common_words.txt`: General English vocabulary