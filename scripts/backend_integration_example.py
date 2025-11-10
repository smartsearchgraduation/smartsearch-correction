# Backend Integration Example with Typo Correction

"""
Integration example for SmartSearch Backend.
Shows how to add TypoCorrector to the existing Flask API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import uuid
import json
import os
import random

# Add TypoCorrector import
import sys
sys.path.append('..')  # Adjust path to reach app/
from app.typo_corrector import TypoCorrector

app = Flask(__name__)
CORS(app)

# Initialize TypoCorrector once at startup
corrector = TypoCorrector()

# ... existing code ...

@app.route('/api/search', methods=['POST'])
def search():
    """
    Handle search query from frontend with typo correction.
    """
    try:
        data = request.get_json()

        if not data or 'raw_text' not in data:
            return jsonify({"error": "Missing 'raw_text' in request body"}), 400

        raw_text = data.get('raw_text', '')
        pipeline_hint = data.get('pipeline_hint', 'text')
        
        query_id = str(uuid.uuid4())
        
        # APPLY TYPO CORRECTION HERE
        correction_result = corrector.correct(raw_text)
        corrected_text = correction_result['normalized_query']
        
        # Log correction if changed
        if correction_result['changed']:
            print(f"Corrected: '{raw_text}' -> '{corrected_text}'")
        
        # TODO: Call FAISS with corrected_text instead of raw_text
        faiss_response = {
            "corrected_text": corrected_text,  # Now using our correction
            "results": []
        }
        
        # ... rest of the code remains the same ...
        corrected_text = faiss_response.get('corrected_text', corrected_text)
        faiss_results = faiss_response.get('results', [])
        
        # ... existing enrichment logic ...
        
        # Store query with both original and corrected
        queries[query_id] = {
            'query_id': query_id,
            'raw_text': raw_text,
            'corrected_text': corrected_text,
            'pipeline_hint': pipeline_hint,
            'timestamp': datetime.now().isoformat(),
            'correction_details': correction_result,  # Store full correction info
            'results': enriched_results,
            'products_sent': products_to_send
        }

        return jsonify({
            'query_id': query_id,
            'corrected_text': corrected_text,
            'products': products_to_send
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ... rest of existing code ...