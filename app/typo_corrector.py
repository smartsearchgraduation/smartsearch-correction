# TypoCorrector class

import re
from symspellpy import SymSpell, Verbosity

from .domain_vocab import load_domain_vocab

class TypoCorrector:
    """
    Typo Correction Module for SmartSearch E-commerce System.
    
    This class provides fast, dictionary-based typo correction using SymSpell.
    It loads domain-specific and general English vocabularies to suggest corrections
    for misspelled words in search queries.
    
    Usage:
        corrector = TypoCorrector()
        result = corrector.correct("iphnoe 15")
        # Returns: {"original_query": "iphnoe 15", "normalized_query": "iphone 15", ...}
    """
    
    def __init__(self, vocab_path: str = "data/domain_vocab.txt"):
        """
        Initialize the corrector with vocabularies.
        
        Args:
            vocab_path: Path to domain-specific vocabulary file
        """
        # Load domain words (e.g., product names, brands)
        self.domain_vocab = load_domain_vocab(vocab_path)
        
        # Simple generic vocab for demo (expanded with SCOWL common words)
        self.generic_vocab = self._load_generic_vocab("data/common_words.txt") | {
            "the", "and", "is", "in", "to", "of", "a", "for", "on", "with", "as", "by", "at", "from",
            "under", "between", "dollars", "gb", "tb", "hz", "iphone", "samsung", "lenovo", "nvidia",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "my", "your", "his", "its", "our", "their", "what", "where", "when", "why", "how", "who",
            "which", "there", "here", "now", "then", "so", "but", "or", "because", "although", "if",
            "then", "than", "too", "very", "also", "not", "no", "yes", "please", "thank", "hello",
            "good", "bad", "big", "small", "new", "old", "hot", "cold", "fast", "slow", "easy", "hard",
            "happy", "sad", "love", "hate", "friend", "family", "work", "school", "home", "car", "book",
            "food", "water", "time", "day", "night", "morning", "evening", "year", "month", "week",
            "today", "yesterday", "tomorrow", "life", "story", "tell", "write", "read", "speak", "listen",
            "laptop", "gaming", "mouse", "keyboard", "monitor", "headphones", "phone", "tablet", "camera", "telephone"
        }
        self.all_vocab = self.domain_vocab | self.generic_vocab
        
        # Initialize SymSpell
        self.sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        for word in self.all_vocab:
            self.sym_spell.create_dictionary_entry(word, 1)  # frequency 1

    def _load_generic_vocab(self, vocab_path: str) -> set[str]:
        vocab = set()
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        vocab.add(word)
        except FileNotFoundError:
            print(f"Warning: {vocab_path} not found. Using empty vocab.")
        return vocab

    def _is_numeric_or_unit(self, token: str) -> bool:
        # Check if token is numeric, unit-like, or operator
        token_lower = token.lower()
        if re.match(r'^\d+(\.\d+)?$', token):  # numbers
            return True
        if token_lower in {"gb", "tb", "hz", "dollars", "under", "between", "<", ">", "$", "€"}:
            return True
        return False

    def _find_best_candidate(self, token: str):
        """
        Find the best correction candidate for a token using SymSpell.
        
        Args:
            token: The word to correct
            
        Returns:
            dict: {'word': corrected_word, 'distance': edit_distance}
        """
        token_lower = token.lower()
        
        # Get suggestions from SymSpell (fuzzy matching)
        suggestions = self.sym_spell.lookup(token_lower, Verbosity.CLOSEST, max_edit_distance=3)
        
        if suggestions:
            best = suggestions[0]
            distance = best.distance
            
            # High confidence for small edit distances
            if distance <= 2:
                confidence = 1.0 - distance / len(token_lower)
                return {"word": best.term, "distance": confidence}
        
        # No good suggestion found, return original
        return {"word": token_lower, "distance": 1.0}

    def correct(self, query: str) -> dict:
        """
        Correct typos in a search query.
        
        Args:
            query: The user's search query string
            
        Returns:
            dict: Correction result with original, corrected query, change flag, and token details
        """
        if not query.strip():
            return {
                "original_query": query,
                "normalized_query": query,
                "changed": False,
                "tokens": []
            }
        
        tokens = query.split()  # Simple tokenization by spaces
        corrected_tokens = []
        token_details = []
        changed = False
        
        for token in tokens:
            # Skip numbers, units, operators
            if self._is_numeric_or_unit(token):
                corrected_tokens.append(token)
                continue
            
            # Find best correction candidate
            candidate = self._find_best_candidate(token)
            corrected_tokens.append(candidate['word'])
            
            # Track changes and confidence
            if candidate['word'] != token:
                changed = True
                token_details.append({
                    "original": token,
                    "corrected": candidate['word'],
                    "confidence": candidate['distance']
                })
        
        normalized_query = " ".join(corrected_tokens)
        
        return {
            "original_query": query,
            "normalized_query": normalized_query,
            "changed": changed,
            "tokens": token_details
        }

    # Hook for future transformer-based corrector (e.g., neuspell)
    def correct_with_transformer(self, query: str) -> dict:
        try:
            from neuspell import BertChecker
            checker = BertChecker()
            checker.from_pretrained()
            corrected = checker.correct_strings([query])[0]
            changed = corrected != query
            return {
                "original_query": query,
                "normalized_query": corrected,
                "changed": changed,
                "tokens": []  # neuspell doesn't provide token-level details easily
            }
        except ImportError:
            print("Neuspell not available, falling back to original")
            return {
                "original_query": query,
                "normalized_query": query,
                "changed": False,
                "tokens": []
            }
        except Exception as e:
            print(f"Error in transformer correction: {e}")
            return {
                "original_query": query,
                "normalized_query": query,
                "changed": False,
                "tokens": []
            }

    def correct_batch(self, queries: list[str]) -> list[dict]:
        """Correct a batch of queries. Returns list of correction results."""
        return [self.correct(query) for query in queries]