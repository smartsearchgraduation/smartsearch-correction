import importlib.resources
from symspellpy import SymSpell
from .domain_vocab import load_domain_vocab


class TypoCorrector:
    def __init__(self, vocab_path: str = "data/domain_vocab.txt"):
        # 1. Initialize SymSpell
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

        # 2. Find the dictionary paths
        dictionary_path = str(importlib.resources.files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt"))
        bigram_path = str(importlib.resources.files("symspellpy").joinpath("frequency_bigramdictionary_en_243_342.txt"))

        # 3. Load the dictionaries
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

        # 4. Load your custom domain vocab *on top* of the main dictionary
        print(f"Loading domain vocabulary from {vocab_path}...")
        try:
            domain_vocab_set = load_domain_vocab(vocab_path)
            # Give domain words a very high count to ensure they are prioritized
            domain_word_count = 100000000
            for word in domain_vocab_set:
                self.sym_spell.create_dictionary_entry(word, domain_word_count)
            print(f"Loaded {len(domain_vocab_set)} domain words.")
        except Exception as e:
            print(f"Warning: Could not load domain vocab from {vocab_path}. Error: {e}")

    def correct(self, query: str) -> dict:
        if not query.strip():
            return {"original_query": query, "normalized_query": query, "changed": False, "tokens": []}

        query_lower = query.lower()

        # Use lookup_compound for whole-query correction.
        # This is fast, context-aware, and handles word splitting/merging.
        suggestions = self.sym_spell.lookup_compound(
            query_lower, max_edit_distance=2, ignore_non_words=True  # This automatically handles numbers, units, etc.
        )

        # Get the top suggestion
        if suggestions:
            normalized_query = suggestions[0].term
        else:
            normalized_query = query_lower

        # This check is now against the lowercased original
        # This fixes the "iPhone" -> "iphone" false positive
        changed = normalized_query != query_lower

        return {
            "original_query": query,
            "normalized_query": normalized_query,
            "changed": changed,
            # Note: lookup_compound does not provide per-token details
            # This is the trade-off for speed and contextual correction.
            "tokens": [],
        }

    def correct_batch(self, queries: list[str]) -> list[dict]:
        """Correct a batch of queries. Returns list of correction results."""
        return [self.correct(query) for query in queries]
