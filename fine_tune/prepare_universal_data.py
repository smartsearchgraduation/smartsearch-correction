"""
Universal E-Commerce Training Data Preparation for T5 Typo Correction
=======================================================================

Generates ~100K training examples across ALL e-commerce categories:
    - electronics, fashion, beauty, home, sports, toys, automotive, grocery, books, office

Data Composition Target (by percentage):
    - 25% identity (correct→correct) with brands, abbreviations, units, currencies
    - 20% synthetic single-word typos (keyboard, deletion, swap, insertion, double, compound)
    - 20% multi-word keyword queries across ALL categories
    - 15% sentence-like natural language queries
    - 5% explicit typo mappings
    - 5% abbreviation/unit/currency examples (protected tokens)
    - 5% spacing/symbol variants for model numbers
    - 3% cross-category queries

Output Format: T5-style JSONL
    {"input_text": "correct: <typo_query>", "target_text": "<correct_query>"}

Usage:
    python prepare_universal_data.py --total 100000 --eval-ratio 0.1 --output-dir ./data
    python prepare_universal_data.py --total 50000 --output-dir /path/to/output
"""

import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import defaultdict

# Import Universal Vocabulary
from universal_ecommerce_vocab import (
    UniversalVocab,
    GLOBAL_UNITS,
    GLOBAL_CURRENCIES,
    GLOBAL_QUANTITY_ABBR,
    GLOBAL_COMMON_SHORTHAND,
    GLOBAL_SIZE_TERMS,
    PROTECTED_COMMON_WORDS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ======================================================================
# QWERTY Keyboard Layout (for typo generation)
# ======================================================================

QWERTY_NEIGHBORS = {
    'q': ['w', 'a'], 'w': ['q', 'e', 'a', 's'], 'e': ['w', 'r', 's', 'd'],
    'r': ['e', 't', 'd', 'f'], 't': ['r', 'y', 'f', 'g'], 'y': ['t', 'u', 'g', 'h'],
    'u': ['y', 'i', 'h', 'j'], 'i': ['u', 'o', 'j', 'k'], 'o': ['i', 'p', 'k', 'l'],
    'p': ['o', 'l'], 'a': ['q', 'w', 's', 'z'], 's': ['a', 'w', 'e', 'd', 'z', 'x'],
    'd': ['s', 'e', 'r', 'f', 'x', 'c'], 'f': ['d', 'r', 't', 'g', 'c', 'v'],
    'g': ['f', 't', 'y', 'h', 'v', 'b'], 'h': ['g', 'y', 'u', 'j', 'b', 'n'],
    'j': ['h', 'u', 'i', 'k', 'n', 'm'], 'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'], 'z': ['a', 's', 'x'], 'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'], 'v': ['c', 'f', 'g', 'b'], 'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'], 'm': ['n', 'j', 'k'],
}

# Multi-word query templates (at least 30 diverse templates)
MULTI_WORD_TEMPLATES = [
    # Electronics
    "{brand} {product} {model}",
    "{brand} {product} with {feature}",
    "best {product} from {brand}",
    "{product} {model} price",
    "{brand} {product} review",
    "{brand} gaming {product}",
    "{product} {color} {storage}",
    "gaming {product} under {price}",
    "{brand} {product} {specs}",
    "lightweight {product} {specs}",

    # Fashion
    "{brand} {product} {size}",
    "{color} {product} for {occasion}",
    "{product} {material} {size}",
    "comfortable {product} {occasion}",
    "{brand} {product} {pattern}",
    "sale {product} {size}",
    "{brand} {product} online",
    "organic {product} {type}",
    "{product} collection {season}",
    "stylish {product} {occasion}",

    # Beauty & Home
    "{brand} {product} {type}",
    "{product} for {skintype} skin",
    "natural {product} {benefit}",
    "{product} {scent} {size}",
    "{brand} {product} bundle",
    "organic {product} {type}",
    "{product} cream with {ingredient}",
    "best {product} {benefit}",
    "{product} set {count}",
    "{brand} {product} refill",

    # Sports & Toys
    "{product} for {sport}",
    "{brand} {product} {skill}",
    "{product} kids {age}",
    "outdoor {product} {purpose}",
    "professional {product} {use}",
    "{brand} {product} durable",
    "educational {product} {age}",
    "waterproof {product} {type}",
    "{product} beginner {purpose}",
    "portable {product} {use}",
]

# Sentence query templates (at least 20 diverse templates)
SENTENCE_TEMPLATES = [
    "i need a {product} for {purpose}",
    "looking for {article} {quality} {product}",
    "can you show me {article} {brand} {product}",
    "what is the best {product} for {purpose}",
    "i want to buy {article} {product} {feature}",
    "find me {article} {product} under {price}",
    "do you have {article} {brand} {product}",
    "show me {product} that are {quality}",
    "i am looking for {product} with {feature}",
    "which {product} should i get for {purpose}",
    "can i get {article} {brand} {product}",
    "where can i find {article} {quality} {product}",
    "i need {article} {product} for {sport}",
    "show me {article} {product} in {color}",
    "what are some good {product} brands",
    "do you sell {article} {brand} {product}",
    "i want {article} {product} that is {feature}",
    "find {article} {product} for kids",
    "show me the latest {product} {feature}",
    "can i find {article} {product} in {size}",
]


# ======================================================================
# Typo Generation Functions
# ======================================================================

def generate_keyboard_typo(word: str) -> str:
    """Generate a typo by replacing a character with a keyboard neighbor."""
    if len(word) < 2:
        return word
    pos = random.randint(0, len(word) - 1)
    char = word[pos].lower()
    if char in QWERTY_NEIGHBORS:
        replacement = random.choice(QWERTY_NEIGHBORS[char])
        return word[:pos] + replacement + word[pos + 1:]
    return word


def generate_deletion_typo(word: str) -> str:
    """Generate a typo by deleting a random character."""
    if len(word) < 3:
        return word
    pos = random.randint(1, len(word) - 2)
    return word[:pos] + word[pos + 1:]


def generate_swap_typo(word: str) -> str:
    """Generate a typo by swapping adjacent characters."""
    if len(word) < 3:
        return word
    pos = random.randint(0, len(word) - 2)
    return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]


def generate_insertion_typo(word: str) -> str:
    """Generate a typo by inserting a random character."""
    pos = random.randint(0, len(word))
    char = random.choice('abcdefghijklmnopqrstuvwxyz')
    return word[:pos] + char + word[pos:]


def generate_double_typo(word: str) -> str:
    """Generate a typo by doubling a character."""
    if len(word) < 2:
        return word
    pos = random.randint(0, len(word) - 1)
    return word[:pos] + word[pos] + word[pos:]


def generate_compound_typo(word: str) -> str:
    """Generate a typo with 2 errors applied (more realistic)."""
    if len(word) < 4:
        return word
    single_generators = [
        generate_keyboard_typo,
        generate_deletion_typo,
        generate_swap_typo,
        generate_insertion_typo,
    ]
    g1, g2 = random.sample(single_generators, 2)
    result = g1(word)
    if result != word:
        result2 = g2(result)
        if result2 != result:
            return result2
    return result


def generate_case_variant(word: str) -> str:
    """Generate case variation."""
    choice = random.randint(0, 3)
    if choice == 0:
        return word.upper()
    elif choice == 1:
        return word.capitalize()
    elif choice == 2:
        return "".join(c.upper() if random.random() < 0.3 else c.lower() for c in word)
    return word


def generate_truncation_typo(word: str) -> str:
    """Generate a truncation (user stops typing early)."""
    if len(word) < 5:
        return word
    cut = random.randint(3, len(word) - 2)
    return word[:cut]


# ======================================================================
# Spacing & Symbol Variants
# ======================================================================

def generate_spacing_variants(phrase: str) -> List[str]:
    """Generate spacing variants for model numbers/specs.

    Examples:
        "16 gb ram" -> ["16gbram", "16 gbram", "16gb ram"]
        "rtx 4070 ti" -> ["rtx4070ti", "rtx4070 ti", "rtx 4070ti"]
    """
    parts = phrase.split()
    if len(parts) < 2 or len(phrase) > 40:
        return []

    variants: Set[str] = set()

    # Remove single spaces (merge neighboring tokens)
    for i in range(len(parts) - 1):
        merged = parts[:]
        merged[i] = merged[i] + merged[i + 1]
        del merged[i + 1]
        variants.add(" ".join(merged))

    # Remove all spaces
    variants.add("".join(parts))

    # Remove last space only (common for "16 gb" -> "16gb")
    if len(parts) >= 2:
        first = " ".join(parts[:-1])
        last = parts[-1]
        variants.add(first + last)

    return list(variants)


def generate_symbol_variants(token: str) -> List[str]:
    """Generate symbol variants (hyphen, underscore) for model numbers.

    Examples:
        "rtx 4070 ti" -> ["rtx-4070-ti", "rtx_4070_ti"]
        "i9-14900k" -> ["i9 14900k", "i9_14900k"]
    """
    variants: Set[str] = set()
    base = token.strip()

    if not base:
        return []

    # Space-based -> hyphen/underscore
    if " " in base:
        parts = base.split()
        variants.add("-".join(parts))
        variants.add("_".join(parts))

        # Merge last two parts: "4070 ti" -> "4070ti"
        if len(parts) >= 2:
            merged_last = parts[:-2] + [parts[-2] + parts[-1]]
            variants.add(" ".join(merged_last))
            variants.add("".join(merged_last))

    # Hyphen-based -> space/underscore
    if "-" in base:
        parts = base.split("-")
        variants.add(" ".join(parts))
        variants.add("_".join(parts))

    return list(variants)


# ======================================================================
# Training Data Generators
# ======================================================================

class UniversalDataGenerator:
    """Generates training data across all e-commerce categories."""

    def __init__(self, total_examples: int = 100000, eval_ratio: float = 0.1):
        self.total_examples = total_examples
        self.eval_ratio = eval_ratio
        self.train_examples = int(total_examples * (1 - eval_ratio))
        self.eval_examples = total_examples - self.train_examples

        # Load vocabulary
        self.vocab = UniversalVocab()
        self.vocab.load_all()

        self.protected = list(self.vocab.get_protected_tokens())
        self.all_brands = list(self.vocab.get_all_brand_names_flat())
        self.all_products = list(self.vocab.get_all_products())
        self.all_typos = self.vocab.get_all_typo_mappings()
        self.all_vocab = list(self.vocab.get_all_vocabulary())

        logger.info(f"Loaded {len(self.all_vocab)} vocabulary items")
        logger.info(f"Loaded {len(self.protected)} protected tokens")
        logger.info(f"Loaded {len(self.all_typos)} explicit typo mappings")

        self.examples = []
        self.stats = defaultdict(int)

    def generate_identity_examples(self, count: int) -> List[Dict]:
        """25% - Generate identity examples (correct→correct).

        Includes ALL brands, units, currencies, abbreviations to prevent
        over-correction. Generates multi-word combos to reach target count.
        """
        examples = []
        seen = set()

        # 1) All protected tokens (brands, units, currencies, etc.)
        for token in self.protected:
            if token not in seen:
                seen.add(token)
                examples.append({
                    "input_text": f"correct: {token}",
                    "target_text": token,
                    "category": "identity_protected"
                })

        # 2) All vocabulary items
        for word in self.all_vocab:
            if word not in seen:
                seen.add(word)
                examples.append({
                    "input_text": f"correct: {word}",
                    "target_text": word,
                    "category": "identity_vocab"
                })

        # 3) Case variants
        for word in random.sample(self.all_vocab, min(len(self.all_vocab), count // 5)):
            if len(word) > 2:
                case_var = generate_case_variant(word)
                if case_var not in seen:
                    seen.add(case_var)
                    examples.append({
                        "input_text": f"correct: {case_var}",
                        "target_text": case_var.lower(),
                        "category": "identity_case"
                    })

        # 4) Multi-word identity combos (brand + product, product + color, etc.)
        combos = [
            (self.all_brands, self.all_products),
            (self.all_products, ["black", "white", "silver", "gold", "blue", "red", "green", "pink"]),
            (self.all_brands, ["pro", "max", "ultra", "lite", "plus", "mini", "air"]),
            (["best", "cheap", "top", "new", "sale"], self.all_products),
        ]
        while len(examples) < count:
            combo_pair = random.choice(combos)
            w1 = random.choice(combo_pair[0]) if combo_pair[0] else "product"
            w2 = random.choice(combo_pair[1]) if combo_pair[1] else "item"
            phrase = f"{w1} {w2}"
            if phrase not in seen:
                seen.add(phrase)
                examples.append({
                    "input_text": f"correct: {phrase}",
                    "target_text": phrase,
                    "category": "identity_multi"
                })

        self.stats["identity"] += len(examples)
        return examples[:count]

    def generate_single_word_typos(self, count: int) -> List[Dict]:
        """20% - Generate synthetic single-word typos.

        Multiple typo variants per word are allowed to reach target count.
        """
        examples = []
        seen_typos = set()
        typo_types = ['keyboard', 'deletion', 'swap', 'insertion', 'double', 'compound', 'truncation']
        typo_weights = [2, 2, 2, 1, 1, 1, 1]
        generators = {
            'keyboard': generate_keyboard_typo,
            'deletion': generate_deletion_typo,
            'swap': generate_swap_typo,
            'insertion': generate_insertion_typo,
            'double': generate_double_typo,
            'compound': generate_compound_typo,
            'truncation': generate_truncation_typo,
        }

        # Filter vocab to words with length >= 3
        usable_vocab = [w for w in self.all_vocab if len(w) >= 3]
        attempts = 0
        max_attempts = count * 5

        while len(examples) < count and attempts < max_attempts:
            word = random.choice(usable_vocab)
            typo_type = random.choices(typo_types, weights=typo_weights, k=1)[0]
            typo = generators[typo_type](word)

            if typo != word and typo not in seen_typos:
                seen_typos.add(typo)
                examples.append({
                    "input_text": f"correct: {typo}",
                    "target_text": word,
                    "category": f"typo_{typo_type}"
                })

            attempts += 1

        self.stats["single_word_typos"] += len(examples)
        return examples[:count]

    def generate_multi_word_queries(self, count: int) -> List[Dict]:
        """20% - Generate multi-word keyword queries across ALL categories."""
        examples = []

        for _ in range(count):
            template = random.choice(MULTI_WORD_TEMPLATES)

            # Generate tokens for template
            tokens = {
                "brand": random.choice(self.all_brands) if self.all_brands else "samsung",
                "product": random.choice(self.all_products) if self.all_products else "laptop",
                "model": f"{random.choice(['pro', 'max', 'ultra', 'lite', 'plus'])} {random.randint(2020, 2025)}",
                "feature": random.choice(["fast", "lightweight", "durable", "wireless", "waterproof"]),
                "color": random.choice(["black", "white", "silver", "gold", "blue", "red"]),
                "storage": random.choice(["128gb", "256gb", "512gb", "1tb"]),
                "specs": random.choice(["8gb ram", "16gb ram", "32gb ram", "high performance"]),
                "size": random.choice(["xs", "s", "m", "l", "xl", "xxl"]),
                "material": random.choice(["cotton", "wool", "silk", "synthetic", "leather"]),
                "pattern": random.choice(["solid", "striped", "floral", "geometric"]),
                "occasion": random.choice(["casual", "formal", "party", "office"]),
                "type": random.choice(["basic", "premium", "deluxe", "professional"]),
                "skintype": random.choice(["oily", "dry", "sensitive", "normal"]),
                "benefit": random.choice(["moisturizing", "whitening", "anti-aging"]),
                "ingredient": random.choice(["aloe", "vitamin c", "retinol", "hyaluronic acid"]),
                "scent": random.choice(["lavender", "rose", "citrus", "vanilla"]),
                "sport": random.choice(["running", "yoga", "gym", "cycling", "swimming"]),
                "skill": random.choice(["beginner", "intermediate", "advanced", "professional"]),
                "age": random.choice(["3-5", "5-8", "8-12", "teens", "adults"]),
                "purpose": random.choice(["fitness", "gaming", "studying", "work"]),
                "quality": random.choice(["best", "premium", "affordable", "budget"]),
                "article": random.choice(["a", "an", "the"]),
                "price": random.choice(["500", "1000", "1500", "2000", "5000"]),
                "count": random.choice(["2", "3", "5", "10", "pack"]),
                "season": random.choice(["spring", "summer", "fall", "winter"]),
                "use": random.choice(["home", "office", "outdoor", "travel"]),
            }

            try:
                query = template.format(**tokens)

                # Introduce typo in ~30% of multi-word queries
                if random.random() < 0.3:
                    words = query.split()
                    if len(words) > 1:
                        idx = random.randint(0, len(words) - 1)
                        word = words[idx]
                        if len(word) > 3:
                            typo_word = random.choice([
                                generate_keyboard_typo(word),
                                generate_deletion_typo(word),
                                generate_insertion_typo(word),
                            ])
                            if typo_word != word:
                                words[idx] = typo_word
                    typo_query = " ".join(words)
                else:
                    typo_query = query

                examples.append({
                    "input_text": f"correct: {typo_query}",
                    "target_text": query,
                    "category": "multi_word"
                })
            except KeyError:
                # Skip if template variables not available
                pass

        self.stats["multi_word"] += len(examples)
        return examples[:count]

    def generate_sentence_queries(self, count: int) -> List[Dict]:
        """15% - Generate sentence-like natural language queries."""
        examples = []

        for _ in range(count):
            template = random.choice(SENTENCE_TEMPLATES)

            tokens = {
                "product": random.choice(self.all_products) if self.all_products else "laptop",
                "purpose": random.choice(["fitness", "gaming", "studying", "work", "casual wear"]),
                "article": random.choice(["a", "an", "the"]),
                "quality": random.choice(["best", "premium", "affordable", "good"]),
                "brand": random.choice(list(self.all_brands)) if self.all_brands else "samsung",
                "feature": random.choice(["waterproof", "lightweight", "durable", "wireless"]),
                "price": random.choice(["500", "1000", "2000", "5000"]),
                "sport": random.choice(["running", "yoga", "cycling", "swimming"]),
                "color": random.choice(["black", "white", "blue", "red"]),
                "size": random.choice(["small", "medium", "large", "extra large"]),
            }

            try:
                sentence = template.format(**tokens)

                # Introduce typo in ~40% of sentences
                if random.random() < 0.4:
                    words = sentence.split()
                    if len(words) > 2:
                        idx = random.randint(1, len(words) - 2)  # Don't change first word usually
                        word = words[idx]
                        if len(word) > 3 and word not in PROTECTED_COMMON_WORDS:
                            typo_word = random.choice([
                                generate_keyboard_typo(word),
                                generate_deletion_typo(word),
                                generate_swap_typo(word),
                            ])
                            if typo_word != word:
                                words[idx] = typo_word
                    typo_sentence = " ".join(words)
                else:
                    typo_sentence = sentence

                examples.append({
                    "input_text": f"correct: {typo_sentence}",
                    "target_text": sentence,
                    "category": "sentence"
                })
            except KeyError:
                pass

        self.stats["sentence"] += len(examples)
        return examples[:count]

    def generate_explicit_typos(self, count: int) -> List[Dict]:
        """5% - Explicit typo mappings from vocabulary.

        Augments with multi-word contexts to exceed raw mapping count.
        """
        examples = []
        mappings_list = list(self.all_typos.items())

        # 1) All raw mappings first
        for typo, correct in mappings_list:
            examples.append({
                "input_text": f"correct: {typo}",
                "target_text": correct,
                "category": "explicit_mapping"
            })

        # 2) Augment: embed known typos in multi-word contexts
        context_words = ["best", "buy", "cheap", "new", "sale", "review",
                         "for", "top", "rated", "deal", "shop", "price"]
        while len(examples) < count:
            typo, correct = random.choice(mappings_list)
            ctx = random.choice(context_words)
            if random.random() < 0.5:
                examples.append({
                    "input_text": f"correct: {ctx} {typo}",
                    "target_text": f"{ctx} {correct}",
                    "category": "explicit_mapping_ctx"
                })
            else:
                examples.append({
                    "input_text": f"correct: {typo} {ctx}",
                    "target_text": f"{correct} {ctx}",
                    "category": "explicit_mapping_ctx"
                })

        self.stats["explicit_typos"] += len(examples)
        return examples[:count]

    def generate_protected_tokens(self, count: int) -> List[Dict]:
        """5% - Abbreviations, units, currencies (must NOT be corrected).

        Augments with multi-word contexts to reach target count.
        """
        examples = []
        seen = set()

        # 1) All raw protected tokens
        for token in self.protected:
            if token not in seen:
                seen.add(token)
                examples.append({
                    "input_text": f"correct: {token}",
                    "target_text": token,
                    "category": "protected_token"
                })

        # 2) Augment: protected tokens in context
        products = self.all_products if self.all_products else ["item"]
        while len(examples) < count:
            token = random.choice(self.protected)
            product = random.choice(products)
            phrase = f"{product} {token}" if random.random() < 0.5 else f"{token} {product}"
            if phrase not in seen:
                seen.add(phrase)
                examples.append({
                    "input_text": f"correct: {phrase}",
                    "target_text": phrase,
                    "category": "protected_in_context"
                })

        self.stats["protected_tokens"] += len(examples)
        return examples[:count]

    def generate_spacing_symbol_variants(self, count: int) -> List[Dict]:
        """5% - Spacing and symbol variants for model numbers."""
        examples = []

        # Model patterns from vocab
        model_patterns = self.vocab.get_all_model_patterns()

        for _ in range(count):
            if model_patterns:
                base, variants = random.choice(list(model_patterns.items()))
                variant = random.choice(variants)
                model = f"{base} {variant}"
            else:
                # Fallback: generate something like "rtx 4070 ti"
                brand = random.choice(list(self.all_brands)) if self.all_brands else "rtx"
                model = f"{brand} {random.randint(1000, 5000)} {random.choice(['pro', 'ti', 'ultra'])}"

            # Generate spacing/symbol variants (typos)
            variants = generate_spacing_variants(model) + generate_symbol_variants(model)

            if variants:
                typo_variant = random.choice(variants)
                examples.append({
                    "input_text": f"correct: {typo_variant}",
                    "target_text": model,
                    "category": "spacing_symbol"
                })

        self.stats["spacing_symbol"] += len(examples)
        return examples[:count]

    def generate_cross_category(self, count: int) -> List[Dict]:
        """3% - Cross-category queries mixing products from different categories."""
        examples = []
        categories = list(self.vocab.categories.values())

        for _ in range(count):
            if not categories:
                break
            # Pick 2-3 random categories
            sample_cats = random.sample(categories, min(3, len(categories)))

            query_parts = []
            for cat in sample_cats:
                if cat.products:
                    query_parts.append(random.choice(cat.products))

            if query_parts:
                query = " ".join(query_parts)

                # Add typo ~30% of the time
                if random.random() < 0.3 and len(query_parts) > 1:
                    words = query.split()
                    idx = random.randint(0, len(words) - 1)
                    word = words[idx]
                    if len(word) > 3:
                        typo_word = generate_keyboard_typo(word)
                        if typo_word != word:
                            words[idx] = typo_word
                    query_with_typo = " ".join(words)
                else:
                    query_with_typo = query

                examples.append({
                    "input_text": f"correct: {query_with_typo}",
                    "target_text": query,
                    "category": "cross_category"
                })

        self.stats["cross_category"] += len(examples)
        return examples[:count]

    def generate_all(self) -> List[Dict]:
        """Generate all training examples according to composition target."""
        logger.info(f"Generating {self.total_examples} training examples...")

        # English-only composition (no Turkish normalization)
        counts = {
            "identity": int(self.total_examples * 0.25),
            "single_word_typos": int(self.total_examples * 0.22),
            "multi_word": int(self.total_examples * 0.20),
            "sentence": int(self.total_examples * 0.15),
            "explicit_typos": int(self.total_examples * 0.05),
            "protected_tokens": int(self.total_examples * 0.05),
            "spacing_symbol": int(self.total_examples * 0.05),
            "cross_category": int(self.total_examples * 0.03),
        }

        logger.info("Generating identity examples (25%)...")
        self.examples.extend(self.generate_identity_examples(counts["identity"]))

        logger.info("Generating single-word typos (22%)...")
        self.examples.extend(self.generate_single_word_typos(counts["single_word_typos"]))

        logger.info("Generating multi-word queries (20%)...")
        self.examples.extend(self.generate_multi_word_queries(counts["multi_word"]))

        logger.info("Generating sentence queries (15%)...")
        self.examples.extend(self.generate_sentence_queries(counts["sentence"]))

        logger.info("Generating explicit typo mappings (5%)...")
        self.examples.extend(self.generate_explicit_typos(counts["explicit_typos"]))

        logger.info("Generating protected tokens (5%)...")
        self.examples.extend(self.generate_protected_tokens(counts["protected_tokens"]))

        logger.info("Generating spacing/symbol variants (5%)...")
        self.examples.extend(self.generate_spacing_symbol_variants(counts["spacing_symbol"]))

        logger.info("Generating cross-category queries (3%)...")
        self.examples.extend(self.generate_cross_category(counts["cross_category"]))

        # Trim to exact count and shuffle
        self.examples = self.examples[:self.total_examples]
        random.shuffle(self.examples)

        logger.info(f"Generated {len(self.examples)} total examples")
        return self.examples

    def split_train_eval(self) -> Tuple[List[Dict], List[Dict]]:
        """Split examples into train and eval sets based on actual count."""
        actual_total = len(self.examples)
        eval_count = max(1, int(actual_total * self.eval_ratio))
        train_count = actual_total - eval_count
        train = self.examples[:train_count]
        eval_set = self.examples[train_count:]
        return train, eval_set

    def get_stats(self) -> Dict:
        """Return generation statistics."""
        return dict(self.stats)


# ======================================================================
# Output Writers
# ======================================================================

def write_jsonl(filepath: Path, examples: List[Dict]):
    """Write examples to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    logger.info(f"Wrote {len(examples)} examples to {filepath}")


def write_stats(filepath: Path, stats: Dict, total: int, eval_ratio: float):
    """Write statistics to JSON file."""
    output_stats = {
        "total_examples": total,
        "train_examples": int(total * (1 - eval_ratio)),
        "eval_examples": int(total * eval_ratio),
        "eval_ratio": eval_ratio,
        "composition": {
            "identity": 0.25,
            "single_word_typos": 0.20,
            "multi_word": 0.20,
            "sentence": 0.15,
            "explicit_typos": 0.05,
            "protected_tokens": 0.05,
            "spacing_symbol": 0.05,
            "cross_category": 0.03,
        },
        "generation_stats": stats,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote stats to {filepath}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate universal e-commerce training data for T5 typo correction"
    )
    parser.add_argument(
        "--total",
        type=int,
        default=100000,
        help="Total number of training examples (default 100000)"
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Eval set ratio (default 0.1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for data files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: total={args.total}, eval_ratio={args.eval_ratio}")

    # Generate data
    generator = UniversalDataGenerator(
        total_examples=args.total,
        eval_ratio=args.eval_ratio
    )

    generator.generate_all()
    train, eval_set = generator.split_train_eval()

    # Write outputs
    train_path = output_dir / "train_t5.jsonl"
    eval_path = output_dir / "eval_t5.jsonl"
    stats_path = output_dir / "data_stats.json"

    write_jsonl(train_path, train)
    write_jsonl(eval_path, eval_set)
    write_stats(stats_path, generator.get_stats(), args.total, args.eval_ratio)

    logger.info("=" * 60)
    logger.info("Data generation complete!")
    logger.info(f"Train set: {len(train)} examples")
    logger.info(f"Eval set: {len(eval_set)} examples")
    logger.info(f"Total: {len(train) + len(eval_set)} examples")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
