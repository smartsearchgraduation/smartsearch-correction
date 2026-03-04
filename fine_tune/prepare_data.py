"""Data Preparation for Fine-tuning Typo + Sentence Correction Models.

Amaç:
    Hem **tek kelime** hem de **tam cümle** query'lerini en az hata ile düzeltmek.

Örnekler:
    "16 gb ram laptop"                      -> "16 gb ram laptop"
    "i need a 16 gm gamibg laptop"        -> "i need a 16 gb gaming laptop"
    "samsng galxy s24 kılıfı"             -> "samsung galaxy s24 kılıfı"

Özellikler:
    - E-commerce vocabulary (brands, products, models, abbreviations)
    - Mevcut ``typo_mappings.txt`` ile birleştirme
    - Akıllı sentetik typo üretimi (keyboard proximity, common patterns)
    - Multi-word + tam cümle query desteği
    - Doğru yazılan cümle/kelimeleri **bozmamayı** öğreten identity örnekleri
    - Train/eval split
    - Birden fazla model formatı (T5, LLM, simple)

Kategoriler:
    - electronics: Elektronik cihazlar (şu an aktif)
    - fashion: Giyim (TODO)
    - home: Ev & Yaşam (TODO)

Kullanım (örnekler):
    python prepare_data.py
    python prepare_data.py --augment --samples 10000
    python prepare_data.py --multi-word 4000 --sentence-ratio 0.4
    python prepare_data.py --max-noise-per-query 3 --min-keep-words 1
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import defaultdict

# Import e-commerce vocabulary
from ecommerce_vocab import ECommerceVocab, ELECTRONICS_MODEL_PATTERNS

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path(__file__).parent / "data"

# Turkish character normalization map (Turkish → ASCII equivalents)
TURKISH_NORMALIZE = str.maketrans({
    '\u0131': 'i',   # ı → i (dotless i)
    '\u015f': 's',   # ş → s
    '\u00f6': 'o',   # ö → o
    '\u00fc': 'u',   # ü → u
    '\u00e7': 'c',   # ç → c
    '\u011f': 'g',   # ğ → g
    '\u0130': 'I',   # İ → I
    '\u015e': 'S',   # Ş → S
    '\u00d6': 'O',   # Ö → O
    '\u00dc': 'U',   # Ü → U
    '\u00c7': 'C',   # Ç → C
    '\u011e': 'G',   # Ğ → G
})


def normalize_turkish(text: str) -> str:
    """Normalize Turkish-specific characters to ASCII equivalents."""
    return text.translate(TURKISH_NORMALIZE)


def generate_turkish_variants(word: str) -> List[Tuple[str, str]]:
    """Generate Turkish char → ASCII typo pairs.

    E.g., "kılıf" (with Turkish chars) → "kilif" (ASCII).
    Users often type without Turkish characters, so the model must
    learn both directions.
    """
    pairs = []
    ascii_word = normalize_turkish(word)
    if ascii_word != word:
        # Turkish → ASCII (most common user pattern)
        pairs.append((word, ascii_word))
        # ASCII → Turkish (reverse normalization, less common but useful)
        pairs.append((ascii_word, ascii_word))
    return pairs


# QWERTY keyboard layout for synthetic typos
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


def load_typo_mappings(filepath: Path) -> List[Tuple[str, str]]:
    """Load typo->correct mappings from file."""
    mappings = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ',' in line:
                parts = line.split(',', 1)
                if len(parts) == 2:
                    typo, correct = parts[0].strip(), parts[1].strip()
                    if typo and correct:
                        mappings.append((typo, correct))
    return mappings


def load_vocabulary(filepath: Path) -> List[str]:
    """Load vocabulary words from file."""
    words = []
    if not filepath.exists():
        return words
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith('#'):
                words.append(word)
    return words


def generate_keyboard_typo(word: str) -> str:
    """Generate a typo by replacing a character with a keyboard neighbor."""
    if len(word) < 2:
        return word
    
    # Pick a random position
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
    pos = random.randint(1, len(word) - 2)  # Don't delete first/last
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
    """Generate a typo with 2 errors applied (more realistic for fast typing)."""
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
    """Generate case variation (common in search queries)."""
    choice = random.randint(0, 3)
    if choice == 0:
        return word.upper()
    elif choice == 1:
        return word.capitalize()
    elif choice == 2:
        # Random mixed case: "iPHone", "sAmsung"
        return "".join(
            c.upper() if random.random() < 0.3 else c.lower()
            for c in word
        )
    return word


def generate_truncation_typo(word: str) -> str:
    """Generate a truncation (user stops typing early)."""
    if len(word) < 5:
        return word
    cut = random.randint(3, len(word) - 2)
    return word[:cut]


def generate_synthetic_typos(word: str, n: int = 3) -> List[str]:
    """Generate multiple synthetic typos for a word."""
    generators = [
        generate_keyboard_typo,
        generate_deletion_typo,
        generate_swap_typo,
        generate_insertion_typo,
        generate_double_typo,
        generate_compound_typo,
        generate_truncation_typo,
    ]

    typos = set()
    attempts = 0
    while len(typos) < n and attempts < n * 3:
        generator = random.choice(generators)
        typo = generator(word)
        if typo != word and typo not in typos:
            typos.add(typo)
        attempts += 1

    return list(typos)


def generate_spacing_variants(phrase: str) -> List[str]:
    """Generate spacing-related typo variants for a short phrase.

    Examples:
        "16 gb ram" -> ["16gb ram", "16 gbram", "16gbram"]
        "rtx 4070"  -> ["rtx4070"]
    """
    parts = phrase.split()
    if len(parts) < 2 or len(phrase) > 40:
        # Only for short phrases to avoid crazy combinations
        return []

    variants: Set[str] = set()

    # 1) Remove a single space (merge two neighboring tokens)
    for i in range(len(parts) - 1):
        merged = parts[:]
        merged[i] = merged[i] + merged[i + 1]
        del merged[i + 1]
        variants.add(" ".join(merged))

    # 2) Remove all spaces (fully bunched)
    variants.add("".join(parts))

    return list(variants)


def generate_symbol_variants(token: str) -> List[str]:
    """Generate simple symbol-based variants for model-like tokens.

    Examples:
        "rtx 4070 ti" -> ["rtx-4070-ti", "rtx_4070_ti", "rtx 4070ti", "rtx4070ti"]
        "i9-14900k"   -> ["i9 14900k", "i9_14900k"]
    """
    variants: Set[str] = set()

    base = token.strip()
    if not base:
        return []

    # If contains spaces, generate hyphen/underscore joined versions
    if " " in base:
        parts = base.split()
        variants.add("-".join(parts))
        variants.add("_".join(parts))
        # Merge last two parts if possible: "4070 ti" -> "4070ti"
        if len(parts) >= 2:
            merged_last = parts[:-2] + [parts[-2] + parts[-1]]
            variants.add(" ".join(merged_last))
            variants.add("".join(merged_last))

    # If contains hyphen, also generate space/underscore version
    if "-" in base:
        parts = base.split("-")
        variants.add(" ".join(parts))
        variants.add("_".join(parts))

    return list(variants)


def create_training_examples(
    mappings: List[Tuple[str, str]],
    vocabulary: List[str],
    augment: bool = True,
    augment_samples: int = 5000,
) -> List[Dict]:
    """Create base (mostly single-token) training examples.

    Bu fonksiyon kelime seviyesinde çalışır; cümle ve query odaklı
    örnekler bunun üstüne eklenir (multi-word + sentence generator).
    """
    examples = []
    
    # 1. Add all explicit mappings
    for typo, correct in mappings:
        examples.append({
            "typo": typo,
            "correct": correct,
            "source": "explicit"
        })
    
    print(f"[1] Explicit mappings: {len(examples)}")
    
    # 2. Add identity mappings (correct -> correct) for vocabulary
    # Bu, modelin doğru kelimeleri değiştirmemesini öğretir
    identity_count = min(len(vocabulary), 4000)
    vocab_sample = random.sample(vocabulary, identity_count)
    for word in vocab_sample:
        examples.append({
            "typo": word,
            "correct": word,
            "source": "identity"
        })
        # Also add case variants as identity (model shouldn't break casing)
        if random.random() < 0.3:
            case_var = generate_case_variant(word)
            examples.append({
                "typo": case_var,
                "correct": case_var.lower(),
                "source": "identity_case"
            })

    print(f"[2] After identity mappings: {len(examples)}")

    # 3. Generate synthetic typos if augmentation is enabled
    if augment:
        augment_count = 0
        target = min(augment_samples, len(vocabulary) * 5)

        for word in vocabulary:
            if augment_count >= target:
                break

            # Skip very short words
            if len(word) < 3:
                continue

            # Generate 2-5 synthetic typos per word (more diverse)
            typos = generate_synthetic_typos(word, n=random.randint(2, 5))
            for typo in typos:
                examples.append({
                    "typo": typo,
                    "correct": word,
                    "source": "synthetic"
                })
                augment_count += 1
                if augment_count >= target:
                    break

        print(f"[3] After augmentation: {len(examples)}")
    
    return examples


def format_for_model(examples: List[Dict], model_type: str) -> List[Dict]:
    """Format examples for specific model architectures.

    Contract:
        - input: list of {"typo": str, "correct": str}
        - output (t5/byt5): {"input_text", "target_text"}
        - output (llm): {"messages": [...]} chat format
        - output (simple): {"input", "output", "source"}
    """
    formatted = []
    
    for ex in examples:
        typo = ex["typo"]
        correct = ex["correct"]
        
        if model_type == "t5" or model_type == "byt5":
            # T5/ByT5 format: "correct: <typo>" -> "<correct>"
            formatted.append({
                "input_text": f"correct: {typo}",
                "target_text": correct
            })
        
        elif model_type == "llm":
            # LLM (Qwen/Llama) chat format
            formatted.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a typo correction assistant for e-commerce search queries. Correct the typo in the user's query. Only output the corrected text, nothing else."
                    },
                    {
                        "role": "user", 
                        "content": typo
                    },
                    {
                        "role": "assistant",
                        "content": correct
                    }
                ]
            })
        
        elif model_type == "simple":
            # Simple input-output format
            formatted.append({
                "input": typo,
                "output": correct,
                "source": ex.get("source", "unknown")
            })
    
    return formatted


def save_jsonl(data: List[Dict], filepath: Path):
    """Save data as JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} examples to {filepath}")


def generate_multi_word_queries(
    vocab: ECommerceVocab,
    count: int = 2000,
    max_noise_per_query: int = 2,
    min_keep_words: int = 1,
) -> List[Tuple[str, str]]:
    """Generate realistic multi-word e-commerce search queries.

    Buradaki amaç şuna benzer query'ler üretmek:
        - "16 gb ram laptop"
        - "rtx 4070 gaming laptop for work and school"

    ve bu query'ler içinde **1..max_noise_per_query** kelimeyi bozulmuş halde
    verip, hedefte tamamen doğru halini öğretmek.
    """
    queries: List[Tuple[str, str]] = []

    # Get vocabulary and typo mappings
    typo_mappings = vocab.get_all_typo_mappings()

    # Query templates for electronics (keyword ağırlıklı)
    templates = [
        "{brand} {product}",
        "{brand} {product} {model}",
        "{product} for {brand}",
        "{brand} {product} case",
        "{brand} {product} charger",
        "{brand} {product} screen protector",
        "{product} {spec}",
        "best {product} {year}",
        "{brand} {product} vs {brand2} {product2}",
        "{color} {product}",
        "{brand} {product} {accessory}",
        "cheap {brand} {product}",
        "{brand} {product} price",
        "{brand} {product} review",
        "{product} with {spec}",
        "{brand} {product} {spec}",
    ]

    # Sample data for templates
    brands = [
        "apple", "samsung", "sony", "lg", "dell", "hp", "lenovo", "asus", "acer", "msi",
        "nvidia", "amd", "intel", "logitech", "corsair", "razer", "bose", "jbl",
    ]
    products = [
        "laptop", "phone", "smartphone", "tablet", "headphones", "monitor", "keyboard", "mouse",
        "speaker", "charger", "case", "earbuds", "watch", "tv", "gpu", "cpu",
    ]
    specs = [
        "144hz", "4k", "wireless", "bluetooth", "gaming", "pro", "ultra", "mini", "max",
        "16 gb ram", "32 gb ram", "1tb ssd", "512gb ssd",
    ]
    colors = ["black", "white", "silver", "gold", "blue", "red", "green"]
    accessories = [
        "case", "cover", "charger", "cable", "stand", "mount", "skin", "protector",
    ]
    years = ["2024", "2025"]

    for _ in range(count):
        template = random.choice(templates)

        # Fill template (doğru, temiz query)
        correct_query = template.format(
            brand=random.choice(brands),
            brand2=random.choice(brands),
            product=random.choice(products),
            product2=random.choice(products),
            model=random.choice(["pro", "max", "ultra", "plus", "lite", "mini", "air"]),
            spec=random.choice(specs),
            color=random.choice(colors),
            accessory=random.choice(accessories),
            year=random.choice(years),
        )

        words = correct_query.split()
        if len(words) < max(min_keep_words, 1):
            continue

        # Decide how many words will be noised (at most distinct positions)
        noise_budget = min(max_noise_per_query, max(1, len(words) - min_keep_words))
        positions = list(range(len(words)))
        random.shuffle(positions)
        noise_positions = positions[:noise_budget]

        noisy_words = list(words)
        for idx in noise_positions:
            word = words[idx]

            # 1) Eğer explicit typo mapping varsa onu kullan
            applied = False
            for typo, correct in typo_mappings.items():
                if correct.lower() == word.lower():
                    noisy_words[idx] = typo
                    applied = True
                    break

            # 2) Yoksa sentetik typo üret
            if not applied and len(word) >= 3:
                synthetic = generate_synthetic_typos(word, n=1)
                if synthetic:
                    noisy_words[idx] = synthetic[0]

        typo_query = " ".join(noisy_words)
        if typo_query != correct_query:
            queries.append((typo_query, correct_query))

    return queries


def generate_sentence_queries(
    vocab: ECommerceVocab,
    count: int = 2000,
    max_noise_per_query: int = 3,
    min_keep_words: int = 3,
) -> List[Tuple[str, str]]:
    """Generate more natural, sentence-like queries.

    Örnek hedefler:
        - "i need a 16 gb gaming laptop for my son"
        - "can you show me cheap samsung galaxy s24 cases"
        - "i'm looking for a wireless mouse for office work"

    Çıktı:
        List[(typo_sentence, correct_sentence)]
    """
    queries: List[Tuple[str, str]] = []
    typo_mappings = vocab.get_all_typo_mappings()

    sentence_templates = [
        "i need a {spec} {product} for my {person}",
        "i want to buy a {spec} {product} for {usage}",
        "can you show me {price} {brand} {product} for {usage}",
        "i am looking for a {brand} {product} with {spec}",
        "which {product} should i buy for {usage}",
        "best {spec} {product} for {usage}",
        "i need a {brand} {product} for {usage}",
        "i need a {spec} {brand} {product} for my {person}",
        "i want to buy {brand} {product} for my {person}",
        "can you find a {price} {spec} {product} for {usage}",
    ]

    people = ["son", "daughter", "kid", "wife", "husband", "brother", "sister", "dad", "mom"]
    usages = [
        "gaming", "office work", "school", "university", "video editing", "programming",
        "photo editing", "browsing", "movies", "music",
    ]
    prices = ["cheap", "budget", "mid range", "high end"]

    brands = [
        "apple", "samsung", "sony", "lg", "dell", "hp", "lenovo", "asus", "acer", "msi",
        "nvidia", "amd", "intel",
    ]
    products = [
        "laptop", "gaming laptop", "desktop", "phone", "smartphone", "tablet",
        "monitor", "headphones", "wireless mouse", "mechanical keyboard",
    ]
    specs = [
        "16 gb ram", "32 gb ram", "64 gb ram",
        "1tb ssd", "512gb ssd", "240hz", "144hz", "4k",
        "wireless", "bluetooth", "gaming",
    ]

    for _ in range(count):
        template = random.choice(sentence_templates)
        correct_sentence = template.format(
            brand=random.choice(brands),
            product=random.choice(products),
            spec=random.choice(specs),
            person=random.choice(people),
            usage=random.choice(usages),
            price=random.choice(prices),
        )

        words = correct_sentence.split()
        if len(words) < max(min_keep_words, 1):
            continue

        noise_budget = min(max_noise_per_query, max(1, len(words) - min_keep_words))
        positions = list(range(len(words)))
        random.shuffle(positions)
        noise_positions = positions[:noise_budget]

        noisy_words = list(words)
        for idx in noise_positions:
            word = words[idx]

            # Önce explicit typo mapping dene
            applied = False
            for typo, correct in typo_mappings.items():
                if correct.lower() == word.lower():
                    noisy_words[idx] = typo
                    applied = True
                    break

            # Yoksa sentetik typo
            if not applied and len(word) >= 3:
                synthetic = generate_synthetic_typos(word, n=1)
                if synthetic:
                    noisy_words[idx] = synthetic[0]

        typo_sentence = " ".join(noisy_words)
        if typo_sentence != correct_sentence:
            queries.append((typo_sentence, correct_sentence))

    return queries


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for typo correction")
    parser.add_argument("--augment", action="store_true", help="Generate synthetic typos")
    parser.add_argument("--samples", type=int, default=5000, help="Number of augmented samples")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Evaluation set ratio")
    parser.add_argument("--categories", nargs="+", default=["electronics"], 
                        help="E-commerce categories to include")
    parser.add_argument("--multi-word", type=int, default=2000,
                        help="Number of keyword-style multi-word query examples")
    parser.add_argument("--sentences", type=int, default=2000,
                        help="Number of natural sentence-like query examples")
    parser.add_argument("--max-noise-per-query", type=int, default=3,
                        help="Maximum number of tokens to corrupt in a single query/sentence")
    parser.add_argument("--min-keep-words", type=int, default=1,
                        help="Minimum number of words that must stay clean in a query")
    parser.add_argument("--spacing-variants", action="store_true",
                        help="Add spacing-based typo variants for important phrases (e.g., '16 gb ram' -> '16gb ram')")
    parser.add_argument("--symbol-variants", action="store_true",
                        help="Add symbol-based variants for model tokens (e.g., 'rtx 4070 ti' -> 'rtx-4070-ti')")
    args = parser.parse_args()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  E-Commerce Typo Correction Data Preparation")
    print("=" * 60)
    
    # Initialize e-commerce vocabulary
    ecom_vocab = ECommerceVocab()
    for cat in args.categories:
        ecom_vocab.add_category(cat)
    
    # Print category stats
    stats = ecom_vocab.get_category_stats()
    print("\n  Category Statistics:")
    for cat_name, cat_stats in stats.items():
        print(f"  {cat_name}: {sum(cat_stats.values())} total items")
    
    print("\n" + "=" * 60)
    print("Loading external data sources...")
    print("=" * 60)
    
    # Load existing typo mappings from file
    typo_mappings = []
    typo_file = DATA_DIR / "typo_mappings.txt"
    if typo_file.exists():
        typo_mappings = load_typo_mappings(typo_file)
        print(f"  Loaded {len(typo_mappings)} mappings from typo_mappings.txt")
    
    # Load typo_dataset.csv if available
    csv_file = DATA_DIR / "typo_dataset.csv"
    if csv_file.exists():
        import csv
        csv_count = 0
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    noisy, clean = row[0].strip(), row[1].strip()
                    if noisy and clean and noisy.lower() != "noisy":
                        typo_mappings.append((noisy, clean))
                        csv_count += 1
        print(f"  Loaded {csv_count} pairs from typo_dataset.csv")

    # Add mappings from e-commerce vocab
    ecom_mappings = ecom_vocab.get_all_typo_mappings()
    for typo, correct in ecom_mappings.items():
        typo_mappings.append((typo, correct))
    print(f"  Added {len(ecom_mappings)} mappings from e-commerce vocab")
    
    # Load vocabularies from files
    vocab_files = [
        "brand_products.txt",
        "electronics_vocab.txt", 
        "domain_vocab.txt",
    ]
    
    vocabulary = list(ecom_vocab.get_all_vocabulary())
    print(f"  E-commerce vocabulary: {len(vocabulary)} terms")
    
    for vf in vocab_files:
        filepath = DATA_DIR / vf
        if filepath.exists():
            words = load_vocabulary(filepath)
            vocabulary.extend(words)
            print(f"  Added {len(words)} words from {vf}")
    
    # Remove duplicates
    vocabulary = list(set(vocabulary))
    print(f"\n  Total unique vocabulary: {len(vocabulary)}")
    print(f"  Total typo mappings: {len(typo_mappings)}")
    
    print("\n" + "=" * 60)
    print("Creating training examples...")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # 1) Single-token / kelime seviyesinde örnekler
    # ------------------------------------------------------------------
    examples = create_training_examples(
        typo_mappings,
        vocabulary,
        augment=args.augment,
        augment_samples=args.samples
    )
    
    # ------------------------------------------------------------------
    # 2) Multi-word keyword query örnekleri ("16 gb ram laptop" vb.)
    # ------------------------------------------------------------------
    if args.multi_word > 0:
        print(f"\n[4] Generating {args.multi_word} multi-word queries...")
        multi_word_queries = generate_multi_word_queries(
            ecom_vocab,
            count=args.multi_word,
            max_noise_per_query=args.max_noise_per_query,
            min_keep_words=args.min_keep_words,
        )
        for typo, correct in multi_word_queries:
            examples.append({
                "typo": typo,
                "correct": correct,
                "source": "multi_word"
            })
        print(f"    Added {len(multi_word_queries)} multi-word examples")
    
    # ------------------------------------------------------------------
    # 3) Doğal cümle benzeri query örnekleri
    # ------------------------------------------------------------------
    if args.sentences > 0:
        print(f"\n[4b] Generating {args.sentences} sentence-like queries...")
        sentence_queries = generate_sentence_queries(
            ecom_vocab,
            count=args.sentences,
            max_noise_per_query=args.max_noise_per_query,
            min_keep_words=max(args.min_keep_words, 3),
        )
        for typo, correct in sentence_queries:
            examples.append({
                "typo": typo,
                "correct": correct,
                "source": "sentence"
            })
        print(f"    Added {len(sentence_queries)} sentence examples")

    # ------------------------------------------------------------------
    # 3b) Turkish character normalization examples
    # ------------------------------------------------------------------
    print("\n[5] Adding Turkish character normalization examples...")
    turkish_words = [
        "kılıf", "şarj", "şarjör", "güç", "gömlek", "çanta", "özellik",
        "işlemci", "güncelleme", "görüntü", "çözünürlük", "büyüklük",
        "küçük", "ürün", "değerlendirme", "ödeme", "fiyat", "sipariş",
        "indirim", "ölçü", "renk", "gönderim", "iade", "garanti",
    ]
    turkish_count = 0
    for word in turkish_words:
        for typo, correct in generate_turkish_variants(word):
            examples.append({
                "typo": typo,
                "correct": correct,
                "source": "turkish_normalization"
            })
            turkish_count += 1
    print(f"    Added {turkish_count} Turkish normalization examples")

    # ------------------------------------------------------------------
    # 4) Spacing & symbol variants for critical phrases/models
    # ------------------------------------------------------------------
    if args.spacing_variants or args.symbol_variants:
        print("\n[6] Adding spacing/symbol variants for specs and models...")
        extra_examples = 0

        # a) From ELECTRONICS_MODEL_PATTERNS (models like "rtx 4070 ti")
        for base, models in ELECTRONICS_MODEL_PATTERNS.items():
            for model in models:
                full = f"{base} {model}".strip()

                # Spacing variants (e.g., "rtx 4070" -> "rtx4070")
                if args.spacing_variants:
                    for var in generate_spacing_variants(full):
                        if var != full:
                            examples.append({
                                "typo": var,
                                "correct": full,
                                "source": "spacing_model",
                            })
                            extra_examples += 1

                # Symbol variants (e.g., "rtx 4070 ti" -> "rtx-4070-ti")
                if args.symbol_variants:
                    for var in generate_symbol_variants(full):
                        if var != full:
                            examples.append({
                                "typo": var,
                                "correct": full,
                                "source": "symbol_model",
                            })
                            extra_examples += 1

        # b) A small curated list of spec phrases (RAM/SSD)
        spec_phrases = [
            "8 gb ram",
            "16 gb ram",
            "32 gb ram",
            "64 gb ram",
            "128 gb ssd",
            "256 gb ssd",
            "512 gb ssd",
            "1tb ssd",
            "2tb ssd",
        ]

        if args.spacing_variants:
            for phrase in spec_phrases:
                for var in generate_spacing_variants(phrase):
                    if var != phrase:
                        examples.append({
                            "typo": var,
                            "correct": phrase,
                            "source": "spacing_spec",
                        })
                        extra_examples += 1

        print(f"    Added {extra_examples} spacing/symbol variant examples")
    
    # Add model number examples (critical for e-commerce)
    print("\n[5] Adding model number examples...")
    model_examples = 0
    for base, models in ELECTRONICS_MODEL_PATTERNS.items():
        for model in models:
            full_name = f"{base} {model}"
            examples.append({
                "typo": full_name,
                "correct": full_name,
                "source": "model_identity"
            })
            model_examples += 1
            
            # Generate typo for base word
            if base in ecom_vocab.get_all_typo_mappings().values():
                for typo, correct in ecom_vocab.get_all_typo_mappings().items():
                    if correct == base:
                        examples.append({
                            "typo": f"{typo} {model}",
                            "correct": full_name,
                            "source": "model_typo"
                        })
                        model_examples += 1
                        break
    
    print(f"    Added {model_examples} model number examples")
    
    # Shuffle and split
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - args.eval_ratio))
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]
    
    print(f"\n  Dataset Split:")
    print(f"  Train examples: {len(train_examples)}")
    print(f"  Eval examples: {len(eval_examples)}")
    
    # Count by source
    source_counts = defaultdict(int)
    for ex in examples:
        source_counts[ex.get("source", "unknown")] += 1
    print(f"\n  Examples by source:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")
    
    print("\n" + "=" * 60)
    print("Saving formatted datasets...")
    print("=" * 60)
    
    # Save in different formats
    for model_type in ["simple", "t5", "llm"]:
        train_formatted = format_for_model(train_examples, model_type)
        eval_formatted = format_for_model(eval_examples, model_type)
        
        save_jsonl(train_formatted, OUTPUT_DIR / f"train_{model_type}.jsonl")
        save_jsonl(eval_formatted, OUTPUT_DIR / f"eval_{model_type}.jsonl")
    
    # Save statistics
    stats_file = OUTPUT_DIR / "data_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "total_examples": len(examples),
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "source_counts": dict(source_counts),
            "categories": args.categories,
            "vocabulary_size": len(vocabulary),
            "typo_mappings": len(typo_mappings),
        }, f, indent=2)
    print(f"Saved statistics to {stats_file}")
    
    print("\n" + "=" * 60)
    print("  Data preparation complete!")
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. python train.py --model_name google/byt5-base --epochs 10 --fp16")
    print("  2. python benchmark.py --models byt5")
    print("  3. python export_onnx.py --model_path outputs/byt5-typo/best")


if __name__ == "__main__":
    main()
