#!/usr/bin/env python3
"""
E-Commerce Training Data Augmentation Script
=============================================
Uses the project's own curated data files to build a much stronger dataset:
  - typo_mappings.txt (790 curated typo->correct pairs)
  - brand_products.txt (754 lines of brand/product vocabulary)
  - electronics_vocab.txt (928 lines of electronics terms)
  - train_final.jsonl (current training data)

Generates realistic e-commerce search query corrections using:
  1. Curated typo mappings in realistic query contexts
  2. QWERTY keyboard-based typos on brand/product vocabulary
  3. Multi-word query augmentation (brand + product + modifier)
  4. Identity pairs for overcorrection prevention

Output: data/train_final_v2.jsonl, data/eval_final_v2.jsonl

Usage:
  cd Correction/fine_tune/t5-large-v2-1
  python augment_training_data.py
  python augment_training_data.py --target 220000
"""

import argparse
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

# ── Paths ──
SCRIPT_DIR = Path(__file__).parent
CORRECTION_DIR = SCRIPT_DIR.parent.parent  # up to Correction/
DATA_DIR = CORRECTION_DIR / "data"         # Correction/data/

TYPO_MAPPINGS = DATA_DIR / "typo_mappings.txt"
BRAND_PRODUCTS = DATA_DIR / "brand_products.txt"
ELECTRONICS_VOCAB = DATA_DIR / "electronics_vocab.txt"
DOMAIN_VOCAB = DATA_DIR / "domain_vocab.txt"

# Training data lives in our own data/ subfolder
LOCAL_DATA_DIR = SCRIPT_DIR / "data"
TRAIN_FILE = LOCAL_DATA_DIR / "train_byt5.jsonl"
EVAL_FILE = LOCAL_DATA_DIR / "eval_byt5.jsonl"

# ── QWERTY adjacency ──
QWERTY = {
    'q': 'wa', 'w': 'qeas', 'e': 'wrds', 'r': 'etdf', 't': 'ryfg',
    'y': 'tugh', 'u': 'yihj', 'i': 'uojk', 'o': 'ipkl', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
    'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huiknm', 'k': 'jiolm',
    'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv',
    'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
}


# ═══════════════════════════════════════════════════════════
# LOADERS
# ═══════════════════════════════════════════════════════════

def load_vocab_file(path):
    """Load a vocabulary file, skip comments and blanks."""
    terms = []
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return terms
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("="):
                continue
            terms.append(line.lower())
    return terms


def load_typo_mappings(path):
    """Load typo,correct pairs from typo_mappings.txt."""
    mappings = []
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return mappings
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("="):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                typo, correct = parts[0].strip(), parts[1].strip()
                if typo and correct and typo != correct:
                    mappings.append((typo, correct))
    return mappings


def load_jsonl(path):
    """Load JSONL file."""
    data = []
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return data
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ═══════════════════════════════════════════════════════════
# TYPO GENERATORS
# ═══════════════════════════════════════════════════════════

def typo_adjacent_key(word):
    """Replace one char with QWERTY neighbor."""
    if len(word) < 3:
        return None
    pos = random.randint(1, len(word) - 2)
    c = word[pos].lower()
    if c in QWERTY and QWERTY[c]:
        return word[:pos] + random.choice(QWERTY[c]) + word[pos + 1:]
    return None


def typo_delete_char(word):
    """Delete one character."""
    if len(word) < 4:
        return None
    for i in range(len(word) - 1):
        if word[i] == word[i + 1]:
            return word[:i] + word[i + 1:]
    pos = random.randint(1, len(word) - 2)
    return word[:pos] + word[pos + 1:]


def typo_swap_adjacent(word):
    """Swap two adjacent characters."""
    if len(word) < 3:
        return None
    pos = random.randint(0, len(word) - 2)
    return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]


def typo_double_char(word):
    """Double a character."""
    if len(word) < 3:
        return None
    pos = random.randint(0, len(word) - 1)
    return word[:pos] + word[pos] + word[pos:]


def typo_insert_random(word):
    """Insert a random character."""
    pos = random.randint(0, len(word))
    c = random.choice(string.ascii_lowercase)
    return word[:pos] + c + word[pos:]


def generate_typo(word):
    """Generate one realistic typo for a word."""
    methods = [typo_adjacent_key, typo_delete_char, typo_swap_adjacent,
               typo_double_char, typo_insert_random]
    random.shuffle(methods)
    for m in methods:
        r = m(word)
        if r and r != word and len(r) >= 2:
            return r
    return None


def make_query_typo(query):
    """Introduce 1 typo into a multi-word query."""
    words = query.split()
    candidates = [(i, w) for i, w in enumerate(words) if len(w) >= 3 and w.isalpha()]
    if not candidates:
        return None
    idx, word = random.choice(candidates)
    typo = generate_typo(word)
    if not typo:
        return None
    words[idx] = typo
    return " ".join(words)


# ═══════════════════════════════════════════════════════════
# BRAND-PRODUCT CONTEXT MAP
# ═══════════════════════════════════════════════════════════

BRAND_CONTEXT = {
    # Laptops & Computers
    "asus": ["laptop", "monitor", "motherboard", "router", "graphics card", "rog", "tuf gaming", "zenbook", "vivobook"],
    "dell": ["laptop", "monitor", "desktop", "xps", "inspiron", "alienware", "latitude"],
    "hp": ["laptop", "printer", "monitor", "omen", "pavilion", "spectre", "envy", "victus"],
    "lenovo": ["laptop", "tablet", "thinkpad", "ideapad", "legion", "yoga", "monitor"],
    "acer": ["laptop", "monitor", "chromebook", "predator", "nitro", "aspire", "swift"],
    "msi": ["laptop", "monitor", "motherboard", "graphics card", "katana", "stealth", "raider"],
    "razer": ["mouse", "keyboard", "headset", "laptop", "blade", "deathadder", "viper", "huntsman"],
    "apple": ["iphone", "macbook", "airpods", "ipad", "watch", "imac", "mac mini"],
    "samsung": ["phone", "galaxy", "monitor", "tv", "ssd", "earbuds", "tablet", "galaxy buds"],
    "microsoft": ["surface", "xbox", "keyboard", "mouse", "windows"],
    # GPUs & CPUs
    "nvidia": ["rtx 4080", "rtx 4090", "rtx 4070", "rtx 3080", "rtx 3060", "geforce", "gpu", "graphics card"],
    "amd": ["ryzen 5", "ryzen 7", "ryzen 9", "radeon", "rx 7900", "rx 7800", "processor", "cpu"],
    "intel": ["core i5", "core i7", "core i9", "processor", "cpu", "arc", "nuc"],
    # Peripherals
    "logitech": ["mouse", "keyboard", "webcam", "headset", "g pro", "mx master", "g502"],
    "corsair": ["keyboard", "mouse", "headset", "ram", "k70", "k100", "void", "vengeance"],
    "steelseries": ["headset", "mouse", "keyboard", "mousepad", "arctis", "apex", "aerox", "rival"],
    "hyperx": ["headset", "keyboard", "mouse", "ram", "cloud", "alloy", "pulsefire"],
    # Audio
    "sony": ["headphones", "earbuds", "tv", "playstation", "camera", "speaker"],
    "bose": ["headphones", "speaker", "earbuds", "soundbar", "noise cancelling"],
    "sennheiser": ["headphones", "earbuds", "microphone", "headset"],
    "jbl": ["speaker", "headphones", "earbuds", "soundbar"],
    # Storage
    "kingston": ["ram", "ssd", "memory card", "flash drive"],
    "seagate": ["hard drive", "hdd", "ssd", "external drive"],
    "sandisk": ["ssd", "memory card", "flash drive", "usb"],
    "crucial": ["ram", "ssd", "memory"],
    "western digital": ["hard drive", "ssd", "hdd", "wd"],
    # Other
    "anker": ["charger", "power bank", "cable", "earbuds", "speaker"],
    "lg": ["monitor", "tv", "phone", "soundbar"],
}

MODIFIERS = [
    "pro", "max", "mini", "plus", "ultra", "lite", "2024", "2025", "new",
    "latest", "premium", "budget", "cheap", "best", "gaming", "wireless",
    "bluetooth", "portable", "black", "white", "silver", "red", "blue", "pink",
]

QUERY_TEMPLATES = [
    "{brand} {product}",
    "{brand} {product} {mod}",
    "best {brand} {product}",
    "{brand} {product} price",
    "cheap {brand} {product}",
    "{brand} {product} review",
    "{brand} {product} sale",
    "{product} {brand}",
    "{brand} {product} under 500",
    "{brand} {product} vs",
]


# ═══════════════════════════════════════════════════════════
# AUGMENTATION ENGINE
# ═══════════════════════════════════════════════════════════

def augment_from_typo_mappings(mappings, electronics_terms, seen_pairs):
    """
    Use curated typo mappings in realistic query contexts.
    E.g., mapping (nvidai, nvidia) -> "nvidai rtx 4080" -> "nvidia rtx 4080"
    """
    examples = []

    correct_to_typos = defaultdict(list)
    for typo, correct in mappings:
        correct_to_typos[correct].append(typo)

    electronics_set = set(electronics_terms)

    for correct, typos in correct_to_typos.items():
        for typo in typos:
            pair = (f"correct: {typo}", correct)
            if pair not in seen_pairs:
                examples.append({
                    "input_text": pair[0],
                    "target_text": correct,
                    "category": "curated_typo_standalone"
                })
                seen_pairs.add(pair)

        if correct in BRAND_CONTEXT:
            products = BRAND_CONTEXT[correct]
            for typo in typos:
                for product in products:
                    for template in random.sample(QUERY_TEMPLATES, min(4, len(QUERY_TEMPLATES))):
                        mod = random.choice(MODIFIERS)
                        correct_query = template.format(brand=correct, product=product, mod=mod)
                        typo_query = correct_query.replace(correct, typo, 1)
                        if typo_query != correct_query:
                            pair = (f"correct: {typo_query}", correct_query)
                            if pair not in seen_pairs:
                                examples.append({
                                    "input_text": pair[0],
                                    "target_text": correct_query,
                                    "category": "curated_typo_in_context"
                                })
                                seen_pairs.add(pair)

        elif correct in electronics_set:
            for typo in typos:
                contexts = [
                    f"best {typo}", f"{typo} sale", f"cheap {typo}",
                    f"{typo} review", f"gaming {typo}", f"wireless {typo}",
                ]
                correct_contexts = [
                    f"best {correct}", f"{correct} sale", f"cheap {correct}",
                    f"{correct} review", f"gaming {correct}", f"wireless {correct}",
                ]
                for tc, cc in zip(contexts, correct_contexts):
                    pair = (f"correct: {tc}", cc)
                    if pair not in seen_pairs:
                        examples.append({
                            "input_text": pair[0],
                            "target_text": cc,
                            "category": "curated_typo_in_context"
                        })
                        seen_pairs.add(pair)

    return examples


def augment_brand_product_queries(seen_pairs, count=15000):
    """Generate realistic brand+product queries with typos."""
    examples = []

    for _ in range(count * 3):
        brand = random.choice(list(BRAND_CONTEXT.keys()))
        product = random.choice(BRAND_CONTEXT[brand])
        template = random.choice(QUERY_TEMPLATES)
        mod = random.choice(MODIFIERS)

        correct_query = template.format(brand=brand, product=product, mod=mod)

        r = random.random()
        if r < 0.4:
            brand_typo = generate_typo(brand)
            if brand_typo:
                typo_query = correct_query.replace(brand, brand_typo, 1)
            else:
                continue
        elif r < 0.8:
            typo_query = make_query_typo(correct_query)
            if not typo_query:
                continue
        else:
            typo_query = make_query_typo(correct_query)
            if not typo_query:
                continue

        if typo_query == correct_query:
            continue

        pair = (f"correct: {typo_query}", correct_query)
        if pair not in seen_pairs:
            examples.append({
                "input_text": pair[0],
                "target_text": correct_query,
                "category": "aug_ecom_query_typo"
            })
            seen_pairs.add(pair)

        if len(examples) >= count:
            break

    return examples


def augment_electronics_vocab_typos(vocab_terms, seen_pairs, count=5000):
    """Generate typos for standalone electronics terms."""
    examples = []
    terms = [t for t in vocab_terms if len(t) >= 4 and t.isalpha()]

    for _ in range(count * 3):
        term = random.choice(terms)
        typo = generate_typo(term)
        if typo and typo != term:
            pair = (f"correct: {typo}", term)
            if pair not in seen_pairs:
                examples.append({
                    "input_text": pair[0],
                    "target_text": term,
                    "category": "aug_electronics_vocab_typo"
                })
                seen_pairs.add(pair)

        if len(examples) >= count:
            break

    return examples


def augment_identity_pairs(brand_terms, product_terms, seen_pairs, count=5000):
    """Generate identity pairs to prevent overcorrection."""
    examples = []

    for brand in BRAND_CONTEXT.keys():
        pair = (f"correct: {brand}", brand)
        if pair not in seen_pairs:
            examples.append({
                "input_text": pair[0],
                "target_text": brand,
                "category": "identity_ecom_brand"
            })
            seen_pairs.add(pair)

    for _ in range(count):
        brand = random.choice(list(BRAND_CONTEXT.keys()))
        product = random.choice(BRAND_CONTEXT[brand])
        template = random.choice(QUERY_TEMPLATES)
        mod = random.choice(MODIFIERS)
        query = template.format(brand=brand, product=product, mod=mod)

        pair = (f"correct: {query}", query)
        if pair not in seen_pairs:
            examples.append({
                "input_text": pair[0],
                "target_text": query,
                "category": "identity_ecom_query"
            })
            seen_pairs.add(pair)

    for term in random.sample(product_terms, min(1000, len(product_terms))):
        pair = (f"correct: {term}", term)
        if pair not in seen_pairs:
            examples.append({
                "input_text": pair[0],
                "target_text": term,
                "category": "identity_electronics_term"
            })
            seen_pairs.add(pair)

    return examples


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Augment e-commerce training data")
    parser.add_argument("--target", type=int, default=220000,
                        help="Target total training examples (default: 220000)")
    args = parser.parse_args()

    print("=" * 70)
    print("E-Commerce Training Data Augmentation")
    print("=" * 70)

    # Ensure output dir exists
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load project data
    print("\n[1/6] Loading project data...")
    typo_mappings = load_typo_mappings(TYPO_MAPPINGS)
    brand_products = load_vocab_file(BRAND_PRODUCTS)
    electronics_vocab = load_vocab_file(ELECTRONICS_VOCAB)
    domain_vocab = load_vocab_file(DOMAIN_VOCAB)

    print(f"  Typo mappings: {len(typo_mappings)} pairs")
    print(f"  Brand products: {len(brand_products)} terms")
    print(f"  Electronics vocab: {len(electronics_vocab)} terms")
    print(f"  Domain vocab: {len(domain_vocab)} terms")

    # Load existing training data
    print("\n[2/6] Loading existing training data...")
    train_data = load_jsonl(TRAIN_FILE)
    eval_data = load_jsonl(EVAL_FILE)
    print(f"  Train: {len(train_data):,}")
    print(f"  Eval: {len(eval_data):,}")

    # Build seen pairs set
    seen_pairs = set()
    for ex in train_data:
        seen_pairs.add((ex["input_text"], ex["target_text"]))
    for ex in eval_data:
        seen_pairs.add((ex["input_text"], ex["target_text"]))

    all_electronics = list(set(brand_products + electronics_vocab + domain_vocab))

    # ── Augmentation ──
    all_augmented = []

    print("\n[3/6] Augmenting from curated typo mappings...")
    curated = augment_from_typo_mappings(typo_mappings, all_electronics, seen_pairs)
    all_augmented.extend(curated)
    print(f"  Generated: {len(curated):,}")

    print("\n[4/6] Generating brand+product query typos...")
    query_typos = augment_brand_product_queries(seen_pairs, count=15000)
    all_augmented.extend(query_typos)
    print(f"  Generated: {len(query_typos):,}")

    print("\n[5/6] Generating electronics vocab typos...")
    vocab_typos = augment_electronics_vocab_typos(all_electronics, seen_pairs, count=5000)
    all_augmented.extend(vocab_typos)
    print(f"  Generated: {len(vocab_typos):,}")

    print("\n[6/6] Generating identity pairs...")
    identity = augment_identity_pairs(brand_products, all_electronics, seen_pairs, count=5000)
    all_augmented.extend(identity)
    print(f"  Generated: {len(identity):,}")

    print(f"\n  TOTAL augmented: {len(all_augmented):,}")

    # ── Split augmented into train/eval (90/10) ──
    random.shuffle(all_augmented)
    eval_split = int(len(all_augmented) * 0.1)
    aug_eval = all_augmented[:eval_split]
    aug_train = all_augmented[eval_split:]

    # ── Combine ──
    final_train = train_data + aug_train
    final_eval = eval_data + aug_eval
    random.shuffle(final_train)
    random.shuffle(final_eval)

    # ── Trim to target if needed ──
    if len(final_train) > args.target:
        identity_indices = [i for i, ex in enumerate(final_train)
                           if ex["input_text"].replace("correct: ", "", 1) == ex["target_text"]]
        remove_count = len(final_train) - args.target
        if remove_count > 0 and identity_indices:
            random.shuffle(identity_indices)
            remove_set = set(identity_indices[:remove_count])
            final_train = [ex for i, ex in enumerate(final_train) if i not in remove_set]

    # ── Stats ──
    total = len(final_train)
    id_count = sum(1 for ex in final_train
                   if ex["input_text"].replace("correct: ", "", 1) == ex["target_text"])
    corr_count = total - id_count
    cats = Counter(ex["category"] for ex in final_train)

    print(f"\n{'=' * 70}")
    print(f"FINAL DATASET")
    print(f"{'=' * 70}")
    print(f"Train: {total:,}")
    print(f"  Identity:   {id_count:,} ({100 * id_count / total:.1f}%)")
    print(f"  Correction: {corr_count:,} ({100 * corr_count / total:.1f}%)")
    print(f"Eval: {len(final_eval):,}")
    print(f"\nCategories:")
    for cat, cnt in cats.most_common():
        print(f"  {cat:<35s} {cnt:>7,} ({100 * cnt / total:.1f}%)")

    # ── Show examples ──
    print(f"\n--- Sample augmented corrections ---")
    aug_corrections = [ex for ex in all_augmented
                       if ex["input_text"].replace("correct: ", "", 1) != ex["target_text"]]
    random.shuffle(aug_corrections)
    for ex in aug_corrections[:25]:
        q = ex["input_text"].replace("correct: ", "")
        t = ex["target_text"]
        print(f"  [{ex['category']:<30s}] '{q}' -> '{t}'")

    # ── Save ──
    out_train = LOCAL_DATA_DIR / "train_final_v2.jsonl"
    out_eval = LOCAL_DATA_DIR / "eval_final_v2.jsonl"

    with open(out_train, "w", encoding="utf-8") as f:
        for ex in final_train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(out_eval, "w", encoding="utf-8") as f:
        for ex in final_eval:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved:")
    print(f"  {out_train} ({len(final_train):,} examples)")
    print(f"  {out_eval} ({len(final_eval):,} examples)")
    print(f"\nDone! Upload the data/ folder to Google Drive under t5-large-v2-1/")


if __name__ == "__main__":
    main()
