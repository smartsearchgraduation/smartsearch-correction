#!/usr/bin/env python3
"""
E-Commerce Training Data Builder for ByT5-Large Spelling Correction
====================================================================

Builds the FINAL training dataset by combining:
  1. Real general English corrections  (Birkbeck, Wikipedia, Norvig)
  2. Real e-commerce identity pairs    (Amazon ESCI queries)
  3. E-commerce brand/product typos    (mined from ESCI product titles)
  4. Natural typo mining               (ESCI query ↔ product title alignment)

Output: ready-to-train T5-style JSONL
  {"input_text": "correct: samsng galaxy", "target_text": "samsung galaxy", "category": "..."}

Usage:
  python build_ecommerce_training_data.py
  python build_ecommerce_training_data.py --total 200000
  python build_ecommerce_training_data.py --esci-sample 500000  (limit ESCI rows for speed)
"""

import argparse
import json
import logging
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── QWERTY keyboard neighbours (for realistic typo generation) ──────────────
QWERTY = {
    'q': 'wa', 'w': 'qeas', 'e': 'wrsd', 'r': 'etdf', 't': 'ryfg',
    'y': 'tugh', 'u': 'yihj', 'i': 'uojk', 'o': 'ipkl', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
    'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huiknm', 'k': 'jiolm',
    'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv',
    'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
}


# ── Typo generators ─────────────────────────────────────────────────────────
def typo_keyboard(word: str) -> str:
    if len(word) < 2: return word
    i = random.randint(0, len(word) - 1)
    ch = word[i].lower()
    if ch in QWERTY:
        return word[:i] + random.choice(QWERTY[ch]) + word[i+1:]
    return word

def typo_delete(word: str) -> str:
    if len(word) < 3: return word
    i = random.randint(1, len(word) - 2)
    return word[:i] + word[i+1:]

def typo_swap(word: str) -> str:
    if len(word) < 3: return word
    i = random.randint(0, len(word) - 2)
    return word[:i] + word[i+1] + word[i] + word[i+2:]

def typo_insert(word: str) -> str:
    i = random.randint(0, len(word))
    ch = random.choice(string.ascii_lowercase)
    return word[:i] + ch + word[i:]

def typo_double(word: str) -> str:
    if len(word) < 2: return word
    i = random.randint(0, len(word) - 1)
    return word[:i] + word[i] + word[i:]

def typo_compound(word: str) -> str:
    """Apply 2 typo operations."""
    if len(word) < 4: return word
    fns = [typo_keyboard, typo_delete, typo_swap, typo_insert]
    f1, f2 = random.sample(fns, 2)
    r = f1(word)
    return f2(r) if r != word else r

def make_typo(word: str) -> str:
    """Generate one realistic typo for a word."""
    fns = [typo_keyboard, typo_delete, typo_swap, typo_insert, typo_double]
    weights = [3, 2, 2, 1, 1]
    fn = random.choices(fns, weights=weights, k=1)[0]
    result = fn(word)
    return result if result != word else typo_keyboard(word)

def make_query_typo(query: str) -> str:
    """Introduce 1–2 typos into a multi-word query."""
    words = query.split()
    if not words:
        return query
    # Pick 1–2 words to corrupt (prefer longer words)
    eligible = [(i, w) for i, w in enumerate(words) if len(w) >= 3]
    if not eligible:
        return query
    n_corrupt = min(random.choices([1, 2], weights=[4, 1], k=1)[0], len(eligible))
    targets = random.sample(eligible, n_corrupt)
    for idx, w in targets:
        words[idx] = make_typo(w)
    return " ".join(words)


# ── Edit distance (for mining) ───────────────────────────────────────────────
def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b): return edit_distance(b, a)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(ca != cb)))
        prev = curr
    return prev[-1]


# ═════════════════════════════════════════════════════════════════════════════
# ESCI Processor – extract e-commerce vocabulary + mine natural corrections
# ═════════════════════════════════════════════════════════════════════════════
class ESCIProcessor:
    """Process Amazon ESCI dataset to extract e-commerce vocabulary and natural typo pairs."""

    def __init__(self, sample_limit: Optional[int] = None):
        self.sample_limit = sample_limit
        self.brands: Set[str] = set()
        self.product_terms: Set[str] = set()
        self.model_numbers: Set[str] = set()
        self.full_queries: Set[str] = set()
        self.natural_corrections: List[Tuple[str, str]] = []  # (typo, correct)

    def load_and_process(self):
        """Load ESCI and extract everything we need."""
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("pip install datasets  gerekli!")
            return

        logger.info("ESCI yükleniyor...")
        ds = load_dataset("tasksource/esci", split="train")

        if self.sample_limit and self.sample_limit < len(ds):
            logger.info(f"ESCI'den {self.sample_limit} satır örnekleniyor...")
            indices = random.sample(range(len(ds)), self.sample_limit)
            ds = ds.select(indices)

        logger.info(f"ESCI: {len(ds)} satır işlenecek")

        import pandas as pd
        df = ds.to_pandas()

        # ── 1. Unique queries → identity pairs & vocabulary ──
        queries = df["query"].dropna().str.strip().str.lower().unique()
        self.full_queries = set(q for q in queries if 2 < len(q) < 60)
        logger.info(f"Unique queries: {len(self.full_queries)}")

        # ── 2. Product titles → brand + product vocabulary ──
        titles = df["product_title"].dropna().str.strip().unique()
        logger.info(f"Unique product titles: {len(titles)}")

        brand_counter = Counter()
        term_counter = Counter()
        model_pattern = re.compile(r'\b([A-Z]{1,5}[\-]?\d{2,5}[A-Za-z]*)\b')

        for title in tqdm(titles[:200000], desc="Parsing product titles"):
            t = str(title)

            # Extract model numbers: "RTX 4070", "i9-14900K", "A15", etc.
            for m in model_pattern.findall(t):
                if len(m) >= 3:
                    self.model_numbers.add(m.lower())

            # First 1-3 words are often brand names
            words = t.split()
            if words:
                brand_counter[words[0].lower()] += 1
            if len(words) >= 2:
                brand_counter[f"{words[0]} {words[1]}".lower()] += 1

            # All meaningful words → product terms
            for w in words:
                w_clean = re.sub(r'[^a-z0-9]', '', w.lower())
                if len(w_clean) >= 3 and not w_clean.isdigit():
                    term_counter[w_clean] += 1

        # Keep brands that appear 10+ times (real brands)
        self.brands = set(b for b, c in brand_counter.items()
                         if c >= 10 and len(b) >= 2 and not b.isdigit())
        logger.info(f"Extracted brands: {len(self.brands)}")

        # Keep product terms appearing 20+ times
        self.product_terms = set(t for t, c in term_counter.items()
                                if c >= 20 and len(t) >= 3)
        logger.info(f"Extracted product terms: {len(self.product_terms)}")
        logger.info(f"Extracted model numbers: {len(self.model_numbers)}")

        # ── 3. Natural typo mining: compare queries to product title words ──
        self._mine_natural_corrections(df)

        logger.info(f"Natural corrections found: {len(self.natural_corrections)}")

    def _mine_natural_corrections(self, df):
        """
        Find natural typo→correct pairs by comparing queries to product titles.
        If a query word is close (edit distance 1-2) to a product title word,
        it's likely a natural misspelling.
        """
        import pandas as pd

        logger.info("Natural typo mining: query ↔ product title karşılaştırılıyor...")

        # Build vocabulary of correct product words (from titles)
        correct_vocab: Dict[str, int] = defaultdict(int)
        titles = df["product_title"].dropna().str.strip().unique()
        for title in titles[:100000]:
            for w in str(title).lower().split():
                w_clean = re.sub(r'[^a-z]', '', w)
                if len(w_clean) >= 3:
                    correct_vocab[w_clean] += 1

        # Only keep words appearing 50+ times (definitely correct)
        known_correct = {w for w, c in correct_vocab.items() if c >= 50}
        logger.info(f"Known correct vocabulary: {len(known_correct)} words")

        # Check each unique query word against known vocabulary
        query_words: Dict[str, int] = defaultdict(int)
        queries = df["query"].dropna().str.strip().str.lower().unique()
        for q in queries:
            for w in q.split():
                w_clean = re.sub(r'[^a-z]', '', w)
                if len(w_clean) >= 3 and w_clean not in known_correct:
                    query_words[w_clean] += 1

        # For each unknown query word, find closest known word
        mined = set()
        unknown_words = [(w, c) for w, c in query_words.items() if c >= 2]
        logger.info(f"Unknown query words to check: {len(unknown_words)}")

        for unknown, count in tqdm(unknown_words[:10000], desc="Mining natural typos"):
            best_match = None
            best_dist = 999
            for correct in known_correct:
                # Quick length filter
                if abs(len(unknown) - len(correct)) > 2:
                    continue
                d = edit_distance(unknown, correct)
                if 0 < d <= 2 and d < best_dist:
                    best_dist = d
                    best_match = correct

            if best_match and (unknown, best_match) not in mined:
                mined.add((unknown, best_match))
                self.natural_corrections.append((unknown, best_match))

        logger.info(f"Mined {len(self.natural_corrections)} natural typo→correct pairs")


# ═════════════════════════════════════════════════════════════════════════════
# Training Data Builder – combines all sources into final dataset
# ═════════════════════════════════════════════════════════════════════════════
class TrainingDataBuilder:
    """Build the final balanced training dataset."""

    def __init__(self, total_target: int = 200_000):
        self.total_target = total_target
        self.examples: List[Dict] = []
        self.stats = defaultdict(int)

    def add_real_general_corrections(self, real_pairs_path: Path, max_count: Optional[int] = None):
        """Load real general English corrections from combined_real_pairs.jsonl"""
        if not real_pairs_path.exists():
            logger.warning(f"{real_pairs_path} bulunamadı – atlanıyor")
            return

        corrections = []
        identities = []

        with open(real_pairs_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    pair = json.loads(line.strip())
                    m, c, src = pair["misspelled"], pair["correct"], pair["source"]
                    if m == c:
                        identities.append((m, c, src))
                    else:
                        corrections.append((m, c, src))
                except (json.JSONDecodeError, KeyError):
                    continue

        logger.info(f"Real data loaded: {len(corrections)} corrections, {len(identities)} identities")

        # Add corrections
        limit = max_count or len(corrections)
        for m, c, src in corrections[:limit]:
            self.examples.append({
                "input_text": f"correct: {m}",
                "target_text": c,
                "category": f"real_general_{src}"
            })
            self.stats["real_general"] += 1

        # Add identities
        for m, c, src in identities[:limit]:
            self.examples.append({
                "input_text": f"correct: {m}",
                "target_text": c,
                "category": f"identity_{src}"
            })
            self.stats["identity_real"] += 1

    def add_esci_identities(self, esci: ESCIProcessor, count: int):
        """Add real Amazon shopping queries as identity pairs."""
        queries = list(esci.full_queries)
        random.shuffle(queries)

        added = 0
        for q in queries:
            if added >= count:
                break
            if len(q) < 60:
                self.examples.append({
                    "input_text": f"correct: {q}",
                    "target_text": q,
                    "category": "identity_esci"
                })
                added += 1
        self.stats["identity_esci"] += added
        logger.info(f"ESCI identity pairs added: {added}")

    def add_natural_corrections(self, esci: ESCIProcessor):
        """Add naturally mined typo→correct pairs from ESCI."""
        for typo, correct in esci.natural_corrections:
            self.examples.append({
                "input_text": f"correct: {typo}",
                "target_text": correct,
                "category": "natural_mined"
            })
            # Also add the correct form as identity (hard negative)
            self.examples.append({
                "input_text": f"correct: {correct}",
                "target_text": correct,
                "category": "identity_hard_negative"
            })
        self.stats["natural_mined"] += len(esci.natural_corrections)
        self.stats["identity_hard_negative"] += len(esci.natural_corrections)
        logger.info(f"Natural mined pairs added: {len(esci.natural_corrections)}")

    def add_ecommerce_typos(self, esci: ESCIProcessor, count: int):
        """Generate e-commerce specific typo pairs from ESCI vocabulary."""
        added = 0

        # ── Brand typos (most important!) ──
        brands = list(esci.brands)
        brand_target = int(count * 0.40)
        logger.info(f"Generating {brand_target} brand typo pairs from {len(brands)} brands...")
        for _ in range(brand_target):
            brand = random.choice(brands)
            typo = make_typo(brand) if len(brand.split()) == 1 else make_query_typo(brand)
            if typo != brand:
                self.examples.append({
                    "input_text": f"correct: {typo}",
                    "target_text": brand,
                    "category": "ecom_brand_typo"
                })
                added += 1
        self.stats["ecom_brand_typo"] += added

        # ── Product term typos ──
        terms = list(esci.product_terms)
        term_target = int(count * 0.25)
        term_added = 0
        for _ in range(term_target):
            term = random.choice(terms)
            typo = make_typo(term)
            if typo != term:
                self.examples.append({
                    "input_text": f"correct: {typo}",
                    "target_text": term,
                    "category": "ecom_product_typo"
                })
                term_added += 1
        self.stats["ecom_product_typo"] += term_added

        # ── Full query typos (multi-word, most realistic) ──
        queries = list(esci.full_queries)
        query_target = int(count * 0.30)
        query_added = 0
        for _ in range(query_target):
            q = random.choice(queries)
            typo_q = make_query_typo(q)
            if typo_q != q:
                self.examples.append({
                    "input_text": f"correct: {typo_q}",
                    "target_text": q,
                    "category": "ecom_query_typo"
                })
                query_added += 1
        self.stats["ecom_query_typo"] += query_added

        # ── Model number variants ──
        models = list(esci.model_numbers)
        if models:
            model_target = int(count * 0.05)
            model_added = 0
            for _ in range(model_target):
                m = random.choice(models)
                typo = make_typo(m)
                if typo != m:
                    self.examples.append({
                        "input_text": f"correct: {typo}",
                        "target_text": m,
                        "category": "ecom_model_typo"
                    })
                    model_added += 1
            self.stats["ecom_model_typo"] += model_added

        total_ecom = added + term_added + query_added
        logger.info(f"E-commerce typo pairs generated: {total_ecom}")

    def add_hard_identity_phrases(self):
        """Add common e-commerce phrases that MUST NOT be changed.
        These are the 'hard negatives' — phrases the model tends to overcorrect."""

        phrases = [
            # Sports
            "sport goods", "sport bra", "sports shoes", "sport watch",
            "running shoes", "running shorts", "running socks",
            "yoga mat", "yoga pants", "yoga block",
            "gym bag", "gym gloves", "gym shorts",
            # Electronics
            "wireless mouse", "wireless keyboard", "wireless earbuds",
            "wired headphones", "gaming mouse", "gaming keyboard",
            "mechanical keyboard", "bluetooth speaker", "smart watch",
            "phone case", "phone charger", "screen protector",
            "usb cable", "hdmi cable", "power bank",
            "laptop bag", "laptop stand", "laptop sleeve",
            "monitor stand", "mouse pad", "desk lamp",
            "web camera", "ring light", "memory card",
            # Fashion
            "leather jacket", "denim jacket", "winter coat",
            "cotton shirt", "silk dress", "wool sweater",
            "high heels", "flat shoes", "ankle boots",
            # Home
            "coffee maker", "air fryer", "vacuum cleaner",
            "water bottle", "lunch box", "cutting board",
            "shower head", "bath towel", "bed sheet",
            "wall clock", "desk organizer", "storage box",
            # Beauty
            "face cream", "body lotion", "hair dryer",
            "nail polish", "lip balm", "sun screen",
            # Common short queries
            "shoes", "laptop", "phone", "headphones", "camera",
            "keyboard", "mouse", "monitor", "tablet", "watch",
            "charger", "cable", "case", "bag", "stand",
            "shirt", "pants", "jacket", "dress", "shoes",
            "cream", "lotion", "shampoo", "soap", "brush",
        ]

        for phrase in phrases:
            self.examples.append({
                "input_text": f"correct: {phrase}",
                "target_text": phrase,
                "category": "identity_hard_phrase"
            })
        self.stats["identity_hard_phrase"] += len(phrases)
        logger.info(f"Hard identity phrases added: {len(phrases)}")

    def balance_and_finalize(self):
        """Balance dataset to target composition and finalize."""
        random.shuffle(self.examples)

        # Trim to target if needed
        if len(self.examples) > self.total_target:
            # Keep all identity and hard phrases, trim others proportionally
            identities = [e for e in self.examples if "identity" in e["category"]]
            corrections = [e for e in self.examples if "identity" not in e["category"]]

            target_corrections = self.total_target - len(identities)
            if target_corrections > 0:
                corrections = corrections[:target_corrections]
            self.examples = identities + corrections

        random.shuffle(self.examples)
        logger.info(f"Final dataset size: {len(self.examples)}")

    def save(self, output_dir: Path, eval_ratio: float = 0.1):
        """Save train and eval splits as T5-format JSONL."""
        output_dir.mkdir(parents=True, exist_ok=True)

        random.shuffle(self.examples)
        split_idx = int(len(self.examples) * (1 - eval_ratio))
        train = self.examples[:split_idx]
        val = self.examples[split_idx:]

        train_path = output_dir / "train_byt5.jsonl"
        eval_path = output_dir / "eval_byt5.jsonl"
        stats_path = output_dir / "training_data_stats.json"

        for path, data in [(train_path, train), (eval_path, val)]:
            with open(path, 'w', encoding='utf-8') as f:
                for ex in data:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            logger.info(f"Wrote {len(data)} examples to {path}")

        # Compute detailed stats
        cat_counts = Counter(e["category"] for e in self.examples)
        identity_count = sum(1 for e in self.examples if "identity" in e["category"])
        correction_count = len(self.examples) - identity_count

        stats = {
            "total": len(self.examples),
            "train": len(train),
            "eval": len(val),
            "identity_total": identity_count,
            "identity_pct": round(100 * identity_count / len(self.examples), 1),
            "correction_total": correction_count,
            "correction_pct": round(100 * correction_count / len(self.examples), 1),
            "categories": dict(cat_counts.most_common()),
        }

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("FINAL TRAINING DATA STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total examples:       {stats['total']:>8}")
        logger.info(f"  Train:              {stats['train']:>8}")
        logger.info(f"  Eval:               {stats['eval']:>8}")
        logger.info(f"  Identity:           {stats['identity_total']:>8} ({stats['identity_pct']}%)")
        logger.info(f"  Correction:         {stats['correction_total']:>8} ({stats['correction_pct']}%)")
        logger.info(f"\nCategory breakdown:")
        for cat, cnt in cat_counts.most_common():
            pct = 100 * cnt / len(self.examples)
            logger.info(f"  {cat:30s}: {cnt:>7} ({pct:5.1f}%)")
        logger.info("=" * 70)

        # Show samples
        logger.info("\nSample training examples:")
        for ex in random.sample(self.examples, min(15, len(self.examples))):
            logger.info(f"  [{ex['category']:25s}]  {ex['input_text']:45s}  →  {ex['target_text']}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Build e-commerce training data for ByT5-Large")
    parser.add_argument("--total", type=int, default=200_000,
                        help="Target total training examples (default: 200000)")
    parser.add_argument("--esci-sample", type=int, default=None,
                        help="Limit ESCI rows processed (for speed; default: all)")
    parser.add_argument("--real-pairs", type=str, default="real_data/combined_real_pairs.jsonl",
                        help="Path to real pairs from download_real_datasets.py")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory (default: data)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Step 1: Process ESCI ──
    esci = ESCIProcessor(sample_limit=args.esci_sample)
    esci.load_and_process()

    # ── Step 2: Build dataset ──
    builder = TrainingDataBuilder(total_target=args.total)

    # Add real general English corrections (~38K)
    builder.add_real_general_corrections(Path(args.real_pairs))

    # Add ESCI identity pairs (target: ~40% of total)
    identity_target = int(args.total * 0.40)
    builder.add_esci_identities(esci, count=identity_target)

    # Add natural corrections mined from ESCI
    builder.add_natural_corrections(esci)

    # Add e-commerce brand/product/query typos (~30% of total)
    ecom_target = int(args.total * 0.30)
    builder.add_ecommerce_typos(esci, count=ecom_target)

    # Add hard identity phrases
    builder.add_hard_identity_phrases()

    # Balance and save
    builder.balance_and_finalize()
    builder.save(Path(args.output_dir))

    logger.info("\nBitti! Şimdi bu veriyle Colab'da ByT5-Large'ı eğitebilirsin.")


if __name__ == "__main__":
    main()
