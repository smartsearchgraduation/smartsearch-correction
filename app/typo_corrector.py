"""
TypoCorrector - E-commerce Search Query Spell Correction Module

Available Models:
- SYMSPELL: Ultra-fast dictionary-based (recommended primary)
- KEYBOARD: Keyboard proximity-aware correction  
- BYT5: ByT5 fine-tuned model for typo correction

API Usage:
    from app.typo_corrector import TypoCorrector, CorrectionModel
    
    corrector = TypoCorrector()
    
    # Simple API - just get corrected string
    corrected = corrector.correct_query("iphnoe 15 pro")
    # Returns: "iphone 15 pro"
    
    # Full API - with metadata
    result = corrector.correct("iphnoe 15 pro", model=CorrectionModel.SYMSPELL)
    # Returns: {"original_query": "...", "normalized_query": "...", "changed": True, ...}
"""

import re
import os
import time
import logging
import numpy as np
from symspellpy import SymSpell, Verbosity
import torch

from .domain_vocab import load_domain_vocab

# Configure logging
logger = logging.getLogger(__name__)


class CorrectionModel:
    """Available correction models."""
    SYMSPELL = "symspell"      # Fast, dictionary-based
    KEYBOARD = "keyboard"      # Keyboard proximity-aware
    BYT5 = "byt5"              # ByT5-small fine-tuned (~180ms, 79% acc)


# QWERTY Keyboard Layout - Her tuşun komşuları
QWERTY_NEIGHBORS = {
    # Row 1
    'q': ['w', 'a', '1', '2'],
    'w': ['q', 'e', 'a', 's', '2', '3'],
    'e': ['w', 'r', 's', 'd', '3', '4'],
    'r': ['e', 't', 'd', 'f', '4', '5'],
    't': ['r', 'y', 'f', 'g', '5', '6'],
    'y': ['t', 'u', 'g', 'h', '6', '7'],
    'u': ['y', 'i', 'h', 'j', '7', '8'],
    'i': ['u', 'o', 'j', 'k', '8', '9'],
    'o': ['i', 'p', 'k', 'l', '9', '0'],
    'p': ['o', 'l', '0', '-'],
    # Row 2
    'a': ['q', 'w', 's', 'z'],
    's': ['a', 'w', 'e', 'd', 'z', 'x'],
    'd': ['s', 'e', 'r', 'f', 'x', 'c'],
    'f': ['d', 'r', 't', 'g', 'c', 'v'],
    'g': ['f', 't', 'y', 'h', 'v', 'b'],
    'h': ['g', 'y', 'u', 'j', 'b', 'n'],
    'j': ['h', 'u', 'i', 'k', 'n', 'm'],
    'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    # Row 3
    'z': ['a', 's', 'x'],
    'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'],
    'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'],
    'm': ['n', 'j', 'k'],
}


def keyboard_char_distance(char1: str, char2: str) -> float:
    """İki karakter arasındaki klavye mesafesini hesapla."""
    c1, c2 = char1.lower(), char2.lower()
    
    if c1 == c2:
        return 0.0
    
    # Doğrudan komşu mu?
    if c1 in QWERTY_NEIGHBORS and c2 in QWERTY_NEIGHBORS.get(c1, []):
        return 0.3  # Çok yakın - komşu tuş (düşük maliyet)
    
    # 2. derece komşu mu? (komşunun komşusu)
    if c1 in QWERTY_NEIGHBORS:
        for neighbor in QWERTY_NEIGHBORS[c1]:
            if neighbor in QWERTY_NEIGHBORS and c2 in QWERTY_NEIGHBORS.get(neighbor, []):
                return 0.7  # Orta mesafe
    
    return 1.5  # Uzak tuşlar (yüksek maliyet)


def keyboard_weighted_edit_distance(typo: str, correct: str) -> float:
    """Keyboard-aware weighted edit distance hesapla."""
    typo, correct = typo.lower(), correct.lower()
    
    if typo == correct:
        return 0.0
    
    m, n = len(typo), len(correct)
    
    # DP tablosu
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i * 1.0  # Silme maliyeti
    for j in range(n + 1):
        dp[0][j] = j * 1.0  # Ekleme maliyeti
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if typo[i-1] == correct[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                # Klavye mesafesine göre substitution maliyeti
                sub_cost = dp[i-1][j-1] + keyboard_char_distance(typo[i-1], correct[j-1])
                del_cost = dp[i-1][j] + 1.0
                ins_cost = dp[i][j-1] + 1.0
                dp[i][j] = min(sub_cost, del_cost, ins_cost)
    
    return dp[m][n]


def load_typo_mappings(filepath: str) -> dict:
    """Load explicit typo->correct mappings from file."""
    mappings = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('='):
                    continue
                parts = line.split(',')
                if len(parts) == 2:
                    typo, correct = parts[0].strip().lower(), parts[1].strip().lower()
                    if typo and correct:
                        mappings[typo] = correct
    except FileNotFoundError:
        logger.warning(f"Typo mappings file not found: {filepath}")
    return mappings
    return mappings


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
    
    def __init__(self, vocab_path: str = None, model: str = CorrectionModel.SYMSPELL):
        """
        Initialize the corrector with vocabularies.
        
        Args:
            vocab_path: Path to domain-specific vocabulary file (optional, uses default if None)
            model: Correction model to use ('symspell', 'mistral', 'hybrid')
        """
        self.correction_model = model
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        
        if vocab_path is None:
            vocab_path = os.path.join(data_dir, 'domain_vocab.txt')
        
        # Load domain words (e.g., product names, brands)
        self.domain_vocab = load_domain_vocab(vocab_path)
        
        # Load electronics-specific vocabulary
        electronics_vocab_path = os.path.join(data_dir, 'electronics_vocab.txt')
        self.electronics_vocab = load_domain_vocab(electronics_vocab_path)
        
        # Load brand-specific product lines (highest priority vocabulary)
        brand_products_path = os.path.join(data_dir, 'brand_products.txt')
        self.brand_products_vocab = load_domain_vocab(brand_products_path)
        
        # Load explicit typo mappings (highest priority)
        typo_mappings_path = os.path.join(data_dir, 'typo_mappings.txt')
        self.typo_mappings = load_typo_mappings(typo_mappings_path)
        
        # Protected terms that should NEVER be changed
        self.protected_patterns = [
            r'^\d+gb$',      # 16gb, 32gb, etc.
            r'^\d+tb$',      # 1tb, 2tb, etc.
            r'^\d+mb$',      # 512mb, etc.
            r'^\d+hz$',      # 60hz, 144hz, etc.
            r'^\d+ghz$',     # 3.5ghz, etc.
            r'^\d+mhz$',     # 3200mhz, etc.
            r'^\d+w$',       # 650w, etc.
            r'^\d+mp$',      # 12mp, 48mp, etc.
            r'^\d+fps$',     # 60fps, 120fps, etc.
            r'^\d+k$',       # 4k, 8k, etc.
            r'^i\d+$',       # i5, i7, i9
            r'^m\d+$',       # m1, m2, m3
            r'^[a-z]\d+$',   # g15, s23, a54, etc. (product model numbers)
            r'^rtx\d+$',     # rtx3060, rtx4090
            r'^gtx\d+$',     # gtx1080, etc.
            r'^rx\d+$',      # rx6800, rx7900
            r'^ryzen\d+$',   # ryzen5, ryzen7, ryzen9
            r'^ps\d+$',      # ps4, ps5
            r'^a\d+$',       # a14, a15 (Apple chips)
            r'^[a-z]+\d+[a-z]*$',  # General alphanumeric model numbers
        ]
        
        # Simple generic vocab for demo (expanded with curated common words)
        # Use curated smaller common words list to avoid noisy corrections
        common_words_path = os.path.join(data_dir, 'curated_common_words.txt')
        self.generic_vocab = load_domain_vocab(common_words_path) | {
            # Basic English words
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
            
            # Everyday English & Slang
            "wanna", "gonna", "gotta", "kinda", "sorta", "dunno", "hey", "hi", "thanks", "pls", "plz",
            "cuz", "bc", "tho", "lol", "brb", "afk", "btw", "idk", "omg", "wtf", "haha", "hehe", "lmao",
            "rofl", "yolo", "fomo", "tl;dr", "smh", "tbh", "imo", "ama", "tldr", "bff", "bae", "fam",
            "lit", "fire", "dope", "sick", "cool", "awesome", "amazing", "fantastic", "terrible", "horrible",
            "awesome", "great", "super", "ultra", "mega", "mini", "micro", "nano", "giga", "tera",
            
            # E-commerce & Shopping Terms
            "cheap", "expensive", "budget", "premium", "gaming", "portable", "wireless", "bluetooth",
            "usb", "hdmi", "wifi", "ethernet", "ssd", "hdd", "ram", "cpu", "gpu", "motherboard",
            "laptop", "notebook", "desktop", "tablet", "phone", "smartphone", "headphones", "earbuds",
            "keyboard", "mouse", "monitor", "display", "screen", "touchscreen", "battery", "charger",
            "adapter", "cable", "case", "bag", "backpack", "stand", "dock", "hub", "router", "modem",
            "printer", "scanner", "webcam", "microphone", "speaker", "soundbar", "projector", "tv",
            "smart", "android", "ios", "windows", "mac", "linux", "chromebook", "gaming", "productivity",
            "business", "student", "professional", "consumer", "enterprise", "budget", "mid-range", "high-end",
            "flagship", "entry-level", "value", "performance", "reliability", "durability", "portability",
            "ergonomics", "design", "build", "quality", "warranty", "support", "service", "shipping",
            "delivery", "returns", "refund", "exchange", "discount", "coupon", "deal", "sale", "offer",
            "promotion", "clearance", "outlet", "warehouse", "store", "online", "retail", "wholesale",
            
            # Technical Specifications
            "core", "cores", "thread", "threads", "ghz", "mhz", "khz", "fps", "hz", "khz", "mbps", "gbps",
            "mb", "gb", "tb", "pb", "kb", "byte", "bytes", "bit", "bits", "pixel", "pixels", "resolution",
            "hd", "fhd", "qhd", "uhd", "4k", "8k", "hdr", "oled", "led", "lcd", "ips", "va", "tn",
            "refresh", "response", "latency", "ping", "upload", "download", "bandwidth", "storage",
            "memory", "capacity", "speed", "performance", "efficiency", "power", "consumption", "battery",
            "life", "runtime", "charging", "fast", "quick", "wireless", "wired", "bluetooth", "usb-c",
            "thunderbolt", "lightning", "micro", "mini", "standard", "full", "half", "quarter", "eighth",
            
            # Colors & Materials
            "black", "white", "gray", "grey", "silver", "gold", "rose", "blue", "red", "green", "yellow",
            "orange", "purple", "pink", "brown", "beige", "tan", "cream", "ivory", "charcoal", "navy",
            "maroon", "burgundy", "teal", "cyan", "magenta", "lime", "olive", "coral", "salmon", "peach",
            "mint", "lavender", "plum", "copper", "brass", "bronze", "chrome", "matte", "glossy", "shiny",
            "dull", "metallic", "plastic", "metal", "aluminum", "steel", "carbon", "fiber", "glass", "ceramic",
            "leather", "fabric", "cloth", "rubber", "silicone", "wood", "bamboo", "recycled", "eco-friendly",
            
            # Brands & Products (additional)
            "apple", "google", "microsoft", "amazon", "dell", "hp", "acer", "asus", "lg", "sony", "panasonic",
            "toshiba", "sharp", "vizio", "samsung", "tcl", "philips", "jbl", "bose", "sony", "beats", "airpods",
            "galaxy", "pixel", "surface", "thinkpad", "xps", "spectre", "envy", "pavilion", "omen", "rog",
            "tuf", "zenbook", "vivobook", "swift", "nitro", "helios", "legion", "ideapad", "yoga", "chromebook",
            "macbook", "imac", "mac", "ipad", "iphone", "ipod", "apple", "watch", "airtag", "homepod", "siri",
            "alexa", "echo", "fire", "kindle", "ring", "blink", "arlo", "nest", "hue", "wyze", "logitech",
            "razer", "corsair", "steelseries", "hyperx", "astro", "turtle", "beach", "plantronics", "jabra",
            "sennheiser", "audio-technica", "shure", "akg", "beyerdynamic", "grifols", "marshall", "fender",
            "yamaha", "roland", "korg", "novation", "native", "instruments", "ableton", "fl", "studio",
            "cubase", "logic", "pro", "tools", "reason", "bitwig", "reaper", "ardour", "audacity", "lmms",
            
            # Numbers as words (for better recognition)
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
            "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
            
            # Common typos that should be preserved or corrected minimally
            "wants", "needs", "looking", "searching", "find", "get", "buy", "purchase", "order", "shop",
            "store", "market", "mall", "supermarket", "grocery", "pharmacy", "electronics", "computer", "tech",
            "gadget", "device", "accessory", "part", "component", "upgrade", "repair", "service", "maintenance"
        }
        
        # Add NLTK words for much larger vocabulary
        nltk_words_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'nltk_words.txt')
        nltk_vocab = load_domain_vocab(nltk_words_path)
        self.generic_vocab |= nltk_vocab
        
        # Combine all vocabularies
        self.all_vocab = self.domain_vocab | self.generic_vocab | self.electronics_vocab | self.brand_products_vocab
        
        # Initialize SymSpell with frequency-based dictionary
        self.sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        
        # Add brand products vocabulary with HIGHEST priority
        for word in self.brand_products_vocab:
            self.sym_spell.create_dictionary_entry(word, 200)  # Highest priority for brand products
        
        # Add electronics vocabulary with HIGH priority
        for word in self.electronics_vocab:
            self.sym_spell.create_dictionary_entry(word, 100)  # High priority
        
        # Add domain words with medium-high frequency (prioritize e-commerce terms)
        for word in self.domain_vocab:
            self.sym_spell.create_dictionary_entry(word, 50)  # Medium-high priority
        
        # Add generic words with lower frequency
        for word in self.generic_vocab - self.domain_vocab - self.electronics_vocab - self.brand_products_vocab:
            # Give higher frequency to curated words, lower to NLTK words
            if word in self.generic_vocab and word not in nltk_vocab:
                self.sym_spell.create_dictionary_entry(word, 10)  # Curated words
            else:
                self.sym_spell.create_dictionary_entry(word, 1)  # NLTK words
        
        # Fine-tuned model paths (lazy loaded)
        fine_tune_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'fine_tune', 'outputs')
        self.finetuned_model_paths = {
            "byt5": os.path.join(fine_tune_dir, 'byt5-typo', 'byt5-small-typo-20251208_2235-final'),
        }
        
        # Lazy-loaded fine-tuned models (will be loaded on first use)
        self._finetuned_models = {}
        self._finetuned_tokenizers = {}

    def _load_generic_vocab(self, vocab_path: str) -> set[str]:
        vocab = set()
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        vocab.add(word)
        except FileNotFoundError:
            logger.warning(f"Vocab file not found: {vocab_path}")
        return vocab

    def _is_numeric_or_unit(self, token: str) -> bool:
        """Check if token is numeric, unit-like, or a protected pattern."""
        token_lower = token.lower()
        
        # Pure numbers
        if re.match(r'^\d+(\.\d+)?$', token):
            return True
        
        # Common units and operators
        if token_lower in {"gb", "tb", "hz", "dollars", "dollar", "under", "between", "<", ">", "$", "€", 
                          "mb", "kb", "ghz", "mhz", "fps", "khz", "mbps", "gbps", "px", "inch", "inches", 
                          "cm", "mm", "kg", "lbs", "oz", "watt", "watts", "volt", "volts", "amp", "amps",
                          "mp", "megapixel", "megapixels"}:
            return True
        
        # Check protected patterns (e.g., 16gb, rtx4090, i7, m2)
        for pattern in self.protected_patterns:
            if re.match(pattern, token_lower):
                return True
        
        return False

    def _find_best_candidate(self, token: str):
        """
        Find the best correction candidate for a token.
        Priority: 1) Explicit typo mappings, 2) SymSpell suggestions
        
        Args:
            token: The word to correct
            
        Returns:
            dict: {'word': corrected_word, 'distance': edit_distance, 'confidence': confidence_score}
        """
        token_lower = token.lower()
        
        # 1. Check explicit typo mappings first (highest priority)
        if token_lower in self.typo_mappings:
            return {
                "word": self.typo_mappings[token_lower],
                "distance": 1,
                "confidence": 0.99  # Very high confidence for explicit mappings
            }
        
        # 2. Check if already in vocabulary (no correction needed)
        if token_lower in self.all_vocab:
            return {"word": token_lower, "distance": 0, "confidence": 1.0}
        
        # 3. Get suggestions from SymSpell (fuzzy matching)
        suggestions = self.sym_spell.lookup(token_lower, Verbosity.CLOSEST, max_edit_distance=3)
        
        if suggestions:
            best = suggestions[0]
            distance = best.distance
            
            # Enhanced confidence calculation for 90%+ accuracy
            if distance == 0:
                confidence = 1.0  # Exact match
            elif distance == 1:
                # Very high confidence for single character changes
                confidence = 0.95 if len(token_lower) > 3 else 0.85
            elif distance == 2:
                # High confidence for double character changes on longer words
                confidence = 0.90 if len(token_lower) > 5 else 0.75
            elif distance == 3:
                # Medium confidence for triple character changes
                confidence = 0.75 if len(token_lower) > 6 else 0.50
            else:
                # Low confidence for larger changes
                confidence = 0.30
            
            # Boost confidence for electronics/domain terms
            if best.term in self.electronics_vocab:
                confidence = min(1.0, confidence + 0.10)
            elif best.term in self.domain_vocab:
                confidence = min(1.0, confidence + 0.05)
            
            # Penalty for very short words with changes
            if len(token_lower) <= 2 and distance > 0:
                confidence *= 0.5
            
            # Heavy penalty for changing numbers to words
            if re.match(r'^\d+', token_lower) and not re.match(r'^\d+', best.term):
                confidence *= 0.2
            
            # Penalty if suggestion length differs too much
            len_diff = abs(len(best.term) - len(token_lower))
            if len_diff > 3:
                confidence *= 0.7
            
            return {"word": best.term, "distance": distance, "confidence": confidence}
        
        # No good suggestion found, return original
        return {"word": token_lower, "distance": 0, "confidence": 1.0}

    def _correct_with_symspell(self, query: str) -> dict:
        """
        Correct typos using SymSpell (fast dictionary-based approach).
        """
        start_time = time.time()
        
        if not query.strip():
            total_latency = time.time() - start_time
            return {
                "original_query": query,
                "normalized_query": query,
                "changed": False,
                "tokens": [],
                "latency": {
                    "total_ms": round(total_latency * 1000, 2),
                    "tokens_per_second": 0.0
                }
            }
        
        tokens = re.split(r'\s+', query.strip())
        corrected_tokens = []
        token_details = []
        changed = False
        
        for token in tokens:
            token_lower = token.lower()
            
            # 1. FIRST check explicit typo mappings (highest priority, even for numbers)
            if token_lower in self.typo_mappings:
                corrected = self.typo_mappings[token_lower]
                corrected_tokens.append(corrected)
                if corrected != token_lower:
                    changed = True
                    token_details.append({
                        "original": token,
                        "corrected": corrected,
                        "confidence": 0.99,
                        "latency_ms": 0.1
                    })
                continue
            
            # 2. Skip protected patterns (numbers, units, specs)
            if self._is_numeric_or_unit(token):
                corrected_tokens.append(token_lower)
                continue
            
            candidate = self._find_best_candidate(token)
            
            # Apply corrections with 85%+ confidence (lowered from 90% for better coverage)
            if candidate['confidence'] >= 0.85 and candidate['word'] != token.lower():
                corrected_tokens.append(candidate['word'])
                changed = True
                token_details.append({
                    "original": token,
                    "corrected": candidate['word'],
                    "confidence": candidate['confidence'],
                    "latency_ms": 0.1  # SymSpell is very fast
                })
            else:
                # Keep original if confidence is too low
                corrected_tokens.append(token.lower())
        
        total_latency = time.time() - start_time
        token_count = len(tokens)
        
        normalized_query = " ".join(corrected_tokens)
        
        return {
            "original_query": query,
            "normalized_query": normalized_query,
            "changed": changed,
            "tokens": token_details,
            "latency": {
                "total_ms": round(total_latency * 1000, 2),
                "tokens_per_second": round(token_count / total_latency, 2) if total_latency > 0 else 0.0
            }
        }

    def correct_query(self, query: str, model: str = None) -> str:
        """
        Simple API: Correct typos and return only the corrected string.
        
        Args:
            query: The user's search query string
            model: Correction model ('symspell', 'keyboard', 'byt5')
                   Default: 'symspell' (fastest)
            
        Returns:
            str: Corrected query string
            
        Example:
            corrector = TypoCorrector()
            result = corrector.correct_query("iphnoe 15 pro")
            # Returns: "iphone 15 pro"
        """
        if not query or not query.strip():
            return query
        
        result = self.correct(query, model=model or CorrectionModel.SYMSPELL)
        return result.get('normalized_query', query)

    def correct(self, query: str, model: str = None) -> dict:
        """
        Full API: Correct typos with detailed metadata.
        
        Args:
            query: The user's search query string
            model: Correction model to use:
                   - 'symspell': Fast dictionary-based (default, ~0.1ms)
                   - 'keyboard': Keyboard proximity-aware (~5ms)
                   - 'byt5': ByT5 fine-tuned (~180ms, 79% acc)
            
        Returns:
            dict: {
                "original_query": str,
                "normalized_query": str,
                "changed": bool,
                "latency_ms": float
            }
        """
        # Use specified model or default
        active_model = model or self.correction_model
        
        if active_model == CorrectionModel.SYMSPELL:
            return self._correct_with_symspell(query)
        elif active_model == CorrectionModel.KEYBOARD:
            return self._correct_with_keyboard(query)
        elif active_model == CorrectionModel.BYT5:
            return self._correct_with_byt5(query)
        else:
            available_models = [
                CorrectionModel.SYMSPELL, CorrectionModel.KEYBOARD, CorrectionModel.BYT5
            ]
            raise ValueError(f"Unknown model: {active_model}. Available: {available_models}")

    def correct_batch(self, queries: list[str]) -> dict:
        """Correct a batch of queries and return aggregated results with latency stats."""
        start_time = time.time()
        
        results = []
        total_tokens = 0
        
        for query in queries:
            result = self.correct(query)
            results.append(result)
            total_tokens += len(result['original_query'].split())
        
        total_latency = time.time() - start_time
        avg_latency_per_query = total_latency / len(queries) if queries else 0
        
        return {
            "results": results,
            "batch_stats": {
                "total_queries": len(queries),
                "total_tokens": total_tokens,
                "total_latency_ms": round(total_latency * 1000, 2),
                "avg_latency_per_query_ms": round(avg_latency_per_query * 1000, 2),
                "queries_per_second": round(len(queries) / total_latency, 2) if total_latency > 0 else 0.0,
                "tokens_per_second": round(total_tokens / total_latency, 2) if total_latency > 0 else 0.0
            }
        }

    def _correct_with_keyboard(self, query: str) -> dict:
        """
        Keyboard proximity-aware spell correction.
        Uses weighted edit distance based on QWERTY keyboard layout.
        Good for correcting typos from adjacent key presses (e.g., mbifia -> nvidia).
        """
        import time
        
        start_time = time.time()
        
        words = query.lower().strip().split()
        token_details = []
        corrected_words = []
        changed = False
        
        # Build vocabulary for keyboard matching
        vocab_set = self.brand_products_vocab | self.electronics_vocab | self.domain_vocab
        vocab_list = list(vocab_set)
        
        for word in words:
            token_start = time.time()
            original_word = word
            
            # Check typo mappings first (highest priority)
            if word in self.typo_mappings:
                corrected = self.typo_mappings[word]
                confidence = 1.0
                source = "typo_mapping"
            # Check if protected pattern
            elif self._is_numeric_or_unit(word):
                corrected = word
                confidence = 1.0
                source = "protected"
            # Check if exact match in vocab
            elif word in vocab_set:
                corrected = word
                confidence = 1.0
                source = "vocab_exact"
            else:
                # Use keyboard-weighted edit distance to find best match
                best_match = word
                best_distance = float('inf')
                
                # Only check if word length is reasonable (3-15 chars)
                if 3 <= len(word) <= 15:
                    for vocab_word in vocab_list:
                        # Skip if length difference is too big
                        if abs(len(word) - len(vocab_word)) > 3:
                            continue
                        
                        dist = keyboard_weighted_edit_distance(word, vocab_word)
                        
                        # Normalize by word length
                        normalized_dist = dist / max(len(word), len(vocab_word))
                        
                        if normalized_dist < best_distance and normalized_dist < 0.5:
                            best_distance = normalized_dist
                            best_match = vocab_word
                
                if best_match != word:
                    corrected = best_match
                    confidence = 1.0 - best_distance
                    source = "keyboard_proximity"
                    changed = True
                else:
                    # Fallback to SymSpell for words not in our vocab
                    suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                    if suggestions and suggestions[0].distance <= 2:
                        corrected = suggestions[0].term
                        confidence = 1.0 - (suggestions[0].distance / len(word))
                        source = "symspell_fallback"
                        if corrected != word:
                            changed = True
                    else:
                        corrected = word
                        confidence = 1.0
                        source = "unchanged"
            
            corrected_words.append(corrected)
            token_end = time.time()
            
            if corrected != original_word:
                token_details.append({
                    "original": original_word,
                    "corrected": corrected,
                    "confidence": round(confidence, 4),
                    "source": source,
                    "latency_ms": round((token_end - token_start) * 1000, 2)
                })
        
        total_latency = time.time() - start_time
        corrected_query = " ".join(corrected_words)
        
        return {
            "original_query": query,
            "normalized_query": corrected_query,
            "changed": changed,
            "model": "keyboard",
            "tokens": token_details,
            "latency": {
                "total_ms": round(total_latency * 1000, 2),
                "tokens_per_second": round(len(words) / total_latency, 2) if total_latency > 0 else 0.0
            }
        }

    # ==================== FINE-TUNED MODEL METHODS ====================
    
    def _load_finetuned_model(self, model_type: str):
        """
        Lazy load a fine-tuned model.
        
        Args:
            model_type: 'byt5'
        """
        if model_type in self._finetuned_models:
            return  # Already loaded
        
        model_path = self.finetuned_model_paths.get(model_type)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Fine-tuned model not found: {model_path}")
        
        logger.info(f"Loading fine-tuned model: {model_type} from {model_path}")
        
        if model_type == "byt5":
            # ByT5 - full model, no adapter
            from transformers import T5ForConditionalGeneration, AutoTokenizer
            self._finetuned_tokenizers[model_type] = AutoTokenizer.from_pretrained(model_path)
            self._finetuned_models[model_type] = T5ForConditionalGeneration.from_pretrained(model_path)
            # Move to GPU if available
            if torch.cuda.is_available():
                self._finetuned_models[model_type] = self._finetuned_models[model_type].cuda()
            elif torch.backends.mps.is_available():
                self._finetuned_models[model_type] = self._finetuned_models[model_type].to("mps")
        
        logger.info(f"Fine-tuned model loaded: {model_type}")

    def _correct_with_byt5(self, query: str) -> dict:
        """Correct using ByT5 fine-tuned model."""
        start_time = time.time()
        
        if not query or not query.strip():
            return {
                "original_query": query,
                "normalized_query": query,
                "changed": False,
                "model": "byt5",
                "latency_ms": 0.0
            }
        
        # Load model if not already loaded
        try:
            self._load_finetuned_model("byt5")
        except Exception as e:
            logger.error(f"Failed to load byt5 model: {e}")
            # Fallback to SymSpell
            return self._correct_with_symspell(query)
        
        model = self._finetuned_models["byt5"]
        tokenizer = self._finetuned_tokenizers["byt5"]
        
        try:
            # ByT5 format: "correct: <query>"
            input_text = f"correct: {query}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
            
            # Move to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=2,
                    early_stopping=True
                )
            
            corrected = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        except Exception as e:
            logger.error(f"Error during byt5 inference: {e}")
            corrected = query
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "original_query": query,
            "normalized_query": corrected,
            "changed": corrected.lower() != query.lower(),
            "model": "byt5",
            "latency_ms": round(latency_ms, 2)
        }