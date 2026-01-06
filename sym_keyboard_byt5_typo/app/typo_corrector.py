"""
TypoCorrector - E-commerce Search Query Spell Correction Module

Available Models:
- SYMSPELL: Ultra-fast dictionary-based (recommended primary)
- KEYBOARD: Keyboard proximity-aware correction  
- E5_ONNX: Semantic similarity with E5 embeddings
- MISTRAL: LLM-based correction (slow, use as backup)

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
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from llama_cpp import Llama

from .domain_vocab import load_domain_vocab

# Configure logging
logger = logging.getLogger(__name__)


class CorrectionModel:
    """Available correction models."""
    SYMSPELL = "symspell"      # Fast, dictionary-based
    MISTRAL = "mistral"        # LLM-based, slow but accurate
    HYBRID = "hybrid"          # SymSpell + Mistral fallback
    E5_ONNX = "e5_onnx"        # Semantic similarity
    KEYBOARD = "keyboard"      # Keyboard proximity-aware
    SYMSPELL_KEYBOARD = "symspell_keyboard"  # SymSpell + Keyboard hybrid (best fast option)
    
    # Fine-tuned models for backup when SymSpell fails
    BYT5 = "byt5"              # ByT5-small fine-tuned (~180ms, 79% acc)
    QWEN = "qwen"              # Qwen2.5-0.5B fine-tuned (~300ms)
    QWEN_1_5B = "qwen-1.5b"    # Qwen2.5-1.5B fine-tuned (~2s, 81% acc)
    LLAMA = "llama"            # Llama-3.2-1B fine-tuned (~260ms, 84% acc)
    MINISTRAL = "ministral"    # Ministral-3B fine-tuned (~400ms, expected ~86%+)
    PHI3 = "phi3"              # Phi-3-Mini fine-tuned (~350ms, expected ~85%+)
    SMART_HYBRID = "smart_hybrid"  # SymSpell first, fine-tuned fallback


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


def normalize_repeated_chars(token: str) -> str:
    """
    Normalize excessively repeated characters in a token.
    
    Examples:
        assssuusss -> asus
        iphoneeeee -> iphone  
        coooool -> cool  
        samsungggg -> samsung
        
    This handles cases where users type with emphasis like "I loooove this".
    
    Strategy:
        - Skip tokens that are pure numbers or alphanumeric product codes
        - First pass: Reduce 3+ consecutive chars to 1
        - This handles most typos and exaggerated typing
        - Words like "cool", "dell", "google" with natural doubles are unaffected
          because they only have 2 consecutive chars
        
    Args:
        token: Input token that may have repeated characters
        
    Returns:
        Token with repeated characters normalized
    """
    if not token or len(token) < 3:
        return token
    
    # Skip tokens that are pure numbers (e.g., 1000, 25000, 2000)
    # These are valid product quantities/sizes and should not be altered
    if token.isdigit():
        return token
    
    # Skip alphanumeric codes that look like model numbers (e.g., rm1000x, rtx4090)
    # They often have valid repeated digits
    if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
        # Only normalize if there are 4+ repeated LETTERS (not digits)
        # This handles "iphoneeeee" but preserves "rm1000x"
        has_excessive_letter_repeat = False
        for i in range(len(token) - 3):
            if token[i].isalpha() and token[i:i+4] == token[i] * 4:
                has_excessive_letter_repeat = True
                break
        if not has_excessive_letter_repeat:
            return token
    
    result = []
    i = 0
    
    while i < len(token):
        char = token[i]
        # Count consecutive occurrences
        count = 1
        while i + count < len(token) and token[i + count] == char:
            count += 1
        
        # Add the character(s)
        # Only reduce repeated LETTERS, not digits
        if count >= 3 and char.isalpha():
            # 3+ letter repetitions: definitely a typo/emphasis, reduce to 1
            result.append(char)
        else:
            # Keep digits and 1-2 letter repetitions as-is
            for _ in range(count):
                result.append(char)
        
        i += count
    
    return ''.join(result)


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
    
    def __init__(self, vocab_path: str = None, model: str = CorrectionModel.SYMSPELL,
                 use_offline_symspell_dict: bool = False, offline_dict_path: str | None = None):
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
            # Price/Currency patterns - NEVER change these
            r'^\$\d+',       # $100, $1500, $99.99
            r'^€\d+',        # €100, €50.99
            r'^£\d+',        # £100, £50.99
            r'^₺\d+',        # ₺1000, ₺500
            r'^\d+\$',       # 100$, 1500$
            r'^\d+€',        # 100€, 50€
            r'^\d+£',        # 100£
            r'^\d+₺',        # 1000₺, 500₺
            r'^\d+tl$',      # 1000tl, 500tl (Turkish Lira text)
            r'^\d+usd$',     # 100usd
            r'^\d+eur$',     # 100eur
            r'^\d+gbp$',     # 100gbp
        ]
        
        # Price regex patterns for preprocessing - preserve these in full query
        self.price_patterns = [
            r'\$\d+(?:\.\d{2})?',      # $100, $99.99
            r'€\d+(?:\.\d{2})?',       # €100, €50.99
            r'£\d+(?:\.\d{2})?',       # £100, £50.99
            r'₺\d+(?:\.\d{2})?',       # ₺1000, ₺500.50
            r'\d+(?:\.\d{2})?\$',      # 100$, 99.99$
            r'\d+(?:\.\d{2})?€',       # 100€
            r'\d+(?:\.\d{2})?£',       # 100£
            r'\d+(?:\.\d{2})?₺',       # 1000₺
            r'\d+\s*tl\b',             # 1000 tl, 500tl
            r'\d+\s*usd\b',            # 100 usd, 100usd
            r'\d+\s*eur\b',            # 100 eur
            r'\d+\s*dolar\b',          # 100 dolar
            r'\d+\s*euro\b',           # 100 euro
            r'\d+\s*lira\b',           # 1000 lira
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

        # Optional: load pre-built offline dictionary for SymSpell
        self._symspell_source = "inline_vocab"
        if use_offline_symspell_dict:
            if offline_dict_path is None:
                offline_dict_path = os.path.join(data_dir, 'symspell_dictionary.txt')
            if os.path.exists(offline_dict_path):
                try:
                    self.sym_spell.load_dictionary(offline_dict_path, term_index=0, count_index=1, separator='\t')
                    self._symspell_source = "offline_dict"
                    logger.info(f"Loaded SymSpell dictionary from offline file: {offline_dict_path}")
                except Exception as e:
                    logger.warning(f"Failed to load offline SymSpell dictionary from {offline_dict_path}: {e}. Falling back to inline vocab.")
                    self._build_inline_symspell_dictionary(nltk_vocab)
            else:
                logger.warning(f"Offline SymSpell dictionary not found at {offline_dict_path}. Falling back to inline vocab.")
                self._build_inline_symspell_dictionary(nltk_vocab)
        else:
            # Default behaviour: build dictionary from in-memory vocabs (current demo behaviour)
            self._build_inline_symspell_dictionary(nltk_vocab)
        
        # Mistral model path (lazy loaded)
        self.mistral_model_path = '/Users/kaantufan/.cache/huggingface/hub/models--mradermacher--spellcheck-mistral-7b-GGUF/snapshots/7817496017c3595d43f476522037cad8914932aa/spellcheck-mistral-7b.Q4_K_M.gguf'
        
        # Fine-tuned model paths (lazy loaded)
        # Primary: Colab-trained model (T4 GPU, 10 epochs, 117K samples)
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        # Fallback: Old models from fine_tune/outputs
        old_fine_tune_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'fine_tune', 'outputs')
        
        self.finetuned_model_paths = {
            # ByT5: Full model - Colab trained (10 epochs, ~95% accuracy expected)
            "byt5": os.path.join(models_dir, 'byt5-typo-final'),
            # Qwen 0.5B: LoRA adapter (base: Qwen/Qwen2.5-0.5B-Instruct)
            "qwen": os.path.join(old_fine_tune_dir, 'qwen-typo', 'qwen-0.5b-typo-20251209_0009-final'),
            # Qwen 1.5B: LoRA adapter (base: Qwen/Qwen2.5-1.5B-Instruct)
            "qwen-1.5b": os.path.join(old_fine_tune_dir, 'qwen-typo', 'qwen-1.5b-typo-20251209_1146-final'),
            # Llama 1B: LoRA adapter (base: meta-llama/Llama-3.2-1B-Instruct)
            "llama": os.path.join(old_fine_tune_dir, 'llama-typo', 'llama-1b-typo-20251209_1454-final'),
            # Not trained yet
            "ministral": None,
            "phi3": None,
        }
        
        # Validate paths exist, fallback to old paths if needed
        for model_key in ["qwen", "qwen-1.5b", "llama"]:
            if not os.path.exists(self.finetuned_model_paths.get(model_key, "")):
                logger.warning(f"Model path not found for {model_key}")
        
        # Check ByT5 exists
        if not os.path.exists(self.finetuned_model_paths["byt5"]):
            logger.warning(f"ByT5 model not found at {self.finetuned_model_paths['byt5']}")
        else:
            logger.info(f"ByT5 model found at {self.finetuned_model_paths['byt5']}")
        
        # Lazy-loaded fine-tuned models (will be loaded on first use)
        self._finetuned_models = {}
        self._finetuned_tokenizers = {}

    def _build_inline_symspell_dictionary(self, nltk_vocab: set[str]) -> None:
        """Build SymSpell dictionary from in-memory vocab sets.

        This preserves the original demo behaviour and is used when
        no offline dictionary is provided or loading fails.
        """
        # Add brand products vocabulary with HIGHEST priority
        for word in self.brand_products_vocab:
            self.sym_spell.create_dictionary_entry(word, 200)

        # Add electronics vocabulary with HIGH priority
        for word in self.electronics_vocab:
            self.sym_spell.create_dictionary_entry(word, 100)

        # Add domain words with medium-high frequency (prioritize e-commerce terms)
        for word in self.domain_vocab:
            self.sym_spell.create_dictionary_entry(word, 50)

        # Add generic words with lower frequency
        for word in self.generic_vocab - self.domain_vocab - self.electronics_vocab - self.brand_products_vocab:
            # Give higher frequency to curated words, lower to NLTK words
            if word in self.generic_vocab and word not in nltk_vocab:
                self.sym_spell.create_dictionary_entry(word, 10)
            else:
                self.sym_spell.create_dictionary_entry(word, 1)

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
        
        # Pure numbers (including single digits like "4", "5", "15")
        if re.match(r'^\d+(\.\d+)?$', token):
            return True
        
        # Single character - don't correct (too risky)
        if len(token) == 1:
            return True
        
        # Two character tokens that start with digit - protect them
        if len(token) == 2 and token[0].isdigit():
            return True
        
        # Common units and operators
        if token_lower in {"gb", "tb", "hz", "dollars", "dollar", "under", "between", "<", ">", "$", "€", "£", "₺",
                          "mb", "kb", "ghz", "mhz", "fps", "khz", "mbps", "gbps", "px", "inch", "inches", 
                          "cm", "mm", "kg", "lbs", "oz", "watt", "watts", "volt", "volts", "amp", "amps",
                          "mp", "megapixel", "megapixels", "ii", "iii", "iv", "v", "vi",
                          "2k", "4k", "8k", "5g", "4g", "3g", "lte", "wifi", "usb", "rgb", "lcd", "led",
                          "hdr", "uhd", "fhd", "qhd", "ssd", "hdd", "nvme", "ddr", "ddr4", "ddr5",
                          "tl", "usd", "eur", "gbp", "euro", "dolar", "lira"}:
            return True
        
        # Check protected patterns (e.g., 16gb, rtx4090, i7, m2)
        for pattern in self.protected_patterns:
            if re.match(pattern, token_lower):
                return True
        
        return False

    def _extract_and_preserve_prices(self, query: str) -> tuple[str, list[tuple[str, str, int]]]:
        """
        Extract price patterns from query and replace with simple placeholders.
        This prevents price symbols from being modified during correction.
        
        Args:
            query: Original search query
            
        Returns:
            tuple: (modified_query, list of (placeholder, original_price, position) tuples)
        """
        preserved = []
        modified_query = query
        
        for i, pattern in enumerate(self.price_patterns):
            for match in re.finditer(pattern, modified_query, re.IGNORECASE):
                original_price = match.group()
                # Use a simple word-like placeholder that models won't change much
                placeholder = "PRICETOKEN"
                pos = len(preserved)
                preserved.append((placeholder, original_price, pos))
                modified_query = modified_query[:match.start()] + placeholder + modified_query[match.end():]
                break  # Only replace first match per pattern, then re-search
        
        return modified_query, preserved

    def _restore_prices(self, query: str, preserved: list[tuple[str, str, int]]) -> str:
        """
        Restore preserved price patterns back into the corrected query.
        Handles cases where model might have altered or removed placeholders.
        
        Args:
            query: Corrected query (may or may not have placeholders)
            preserved: List of (placeholder, original_price, position) tuples
            
        Returns:
            str: Query with original prices restored
        """
        if not preserved:
            return query
            
        result = query
        
        for placeholder, original, pos in preserved:
            # Try to replace placeholder with original price
            # Check for various case variations and common model corruptions
            variants = [
                placeholder, 
                placeholder.lower(), 
                placeholder.upper(),
                "pricetoken",
                "pricetooken",  # Common ByT5 corruption
                "pricetokeen",
                "pricetokon",
                "price token",
                "price_token",
            ]
            
            replaced = False
            for variant in variants:
                if variant in result.lower():
                    # Case-insensitive replace
                    pattern = re.compile(re.escape(variant), re.IGNORECASE)
                    result = pattern.sub(original, result, count=1)
                    replaced = True
                    break
            
            if not replaced:
                # Placeholder not found - model might have removed or changed it
                # Try to find any price-like pattern that model might have generated
                price_in_result = re.search(r'[\$€£₺]\s*\d+(?:[,\.]\d+)?|\d+(?:[,\.]\d+)?\s*[\$€£₺]', result)
                if price_in_result:
                    result = result[:price_in_result.start()] + original + result[price_in_result.end():]
        
        return result

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
                confidence *= 0.3  # More aggressive penalty for 2-char tokens
            elif len(token_lower) <= 3 and distance > 0:
                confidence *= 0.6  # Penalty for 3-char tokens
            
            # Heavy penalty for changing numbers to words (e.g., "4" -> "for")
            if re.match(r'^\d+', token_lower) and not re.match(r'^\d+', best.term):
                confidence *= 0.1  # Very heavy penalty
            
            # Penalty for changing short alphanumeric to pure word
            if len(token_lower) <= 3 and any(c.isdigit() for c in token_lower) and not any(c.isdigit() for c in best.term):
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
        Preserves price patterns like $100, 500₺, 100 euro etc.
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
        
        # Extract and preserve price patterns before processing
        processed_query, preserved_prices = self._extract_and_preserve_prices(query)
        
        tokens = re.split(r'\s+', processed_query.strip())
        corrected_tokens = []
        token_details = []
        changed = False
        
        for token in tokens:
            token_lower = token.lower()
            
            # 0. PRE-PROCESS: Normalize repeated characters (e.g., "assssuusss" -> "asus")
            normalized_token = normalize_repeated_chars(token_lower)
            if normalized_token != token_lower:
                # Token was normalized, mark as changed
                token_lower = normalized_token
                changed = True
            
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
            
            candidate = self._find_best_candidate(normalized_token)  # Use normalized token
            
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
        
        # Restore preserved price patterns
        normalized_query = self._restore_prices(normalized_query, preserved_prices)
        
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

    def _correct_with_mistral_full(self, query: str) -> dict:
        """
        Correct typos using Mistral 7B model with full latency tracking.
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
        
        # Get Mistral correction
        corrected_query = self._correct_with_mistral(query)
        total_latency = time.time() - start_time
        
        # Determine if changed
        changed = corrected_query.lower().strip() != query.lower().strip()
        
        # Token-level analysis with latency estimation
        original_tokens = query.split()
        corrected_tokens = corrected_query.split()
        token_details = []
        
        # Estimate token-level latency (divide total latency by number of tokens)
        token_count = len(original_tokens)
        token_latency = total_latency / max(token_count, 1)
        
        if changed and len(original_tokens) == len(corrected_tokens):
            # Detailed token comparison when lengths match
            for i, (orig, corr) in enumerate(zip(original_tokens, corrected_tokens)):
                if orig.lower() != corr.lower():
                    token_details.append({
                        "original": orig,
                        "corrected": corr,
                        "confidence": 0.9,  # High confidence for AI model
                        "latency_ms": round(token_latency * 1000, 2)
                    })
        elif changed:
            # Fallback for different token counts
            token_details.append({
                "original": query,
                "corrected": corrected_query,
                "confidence": 0.8,
                "latency_ms": round(total_latency * 1000, 2)
            })
        
        return {
            "original_query": query,
            "normalized_query": corrected_query,
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
            model: Correction model ('symspell', 'mistral', 'keyboard', 'e5_onnx')
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
                   - 'e5_onnx': Semantic similarity (~10ms)
                   - 'mistral': LLM-based (~1000ms, backup)
                   - 'byt5': ByT5 fine-tuned (~180ms, 79% acc)
                   - 'qwen': Qwen2.5-0.5B fine-tuned (~300ms)
                   - 'qwen-1.5b': Qwen2.5-1.5B fine-tuned (~2s, 81% acc)
                   - 'llama': Llama-3.2-1B fine-tuned (~260ms, 84% acc) - BEST
                   - 'smart_hybrid': SymSpell + Llama fallback (recommended)
            
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
        elif active_model == CorrectionModel.MISTRAL:
            return self._correct_with_mistral_full(query)
        elif active_model == CorrectionModel.HYBRID:
            return self._correct_with_hybrid(query)
        elif active_model == CorrectionModel.E5_ONNX:
            return self._correct_with_e5_onnx(query)
        elif active_model == CorrectionModel.KEYBOARD:
            return self._correct_with_keyboard(query)
        elif active_model == CorrectionModel.SYMSPELL_KEYBOARD:
            return self._correct_with_symspell_keyboard(query)
        # Fine-tuned models
        elif active_model == CorrectionModel.BYT5:
            return self._correct_with_byt5(query)
        elif active_model == CorrectionModel.QWEN:
            return self._correct_with_qwen(query)
        elif active_model == CorrectionModel.QWEN_1_5B:
            return self._correct_with_qwen_1_5b(query)
        elif active_model == CorrectionModel.LLAMA:
            return self._correct_with_llama(query)
        elif active_model == CorrectionModel.MINISTRAL:
            return self._correct_with_ministral(query)
        elif active_model == CorrectionModel.PHI3:
            return self._correct_with_phi3(query)
        elif active_model == CorrectionModel.SMART_HYBRID:
            return self._correct_with_smart_hybrid(query)
        else:
            available_models = [
                CorrectionModel.SYMSPELL, CorrectionModel.MISTRAL, CorrectionModel.HYBRID,
                CorrectionModel.E5_ONNX, CorrectionModel.KEYBOARD, CorrectionModel.SYMSPELL_KEYBOARD,
                CorrectionModel.BYT5, CorrectionModel.QWEN, CorrectionModel.QWEN_1_5B,
                CorrectionModel.LLAMA, CorrectionModel.MINISTRAL, CorrectionModel.PHI3, CorrectionModel.SMART_HYBRID
            ]
            raise ValueError(f"Unknown model: {active_model}. Available: {available_models}")

    def _correct_with_hybrid(self, query: str) -> dict:
        """
        Hybrid approach: Use SymSpell first, fallback to Mistral for low confidence.
        Best balance between speed and accuracy.
        """
        # First try SymSpell (fast)
        symspell_result = self._correct_with_symspell(query)
        
        # Check if any corrections were made with low confidence
        low_confidence_tokens = [t for t in symspell_result.get('tokens', []) if t.get('confidence', 1.0) < 0.90]
        
        # If SymSpell made corrections and all are high confidence, return it
        if not low_confidence_tokens:
            return symspell_result
        
        # Otherwise, use Mistral for the full query (slower but more accurate)
        return self._correct_with_mistral_full(query)

    # Hook for future transformer-based corrector (e.g., neuspell)
    def correct_with_transformer(self, query: str) -> dict:
        """
        Correct typos using hybrid approach: SymSpell + embedding fallback.
        Uses SymSpell for fast correction, embedding for semantic similarity when needed.
        """
        if not query.strip():
            return {
                "original_query": query,
                "normalized_query": query,
                "changed": False,
                "tokens": []
            }
        
        # Lazy load the model
        if not hasattr(self, 'embedding_model'):
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        tokens = query.split()
        corrected_tokens = []
        token_details = []
        changed = False
        
        for token in tokens:
            if self._is_numeric_or_unit(token):
                corrected_tokens.append(token)
                continue
            
            # Try SymSpell first
            suggestions = self.sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=3)
            if suggestions and suggestions[0].distance <= 2:  # Edit distance <= 2
                candidate = suggestions[0].term
                corrected_tokens.append(candidate)
                if candidate != token:
                    changed = True
                    token_details.append({
                        "original": token,
                        "corrected": candidate,
                        "confidence": 1.0 - (suggestions[0].distance / 10.0)  # Higher confidence for lower distance
                    })
                continue
            
            # Fallback to embedding: find most similar words from vocab
            token_embedding = self.embedding_model.encode([token], convert_to_tensor=True)
            
            # Get top 50 most similar words (for efficiency)
            vocab_list = list(self.all_vocab)
            if len(vocab_list) > 50:
                # Encode vocab in batches
                vocab_embeddings = self.embedding_model.encode(vocab_list[:50], convert_to_tensor=True)
                similarities = F.cosine_similarity(token_embedding, vocab_embeddings, dim=1)
                top_similarities, top_indices = torch.topk(similarities, k=5)
                
                # Filter by similarity threshold > 0.7
                valid_indices = [i for i, sim in enumerate(top_similarities) if sim > 0.7]
                if valid_indices:
                    best_idx = valid_indices[0]  # Highest similarity
                    best_candidate = vocab_list[top_indices[best_idx]]
                    confidence = top_similarities[best_idx].item()
                else:
                    best_candidate = token  # No good match
                    confidence = 0.0
            else:
                best_candidate = token
                confidence = 0.0
            
            corrected_tokens.append(best_candidate)
            if best_candidate != token:
                changed = True
                token_details.append({
                    "original": token,
                    "corrected": best_candidate,
                    "confidence": confidence
                })
        
        normalized_query = " ".join(corrected_tokens)
        
        return {
            "original_query": query,
            "normalized_query": normalized_query,
            "changed": changed,
            "tokens": token_details
        }

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

    def _correct_with_mistral(self, query: str) -> str:
        """
        Use Mistral 7B model for advanced spell checking with e-commerce domain knowledge.
        This is the LAST RESORT fallback for typos that other models can't handle.
        Designed to never fail - always returns a valid correction.
        """
        original_query = query.lower().strip()
        
        try:
            # STEP 1: Apply typo mappings first (fast, accurate)
            words = original_query.split()
            pre_corrected_words = []
            for word in words:
                if word in self.typo_mappings:
                    pre_corrected_words.append(self.typo_mappings[word])
                else:
                    pre_corrected_words.append(word)
            pre_corrected = " ".join(pre_corrected_words)
            
            # If all words are in known vocabulary, return early
            all_known = all(
                w in self.typo_mappings or 
                w in self.all_vocab or 
                self._is_numeric_or_unit(w)
                for w in pre_corrected.split()
            )
            if all_known:
                return pre_corrected
            
            # STEP 2: Lazy load Mistral model
            if not hasattr(self, 'mistral_model') or self.mistral_model is None:
                logger.info("Loading Mistral model...")
                self.mistral_model = Llama(
                    model_path=self.mistral_model_path,
                    n_ctx=4096,
                    n_threads=4,
                    verbose=False
                )
                logger.info("Mistral model loaded successfully.")
            
            # Build comprehensive brand list for context
            top_brands = [
                "apple", "samsung", "sony", "lg", "asus", "msi", "dell", "hp", "lenovo", "acer",
                "nvidia", "amd", "intel", "logitech", "corsair", "razer", "steelseries", "hyperx",
                "microsoft", "google", "xiaomi", "huawei", "oppo", "realme", "oneplus", "motorola",
                "jbl", "bose", "sennheiser", "audio-technica", "beyerdynamic", "shure", "marshall",
                "kingston", "crucial", "seagate", "western digital", "sandisk", "toshiba",
                "gigabyte", "asrock", "evga", "zotac", "palit", "sapphire", "powercolor",
                "rog", "tuf", "strix", "alienware", "predator", "republic of gamers",
                "thinkpad", "zenbook", "vivobook", "xps", "inspiron", "galaxy", "iphone", "ipad",
                "macbook", "airpods", "playstation", "xbox", "nintendo", "switch",
                "geforce", "radeon", "ryzen", "core", "rtx", "gtx", "rx"
            ]
            
            # Enhanced system prompt with comprehensive e-commerce domain knowledge
            system_prompt = f"""You are an expert e-commerce spell checker specialized in electronics and technology products.

KNOWN BRANDS: {', '.join(top_brands[:30])}

RULES:
1. ONLY fix obvious spelling mistakes
2. NEVER change correct brand names, model numbers, or technical terms
3. Keep numbers and units (gb, tb, mhz, hz, w, v, a, mah) unchanged
4. Preserve the original meaning - do not add or remove words
5. Return ONLY the corrected text, nothing else

COMMON TYPO PATTERNS:
- Missing letters: "samung" → "samsung", "nvida" → "nvidia"
- Swapped letters: "smasung" → "samsung", "nvidai" → "nvidia"
- Adjacent key errors: "logitecj" → "logitech", "corsaur" → "corsair"
- Double letters: "appple" → "apple", "samsuung" → "samsung"
- Wrong vowels: "logetech" → "logitech", "rasberry" → "raspberry"

EXAMPLES:
- "iphnoe 15 pro max" → "iphone 15 pro max"
- "samsng galaxy s24 ultra" → "samsung galaxy s24 ultra"
- "nvidea geforce rtx 4090" → "nvidia geforce rtx 4090"
- "asus rg strix motherboard" → "asus rog strix motherboard"
- "logitec g502 gaming mouse" → "logitech g502 gaming mouse"
- "corsiar k70 rgb keyboard" → "corsair k70 rgb keyboard"
- "razr blade 15 laptop" → "razer blade 15 laptop"
- "lenvo thinkpad x1 carbon" → "lenovo thinkpad x1 carbon"
- "i ned a chep gaming laptop" → "i need a cheap gaming laptop"
- "best wireles earbuds 2024" → "best wireless earbuds 2024"
- "mechancal keyboard with rgb" → "mechanical keyboard with rgb"
- "4k moniter for gaming" → "4k monitor for gaming"
- "usb c charger 65w" → "usb c charger 65w"
- "bluethooth speaker portable" → "bluetooth speaker portable"
- "grafics card for pc" → "graphics card for pc"
- "msi mpg z790 motherboard" → "msi mpg z790 motherboard"
- "hyperx cloud ii headset" → "hyperx cloud ii headset"
- "steelseries arctis 7 wireles" → "steelseries arctis 7 wireless"
- "jbl flip 6 waterprrof" → "jbl flip 6 waterproof"
- "bose quietcomfrt earbuds" → "bose quietcomfort earbuds"
- "mbifia rtx 4080" → "nvidia rtx 4080"
- "lpgitech g pro x" → "logitech g pro x"
"""

            # Create the prompt (without <s> - llama.cpp adds it automatically)
            prompt = f"""[INST] {system_prompt}

Fix any spelling errors in this search query. Return ONLY the corrected text:

"{pre_corrected}"

[/INST]"""
            
            # CRITICAL: Reset KV cache before each call to ensure consistent results
            # Without this, previous prompts pollute the cache and cause inconsistent outputs
            self.mistral_model.reset()
            
            # Generate response with optimized parameters for accuracy
            output = self.mistral_model(
                prompt,
                max_tokens=len(query.split()) * 4 + 30,
                temperature=0.01,
                top_p=0.95,
                top_k=10,
                repeat_penalty=1.1,
                stop=["</s>", "[INST]", "\n\n", "Query:", "Fix any"]
            )
            
            corrected = output['choices'][0]['text'].strip()
            
            # STEP 3: Robust response cleaning and validation
            corrected = self._clean_mistral_response(corrected, original_query)
            
            # STEP 4: Validate the correction
            is_valid, reason = self._validate_mistral_correction(original_query, corrected)
            
            if not is_valid:
                logger.debug(f"Mistral validation failed ({reason}), using SymSpell fallback")
                return self._correct_with_symspell(original_query)['normalized_query']
            
            return corrected.lower()
            
        except Exception as e:
            logger.warning(f"Mistral correction failed: {e}, falling back to SymSpell")
            # NEVER fail - always return something valid
            try:
                return self._correct_with_symspell(original_query)['normalized_query']
            except:
                return original_query  # Ultimate fallback
    
    def _clean_mistral_response(self, response: str, original_query: str) -> str:
        """Clean and sanitize Mistral model response."""
        if not response:
            return original_query
        
        cleaned = response.lower().strip()
        
        # Remove commas that the model might add (we want space-separated words)
        cleaned = cleaned.replace(',', ' ')
        # Normalize multiple spaces to single space
        cleaned = ' '.join(cleaned.split())
        
        # If response is just a prefix with no content, return original
        just_prefix_patterns = [
            'correction:', 'corrected:', 'fixed:', 'result:', 'output:', 
            'answer:', 'response:', 'text:', 'query:'
        ]
        
        for pattern in just_prefix_patterns:
            if cleaned == pattern or cleaned == pattern.rstrip(':'):
                return original_query
        
        # Remove common prefixes/suffixes that models add
        prefixes_to_remove = [
            'corrected:', 'fixed:', 'result:', 'output:', 'answer:',
            'the corrected text is:', 'here is the corrected text:',
            'corrected text:', 'the fixed query is:', 'corrected query:',
            'correction:', 'the correction is:', 'fixed text:',
            'here is the fixed text:', 'the answer is:',
            '"', "'", '`', '*'
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Also check for prefixes at the end (like "corrected:" at end of response)
        for prefix in prefixes_to_remove:
            if cleaned.endswith(prefix.rstrip(':')):
                cleaned = cleaned[:-len(prefix.rstrip(':'))].strip()
        
        # Remove trailing quotes and punctuation
        cleaned = cleaned.strip('"\'`.,;:*')
        
        # Remove any explanation text after the correction
        for separator in ['\n', ' - ', '(', '[', 'note:', 'explanation:', '---', '***']:
            if separator in cleaned:
                cleaned = cleaned.split(separator)[0].strip()
        
        # If cleaned is empty or just whitespace, return original
        if not cleaned or cleaned.isspace():
            return original_query
        
        # If response is too short compared to original, return original
        if len(cleaned) < max(2, len(original_query) * 0.3):
            return original_query
        
        # If response still contains artifact words, return original
        artifact_words = ['corrected', 'fixed', 'result', 'output', 'answer', 'correction', 'response']
        if any(cleaned == word or cleaned.startswith(word + ':') for word in artifact_words):
            return original_query
        
        return cleaned
    
    def _validate_mistral_correction(self, original: str, corrected: str) -> tuple:
        """
        Validate Mistral correction to prevent hallucinations.
        Returns (is_valid, reason) tuple.
        """
        if not corrected:
            return False, "empty_response"
        
        orig_words = original.split()
        corr_words = corrected.split()
        
        # Check word count - shouldn't change dramatically
        if len(corr_words) > len(orig_words) + 2:
            return False, "too_many_words_added"
        
        if len(corr_words) < max(1, len(orig_words) - 2):
            return False, "too_many_words_removed"
        
        # Check that at least 50% of original characters are preserved
        orig_chars = set(original.replace(' ', ''))
        corr_chars = set(corrected.replace(' ', ''))
        common_chars = orig_chars.intersection(corr_chars)
        
        if len(common_chars) < len(orig_chars) * 0.4:
            return False, "too_different"
        
        # Check for obviously wrong responses (common LLM artifacts)
        bad_patterns = [
            'i cannot', 'i can\'t', 'sorry', 'as an ai', 'here is',
            'the corrected', 'note:', 'explanation:', 'however',
            '```', '**', '__', '##', '</', 'http', 'www.'
        ]
        
        for pattern in bad_patterns:
            if pattern in corrected.lower():
                return False, f"contains_artifact:{pattern}"
        
        # All checks passed
        return True, "valid"

    def _init_e5_onnx(self):
        """Initialize E5 ONNX model for semantic similarity correction."""
        if not hasattr(self, '_e5_model'):
            try:
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                from transformers import AutoTokenizer
                
                model_path = os.path.join(os.path.dirname(__file__), "..", "models", "e5-small-v2-onnx")
                
                # Check if model exists locally
                if os.path.exists(model_path):
                    self._e5_model = ORTModelForFeatureExtraction.from_pretrained(model_path)
                    self._e5_tokenizer = AutoTokenizer.from_pretrained(model_path)
                else:
                    # Download from hub
                    self._e5_model = ORTModelForFeatureExtraction.from_pretrained(
                        "intfloat/e5-small-v2",
                        export=True
                    )
                    self._e5_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
                    # Save for next time
                    os.makedirs(model_path, exist_ok=True)
                    self._e5_model.save_pretrained(model_path)
                    self._e5_tokenizer.save_pretrained(model_path)
                
                # Build vocabulary embeddings for semantic matching
                self._build_vocab_embeddings()
                
            except Exception as e:
                logger.warning(f"E5 ONNX initialization failed: {e}")
                self._e5_model = None
                self._e5_tokenizer = None

    def _build_vocab_embeddings(self):
        """Build embeddings for vocabulary words for semantic matching."""
        import numpy as np
        
        # Collect all unique words from vocabularies
        vocab_words = set()
        
        # Add brand products (single words only, skip comments and multi-word entries)
        for product in self.brand_products_vocab:
            word = product.lower().strip()
            if word and not word.startswith('#') and ' ' not in word:
                vocab_words.add(word)
        
        # Add electronics vocab (single words only)
        for term in self.electronics_vocab:
            word = term.lower().strip()
            if word and not word.startswith('#') and ' ' not in word:
                vocab_words.add(word)
        
        # Add domain vocab (single words only)
        for word in self.domain_vocab:
            w = word.lower().strip()
            if w and not w.startswith('#') and ' ' not in w:
                vocab_words.add(w)
        
        # Add typo mapping targets (single words only)
        for correct_word in self.typo_mappings.values():
            word = correct_word.lower().strip()
            if word and not word.startswith('#') and ' ' not in word:
                vocab_words.add(word)
        
        self._vocab_list = list(vocab_words)
        
        if len(self._vocab_list) > 0 and self._e5_model is not None:
            # E5 requires "query: " or "passage: " prefix
            texts = ["passage: " + w for w in self._vocab_list]
            
            # Batch embedding for efficiency
            batch_size = 64
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self._e5_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                outputs = self._e5_model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
                all_embeddings.append(embeddings)
            
            self._vocab_embeddings = np.vstack(all_embeddings)
            # Normalize for cosine similarity
            self._vocab_embeddings = self._vocab_embeddings / np.linalg.norm(self._vocab_embeddings, axis=1, keepdims=True)

    def _get_e5_embedding(self, text: str):
        """Get E5 embedding for a text."""
        import numpy as np
        
        inputs = self._e5_tokenizer(f"query: {text}", return_tensors="pt", truncation=True)
        outputs = self._e5_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def _find_similar_word(self, word: str, threshold: float = 0.85) -> tuple:
        """Find most similar word in vocabulary using E5 embeddings."""
        import numpy as np
        
        # Get embedding for input word
        word_embedding = self._get_e5_embedding(word)
        
        # Compute cosine similarities
        similarities = np.dot(self._vocab_embeddings, word_embedding)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_word = self._vocab_list[best_idx]
        
        if best_score >= threshold and best_word.lower() != word.lower():
            return best_word, best_score
        
        return word, 1.0  # Return original if no good match

    def _correct_with_e5_onnx(self, query: str) -> dict:
        """
        Correct typos using E5 ONNX semantic similarity.
        Uses SymSpell first, then E5 for low-confidence corrections.
        """
        import time
        import numpy as np
        
        start_time = time.time()
        
        # Initialize E5 if needed
        self._init_e5_onnx()
        
        if self._e5_model is None:
            # Fallback to SymSpell if E5 failed to load
            return self._correct_with_symspell(query)
        
        # Use SymSpell first for typo correction, then E5 for semantic verification
        symspell_result = self._correct_with_symspell(query)
        
        # If SymSpell made no changes or high confidence, use it directly
        low_confidence_tokens = [t for t in symspell_result.get('tokens', []) if t.get('confidence', 1.0) < 0.85]
        
        if not low_confidence_tokens:
            # Add E5 model marker
            symspell_result['model'] = 'e5_onnx (symspell path)'
            return symspell_result
        
        # For low confidence corrections, verify with E5 semantic similarity
        words = symspell_result['normalized_query'].split()
        token_details = symspell_result.get('tokens', [])
        corrected_words = []
        changed = symspell_result.get('changed', False)
        
        for word in words:
            token_start = time.time()
            original_word = word
            
            # Check if this word had a low confidence correction
            low_conf_token = next((t for t in low_confidence_tokens if t.get('corrected') == word), None)
            
            if low_conf_token:
                # Use E5 to find better match
                e5_corrected, similarity = self._find_similar_word(word, threshold=0.85)
                if similarity > low_conf_token.get('confidence', 0):
                    corrected_words.append(e5_corrected)
                    if e5_corrected != word:
                        changed = True
                        # Update token details
                        for t in token_details:
                            if t.get('corrected') == word:
                                t['corrected'] = e5_corrected
                                t['confidence'] = similarity
                                t['source'] = 'e5_semantic'
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        total_latency = time.time() - start_time
        corrected_query = " ".join(corrected_words)
        
        return {
            "original_query": query,
            "normalized_query": corrected_query,
            "changed": changed,
            "model": "e5_onnx",
            "tokens": token_details,
            "latency": {
                "total_ms": round(total_latency * 1000, 2),
                "tokens_per_second": round(len(words) / total_latency, 2) if total_latency > 0 else 0.0
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

    def _correct_with_symspell_keyboard(self, query: str) -> dict:
        """
        Hybrid SymSpell + Keyboard correction (optimized).
        Preserves price patterns like $100, 500₺, 100 euro etc.
        
        Strategy:
        1. Run SymSpell first (fast dictionary lookup)
        2. ONLY if SymSpell has low confidence (<0.85), try keyboard proximity
        3. Pick the best result based on confidence
        
        This is faster than pure keyboard because keyboard is only used as fallback.
        """
        import time
        
        start_time = time.time()
        
        if not query.strip():
            return {
                "original_query": query,
                "normalized_query": query,
                "changed": False,
                "model": "symspell_keyboard",
                "tokens": [],
                "latency": {"total_ms": 0.0, "tokens_per_second": 0.0}
            }
        
        # Extract and preserve price patterns before processing
        processed_query, preserved_prices = self._extract_and_preserve_prices(query)
        
        words = processed_query.lower().strip().split()
        corrected_words = []
        token_details = []
        changed = False
        
        # Lazy build vocab list only if needed
        vocab_list = None
        vocab_set = None
        
        for word in words:
            original_word = word
            
            # Normalize repeated characters first
            normalized_word = normalize_repeated_chars(word)
            if normalized_word != word:
                word = normalized_word
                changed = True
            
            # 1. Check typo mappings first (highest priority)
            if word in self.typo_mappings:
                corrected = self.typo_mappings[word]
                corrected_words.append(corrected)
                if corrected != original_word:
                    changed = True
                    token_details.append({
                        "original": original_word,
                        "corrected": corrected,
                        "confidence": 0.99,
                        "source": "typo_mapping"
                    })
                continue
            
            # 2. Skip protected patterns
            if self._is_numeric_or_unit(word):
                corrected_words.append(word)
                continue
            
            # 3. Check exact vocab match
            if word in self.all_vocab:
                corrected_words.append(word)
                continue
            
            # 4. Try SymSpell first (fast)
            symspell_result = self._find_best_candidate(word)
            symspell_word = symspell_result['word']
            symspell_conf = symspell_result['confidence']
            
            # If SymSpell is confident (>= 0.85), use it directly - no keyboard needed
            if symspell_conf >= 0.85:
                if symspell_word != word:
                    corrected_words.append(symspell_word)
                    changed = True
                    token_details.append({
                        "original": original_word,
                        "corrected": symspell_word,
                        "confidence": round(symspell_conf, 4),
                        "source": "symspell"
                    })
                else:
                    corrected_words.append(word)
                continue
            
            # 5. SymSpell has low confidence - try keyboard proximity as fallback
            keyboard_word = word
            keyboard_conf = 0.0
            
            if 3 <= len(word) <= 15:
                # Lazy init vocab list
                if vocab_list is None:
                    vocab_set = self.brand_products_vocab | self.electronics_vocab | self.domain_vocab
                    vocab_list = list(vocab_set)
                
                keyboard_dist = float('inf')
                for vocab_word in vocab_list:
                    if abs(len(word) - len(vocab_word)) > 3:
                        continue
                    
                    dist = keyboard_weighted_edit_distance(word, vocab_word)
                    normalized_dist = dist / max(len(word), len(vocab_word))
                    
                    if normalized_dist < keyboard_dist and normalized_dist < 0.5:
                        keyboard_dist = normalized_dist
                        keyboard_word = vocab_word
                        keyboard_conf = 1.0 - normalized_dist
            
            # 6. Pick the best between SymSpell and Keyboard
            if keyboard_conf > symspell_conf and keyboard_word != word:
                corrected = keyboard_word
                source = "keyboard"
                confidence = keyboard_conf
            elif symspell_word != word:
                corrected = symspell_word
                source = "symspell"
                confidence = symspell_conf
            else:
                corrected = word
                source = "unchanged"
                confidence = 1.0
            
            corrected_words.append(corrected)
            
            if corrected != original_word:
                changed = True
                token_details.append({
                    "original": original_word,
                    "corrected": corrected,
                    "confidence": round(confidence, 4),
                    "source": source
                })
        
        total_latency = time.time() - start_time
        corrected_query = " ".join(corrected_words)
        
        # Restore preserved price patterns
        corrected_query = self._restore_prices(corrected_query, preserved_prices)
        
        return {
            "original_query": query,
            "normalized_query": corrected_query,
            "changed": changed,
            "model": "symspell_keyboard",
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
            model_type: One of 'byt5', 'qwen', 'qwen-1.5b', 'llama'
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
        
        elif model_type in ["qwen", "qwen-1.5b"]:
            # Qwen - LoRA adapter on base model (Instruct versions)
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            # Determine base model - using Instruct versions
            if model_type == "qwen":
                base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            else:
                base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            
            self._finetuned_tokenizers[model_type] = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self._finetuned_models[model_type] = PeftModel.from_pretrained(base_model, model_path)
            
            # Move to MPS if on Apple Silicon
            if not torch.cuda.is_available() and torch.backends.mps.is_available():
                self._finetuned_models[model_type] = self._finetuned_models[model_type].to("mps")
        
        elif model_type == "llama":
            # Llama - LoRA adapter on base model (requires HF token)
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            # Using Instruct version (matches adapter_config.json)
            base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
            
            self._finetuned_tokenizers[model_type] = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self._finetuned_models[model_type] = PeftModel.from_pretrained(base_model, model_path)
            
            # Move to MPS if on Apple Silicon
            if not torch.cuda.is_available() and torch.backends.mps.is_available():
                self._finetuned_models[model_type] = self._finetuned_models[model_type].to("mps")
        
        elif model_type == "ministral":
            # Ministral 3B - LoRA adapter on base model
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            base_model_name = "mistralai/Ministral-3-3B-Instruct-2512"
            
            self._finetuned_tokenizers[model_type] = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            self._finetuned_models[model_type] = PeftModel.from_pretrained(base_model, model_path)
            
            # Move to MPS if on Apple Silicon
            if not torch.cuda.is_available() and torch.backends.mps.is_available():
                self._finetuned_models[model_type] = self._finetuned_models[model_type].to("mps")
        
        elif model_type == "phi3":
            # Phi-3 Mini - LoRA adapter on base model
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            base_model_name = "microsoft/Phi-3-mini-4k-instruct"
            
            self._finetuned_tokenizers[model_type] = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                attn_implementation="eager"  # For MPS compatibility
            )
            self._finetuned_models[model_type] = PeftModel.from_pretrained(base_model, model_path)
            
            # Move to MPS if on Apple Silicon
            if not torch.cuda.is_available() and torch.backends.mps.is_available():
                self._finetuned_models[model_type] = self._finetuned_models[model_type].to("mps")
        
        logger.info(f"Fine-tuned model loaded: {model_type}")

    def _correct_with_finetuned(self, query: str, model_type: str = "llama") -> dict:
        """
        Correct typos using a fine-tuned model.
        Preserves price patterns like $100, 500₺, 100 euro etc.
        
        Args:
            query: The search query to correct
            model_type: One of 'byt5', 'qwen', 'qwen-1.5b', 'llama'
            
        Returns:
            dict with original_query, normalized_query, changed, model, latency_ms
        """
        start_time = time.time()
        
        if not query or not query.strip():
            return {
                "original_query": query,
                "normalized_query": query,
                "changed": False,
                "model": model_type,
                "latency_ms": 0.0
            }
        
        # Extract and preserve price patterns before processing
        processed_query, preserved_prices = self._extract_and_preserve_prices(query)
        
        # Load model if not already loaded
        try:
            self._load_finetuned_model(model_type)
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            # Fallback to SymSpell
            return self._correct_with_symspell(query)
        
        model = self._finetuned_models[model_type]
        tokenizer = self._finetuned_tokenizers[model_type]
        
        try:
            if model_type == "byt5":
                # ByT5 format: "correct: <query>"
                input_text = f"correct: {processed_query}"
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
            
            elif model_type == "phi3":
                # Phi-3 chat format: <|user|>\n{query}<|end|>\n<|assistant|>\n
                prompt = f"<|user|>\n{processed_query}<|end|>\n<|assistant|>\n"
                inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
                
                # Move to same device as model
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=False  # Fix for DynamicCache issue with Phi-3
                    )
                
                full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the output part after assistant tag
                if "<|assistant|>" in full_output:
                    corrected = full_output.split("<|assistant|>")[-1].strip()
                else:
                    corrected = full_output[len(prompt):].strip()
                
                # Clean up special tokens and extra text
                corrected = corrected.split("<|end|>")[0].strip()
                corrected = corrected.split("\n")[0].strip()
            
            else:
                # Qwen/Llama/Ministral format: "Input: <query>\nOutput:"
                prompt = f"Input: {processed_query}\nOutput:"
                inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
                
                # Move to same device as model
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the output part
                if "Output:" in full_output:
                    corrected = full_output.split("Output:")[-1].strip()
                else:
                    corrected = full_output[len(prompt):].strip()
                
                # Clean up any extra text
                corrected = corrected.split("\n")[0].strip()
        
        except Exception as e:
            logger.error(f"Error during {model_type} inference: {e}")
            corrected = processed_query
        
        # Restore preserved price patterns
        corrected = self._restore_prices(corrected, preserved_prices)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "original_query": query,
            "normalized_query": corrected,
            "changed": corrected.lower() != query.lower(),
            "model": model_type,
            "latency_ms": round(latency_ms, 2)
        }

    def _correct_with_byt5(self, query: str) -> dict:
        """Correct using ByT5 fine-tuned model."""
        return self._correct_with_finetuned(query, "byt5")

    def _correct_with_qwen(self, query: str) -> dict:
        """Correct using Qwen2.5-0.5B fine-tuned model."""
        return self._correct_with_finetuned(query, "qwen")

    def _correct_with_qwen_1_5b(self, query: str) -> dict:
        """Correct using Qwen2.5-1.5B fine-tuned model."""
        return self._correct_with_finetuned(query, "qwen-1.5b")

    def _correct_with_llama(self, query: str) -> dict:
        """Correct using Llama-3.2-1B fine-tuned model (84% accuracy)."""
        return self._correct_with_finetuned(query, "llama")

    def _correct_with_ministral(self, query: str) -> dict:
        """Correct using Ministral-3B fine-tuned model (expected ~86%+ accuracy)."""
        return self._correct_with_finetuned(query, "ministral")

    def _correct_with_phi3(self, query: str) -> dict:
        """Correct using Phi-3-Mini fine-tuned model (expected ~85%+ accuracy)."""
        return self._correct_with_finetuned(query, "phi3")

    def _correct_with_smart_hybrid(self, query: str) -> dict:
        """
        Smart Hybrid: SymSpell first, then fine-tuned model fallback.
        
        Use this when:
        - SymSpell makes no changes but the query might still have typos
        - SymSpell has low confidence
        - Zero retrieval scenario (backend can trigger this)
        
        Flow:
        1. Try SymSpell (fast, ~0.1ms)
        2. If no change or low confidence, try Llama (best accuracy, ~260ms)
        """
        start_time = time.time()
        
        # Step 1: Try SymSpell first (fast)
        symspell_result = self._correct_with_symspell(query)
        
        # If SymSpell made changes, trust it
        if symspell_result.get("changed", False):
            # Check confidence of all tokens
            tokens = symspell_result.get("tokens", [])
            low_confidence = any(t.get("confidence", 1.0) < 0.8 for t in tokens)
            
            if not low_confidence:
                symspell_result["model"] = "smart_hybrid_symspell"
                return symspell_result
        
        # Step 2: SymSpell didn't change or low confidence - try Llama
        try:
            llama_result = self._correct_with_llama(query)
            llama_result["model"] = "smart_hybrid_llama"
            llama_result["symspell_result"] = symspell_result.get("normalized_query", query)
            return llama_result
        except Exception as e:
            logger.warning(f"Llama fallback failed: {e}, returning SymSpell result")
            symspell_result["model"] = "smart_hybrid_symspell_only"
            return symspell_result