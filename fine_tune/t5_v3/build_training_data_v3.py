#!/usr/bin/env python3
"""
T5-Large v3 Training Data Builder
==================================
Builds high-quality English e-commerce spelling correction training data.

Key improvements over v2.1:
  - Realistic phonetic typo patterns (vowel shifts, consonant confusion)
  - Compound typos (2-3 errors per word) at higher rates
  - Broad e-commerce coverage (fashion, beauty, grocery, auto, not just electronics)
  - Lower identity ratio (20-25% vs 40%)
  - Real-world query structures (long queries, with numbers/sizes/colors)
  - Weighted category balancing for even coverage
  - Deduplication + quality filters

Output: data/train_v3.jsonl, data/eval_v3.jsonl
Format: {"input_text": "correct: corsiar keybord", "target_text": "corsair keyboard", "category": "..."}

Usage:
  python build_training_data_v3.py
  python build_training_data_v3.py --target 400000
"""

import argparse
import json
import math
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

random.seed(42)

SCRIPT_DIR = Path(__file__).parent
CORRECTION_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = CORRECTION_DIR / "data"
OUT_DIR = SCRIPT_DIR / "data"

# ═══════════════════════════════════════════════════════════════
# KEYBOARD LAYOUTS + PHONETIC CONFUSION MAPS
# ═══════════════════════════════════════════════════════════════

QWERTY = {
    'q': 'wa12', 'w': 'qeas23', 'e': 'wrsd34', 'r': 'etdf45',
    't': 'ryfg56', 'y': 'tugh67', 'u': 'yihj78', 'i': 'uojk89',
    'o': 'ipkl90', 'p': 'ol0',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
    'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huiknm', 'k': 'jiolm',
    'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv',
    'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
    '1': 'q2', '2': '1w3', '3': '2e4', '4': '3r5', '5': '4t6',
    '6': '5y7', '7': '6u8', '8': '7i9', '9': '8o0', '0': '9p',
}

# Phonetically confused letter pairs (English)
# These cause real human typos, not just keyboard proximity
PHONETIC_CONFUSIONS = {
    'a': ['e', 'u'],        # "samsung" -> "sumsung", "sandal" -> "sendal"
    'e': ['a', 'i'],        # "leather" -> "lather", "headset" -> "hidset"
    'i': ['e', 'y'],        # "wireless" -> "wereless", "mini" -> "miny"
    'o': ['u', 'a'],        # "monitor" -> "munitor", "logitech" -> "lagitech"
    'u': ['o', 'a'],        # "bluetooth" -> "bloetooth"
    'c': ['k', 's'],        # "corsair" -> "korsair"
    'k': ['c'],             # "keyboard" -> "ceyboard"
    's': ['z', 'c'],        # "samsung" -> "zamsung"
    'z': ['s'],             # "razer" -> "raser"
    'f': ['ph', 'v'],       # "phone" -> "fone" (reverse)
    'ph': ['f'],            # "earphones" -> "earfones"
    'th': ['t'],            # "bluetooth" -> "blutoot"
    'ck': ['k', 'c'],       # "black" -> "blak"
    'ee': ['ea', 'ie'],     # "steel" -> "steal"
    'ea': ['ee'],           # "headphones" -> "heedphones"
    'ou': ['ow', 'u'],      # "mouse" -> "mowse"
    'oo': ['u'],            # "bluetooth" -> "blututh"
    'ie': ['ei'],           # "series" -> "sereis"
    'ei': ['ie'],           # "receive" -> "recieve"
}

# Common letter omission patterns (people skip these)
OMISSION_PATTERNS = {
    'tion': ['tion', 'toin', 'tin', 'shon'],
    'ing': ['ing', 'in', 'ig'],
    'ment': ['ment', 'mnt', 'ment'],
    'ness': ['ness', 'nss', 'nes'],
    'able': ['able', 'abl', 'ble'],
    'ible': ['ible', 'ibl', 'ble'],
    'ght': ['ght', 'gt', 'ht'],
    'ough': ['ough', 'uff', 'off'],
}


# ═══════════════════════════════════════════════════════════════
# COMPREHENSIVE E-COMMERCE VOCABULARY
# ═══════════════════════════════════════════════════════════════

BRAND_CATALOG = {
    # ── Electronics ──
    "apple":       ["iphone", "macbook", "airpods", "ipad", "apple watch", "imac", "mac mini", "macbook air", "macbook pro", "iphone 15", "iphone 14", "iphone 13"],
    "samsung":     ["galaxy", "galaxy s24", "galaxy s23", "galaxy buds", "galaxy watch", "galaxy tab", "galaxy z flip", "galaxy z fold", "monitor", "tv", "ssd", "qled tv"],
    "sony":        ["playstation", "ps5", "headphones", "earbuds", "wh-1000xm5", "wf-1000xm5", "bravia", "camera", "alpha", "dualsense"],
    "nvidia":      ["rtx 4090", "rtx 4080", "rtx 4070", "rtx 4060", "rtx 3080", "rtx 3060", "geforce", "gpu", "graphics card", "shield"],
    "amd":         ["ryzen 5", "ryzen 7", "ryzen 9", "radeon", "rx 7900", "rx 7800", "rx 7600", "processor", "cpu", "threadripper"],
    "intel":       ["core i5", "core i7", "core i9", "core ultra", "processor", "cpu", "arc", "nuc"],
    "asus":        ["rog", "tuf gaming", "zenbook", "vivobook", "proart", "rog strix", "rog zephyrus", "motherboard", "router", "monitor"],
    "dell":        ["xps", "inspiron", "alienware", "latitude", "optiplex", "monitor", "ultrasharp"],
    "hp":          ["omen", "pavilion", "spectre", "envy", "victus", "elitebook", "printer", "monitor"],
    "lenovo":      ["thinkpad", "ideapad", "legion", "yoga", "tab", "thinkcentre"],
    "acer":        ["predator", "nitro", "aspire", "swift", "chromebook", "monitor"],
    "msi":         ["katana", "stealth", "raider", "creator", "motherboard", "monitor", "graphics card"],
    "razer":       ["deathadder", "viper", "huntsman", "blackwidow", "kraken", "blade", "basilisk", "mouse", "keyboard", "headset"],
    "logitech":    ["g pro", "g502", "mx master", "mx keys", "g915", "g733", "g305", "mouse", "keyboard", "webcam", "headset"],
    "corsair":     ["k70", "k100", "void", "vengeance", "hs80", "scimitar", "harpoon", "keyboard", "mouse", "headset", "ram"],
    "steelseries": ["arctis", "apex", "aerox", "rival", "prime", "headset", "mouse", "keyboard", "mousepad"],
    "hyperx":      ["cloud", "alloy", "pulsefire", "fury", "headset", "keyboard", "mouse", "ram"],
    "kingston":    ["fury", "ssd", "ram", "memory card", "flash drive", "a2000", "nv2"],
    "seagate":     ["barracuda", "firecuda", "ironwolf", "expansion", "hard drive", "hdd", "ssd", "external drive"],
    "sandisk":     ["extreme", "ultra", "ssd", "memory card", "flash drive", "usb"],
    "crucial":     ["p5", "mx500", "ballistix", "ram", "ssd", "memory"],
    "western digital": ["wd blue", "wd black", "my passport", "my book", "wd red", "ssd", "hard drive"],
    "anker":       ["powercore", "soundcore", "eufy", "charger", "power bank", "cable", "earbuds", "speaker"],
    "bose":        ["quietcomfort", "soundlink", "noise cancelling", "headphones", "speaker", "earbuds", "soundbar"],
    "sennheiser":  ["momentum", "hd 660s", "ie 600", "headphones", "earbuds", "microphone"],
    "jbl":         ["flip", "charge", "tune", "club", "speaker", "headphones", "earbuds", "soundbar"],
    "lg":          ["oled tv", "ultragear", "gram", "monitor", "tv", "soundbar", "refrigerator"],
    "microsoft":   ["surface", "surface pro", "surface laptop", "xbox", "xbox series x", "xbox controller"],
    "google":      ["pixel", "pixel 8", "pixel buds", "chromecast", "nest", "pixel watch"],
    "oneplus":     ["nord", "buds", "watch", "phone", "12", "11"],
    "xiaomi":      ["redmi", "poco", "mi band", "earbuds", "phone", "tablet"],
    "marshall":    ["stanmore", "kilburn", "major", "monitor", "speaker", "headphones"],
    "dyson":       ["v15", "v12", "airwrap", "supersonic", "purifier", "vacuum", "fan", "hair dryer"],
    "canon":       ["eos", "r5", "r6", "powershot", "printer", "camera", "lens"],
    "nikon":       ["z9", "z8", "z6", "d850", "coolpix", "camera", "lens"],
    "gopro":       ["hero 12", "hero 11", "max", "camera", "action camera"],
    "dji":         ["mini 4", "mavic", "air 3", "osmo", "pocket", "drone", "action camera"],
    "nintendo":    ["switch", "switch oled", "joy-con", "pro controller", "zelda", "mario"],
    "tp-link":     ["deco", "archer", "router", "mesh wifi", "extender", "switch"],
    "netgear":     ["nighthawk", "orbi", "router", "mesh wifi", "extender"],
    # ── Fashion ──
    "nike":        ["air max", "air force 1", "dunk", "jordan", "air jordan", "pegasus", "vapormax", "hoodie", "joggers", "sneakers", "running shoes"],
    "adidas":      ["ultraboost", "yeezy", "stan smith", "superstar", "samba", "gazelle", "nmd", "hoodie", "joggers", "sneakers"],
    "puma":        ["suede", "rs-x", "cali", "future rider", "sneakers", "hoodie", "joggers"],
    "new balance":  ["550", "990", "574", "327", "2002r", "sneakers", "running shoes"],
    "converse":    ["chuck taylor", "all star", "run star", "sneakers"],
    "vans":        ["old skool", "sk8-hi", "authentic", "era", "slip-on", "sneakers"],
    "reebok":      ["classic", "nano", "club c", "sneakers", "training shoes"],
    "under armour": ["hovr", "charged", "project rock", "running shoes", "training shoes", "hoodie"],
    "gucci":       ["ace", "horsebit", "marmont", "bag", "sneakers", "belt", "wallet"],
    "louis vuitton": ["neverfull", "speedy", "keepall", "bag", "wallet", "belt"],
    "zara":        ["dress", "blazer", "jeans", "coat", "shirt", "shoes"],
    "h&m":         ["dress", "hoodie", "jeans", "jacket", "t-shirt"],
    "uniqlo":      ["heattech", "airism", "ultra light down", "jeans", "t-shirt", "jacket"],
    "levi's":      ["501", "511", "505", "jeans", "trucker jacket", "shorts"],
    "ralph lauren": ["polo", "shirt", "sweater", "jacket", "cap"],
    "tommy hilfiger": ["polo", "jacket", "jeans", "hoodie", "sneakers"],
    "calvin klein": ["underwear", "jeans", "perfume", "t-shirt", "bra"],
    "the north face": ["puffer", "nuptse", "jacket", "backpack", "hoodie", "fleece"],
    "patagonia":   ["down jacket", "fleece", "baggies", "nano puff", "backpack"],
    "columbia":    ["omni-heat", "jacket", "hiking boots", "fleece", "shorts"],
    "timberland":  ["boots", "6-inch", "earthkeepers", "chukka", "shoes"],
    "dr. martens": ["1460", "1461", "jadon", "boots", "shoes", "platform"],
    "birkenstock": ["arizona", "boston", "madrid", "sandals", "clogs"],
    "crocs":       ["classic clog", "literide", "bayaband", "sandals", "slides"],
    "ray-ban":     ["wayfarer", "aviator", "clubmaster", "round", "sunglasses"],
    "oakley":      ["holbrook", "sutro", "radar", "sunglasses", "goggles"],
    # ── Beauty & Personal Care ──
    "maybelline":  ["fit me", "lash sensational", "superstay", "foundation", "mascara", "lipstick", "concealer"],
    "l'oreal":     ["elvive", "revitalift", "true match", "shampoo", "foundation", "serum", "hair color"],
    "mac":         ["studio fix", "lipstick", "powder", "foundation", "eyeshadow", "concealer"],
    "nyx":         ["butter gloss", "epic ink", "lip liner", "setting spray", "eyebrow pencil"],
    "clinique":    ["moisture surge", "dramatically different", "moisturizer", "serum", "cleanser"],
    "the ordinary": ["niacinamide", "hyaluronic acid", "retinol", "aha bha", "serum", "moisturizer"],
    "cerave":      ["moisturizing cream", "cleanser", "sunscreen", "lotion", "serum"],
    "neutrogena":  ["hydro boost", "sunscreen", "cleanser", "moisturizer", "retinol"],
    "olaplex":     ["no. 3", "no. 4", "shampoo", "conditioner", "hair treatment", "bond repair"],
    "dove":        ["body wash", "shampoo", "deodorant", "soap", "conditioner"],
    "old spice":   ["deodorant", "body wash", "shampoo", "cologne"],
    "gillette":    ["fusion", "mach3", "razor", "shaving cream", "blades"],
    "oral-b":      ["electric toothbrush", "io series", "pro", "toothbrush", "replacement heads"],
    "philips":     ["oneblade", "norelco", "sonicare", "shaver", "toothbrush", "trimmer"],
    "braun":       ["series 9", "series 7", "silk-epil", "shaver", "trimmer", "epilator"],
    # ── Home & Kitchen ──
    "ikea":        ["kallax", "malm", "billy", "poang", "hemnes", "bookshelf", "desk", "sofa", "bed frame"],
    "kitchenaid":  ["mixer", "stand mixer", "blender", "food processor", "artisan"],
    "instant pot":  ["duo", "duo plus", "ultra", "pressure cooker", "air fryer"],
    "ninja":       ["blender", "air fryer", "food processor", "foodi", "creami"],
    "keurig":      ["k-elite", "k-supreme", "k-mini", "coffee maker", "k-cups"],
    "nespresso":   ["vertuo", "original", "lattissima", "coffee maker", "capsules"],
    "roomba":      ["j7", "i7", "s9", "combo", "robot vacuum"],
    "cuisinart":   ["food processor", "coffee maker", "toaster", "blender"],
    "breville":    ["barista express", "smart oven", "juicer", "espresso machine"],
    # ── Sports & Outdoors ──
    "yeti":        ["tumbler", "rambler", "cooler", "bottle", "mug"],
    "hydroflask":  ["water bottle", "tumbler", "mug", "food jar"],
    "garmin":      ["forerunner", "fenix", "venu", "instinct", "gps watch", "fitness tracker"],
    "fitbit":      ["charge", "versa", "sense", "inspire", "fitness tracker"],
    "peloton":     ["bike", "tread", "guide", "cycling shoes"],
    "theragun":    ["pro", "elite", "mini", "massage gun"],
    "wilson":      ["tennis racket", "basketball", "football", "golf balls"],
    # ── Grocery ──
    "nestle":      ["nescafe", "kitkat", "maggi", "nespresso", "cereal"],
    "coca-cola":   ["coke", "diet coke", "coke zero", "sprite", "fanta"],
    "pepsi":       ["pepsi max", "diet pepsi", "mountain dew", "gatorade"],
    "kellogg's":   ["corn flakes", "frosties", "crunchy nut", "special k", "cereal"],
    # ── Automotive ──
    "bosch":       ["wiper blades", "car battery", "spark plugs", "alternator", "brake pads"],
    "michelin":    ["tires", "all season", "pilot sport", "primacy"],
    "castrol":     ["edge", "gtx", "magnatec", "motor oil", "engine oil"],
    "meguiar's":   ["car wash", "wax", "polish", "detailing", "ceramic coating"],
}

# Common English product terms (not brand-specific)
PRODUCT_TERMS = [
    "laptop", "phone", "tablet", "headphones", "earbuds", "speaker", "monitor",
    "keyboard", "mouse", "webcam", "microphone", "printer", "scanner", "router",
    "charger", "cable", "adapter", "power bank", "case", "screen protector",
    "stand", "mount", "dock", "hub", "hard drive", "ssd", "ram", "memory card",
    "usb drive", "controller", "gamepad", "joystick", "vr headset",
    "smartwatch", "fitness tracker", "drone", "action camera", "ring light",
    "tripod", "gimbal", "portable monitor", "drawing tablet",
    "sneakers", "running shoes", "boots", "sandals", "dress", "jeans",
    "hoodie", "jacket", "t-shirt", "polo shirt", "sweater", "coat",
    "backpack", "wallet", "belt", "sunglasses", "watch", "bracelet",
    "foundation", "mascara", "lipstick", "moisturizer", "sunscreen",
    "shampoo", "conditioner", "body wash", "perfume", "serum", "cleanser",
    "blender", "coffee maker", "air fryer", "vacuum", "toaster",
    "pressure cooker", "food processor", "slow cooker", "rice cooker",
    "treadmill", "exercise bike", "yoga mat", "dumbbells", "resistance bands",
    "water bottle", "cooler", "tent", "sleeping bag", "hiking boots",
    "car charger", "dash cam", "car mount", "floor mats", "seat covers",
    "tires", "wiper blades", "motor oil", "car battery",
    "dog food", "cat food", "pet bed", "leash", "collar",
    "desk", "chair", "bookshelf", "lamp", "curtains", "rug", "pillow",
    "mattress", "bed frame", "sofa", "coffee table", "dining table",
    "stroller", "car seat", "diapers", "baby monitor", "high chair",
]

MODIFIERS = [
    "pro", "max", "mini", "plus", "ultra", "lite", "slim", "se",
    "2024", "2025", "2026", "new", "latest", "gen 2", "gen 3", "v2",
    "premium", "budget", "cheap", "affordable", "best", "top rated",
    "gaming", "wireless", "bluetooth", "portable", "foldable", "waterproof",
    "noise cancelling", "mechanical", "ergonomic", "adjustable", "rechargeable",
    "black", "white", "silver", "grey", "navy", "red", "blue", "pink", "green",
    "small", "medium", "large", "xl", "xxl", "size 10", "size 42",
    "men", "women", "kids", "unisex",
    "for home", "for office", "for travel", "for gym",
    "set", "bundle", "pack of 3", "2 pack", "refurbished", "renewed",
    "under 50", "under 100", "under 200", "under 500",
]

QUERY_TEMPLATES = [
    "{brand} {product}",
    "{brand} {product} {mod}",
    "best {brand} {product}",
    "{brand} {product} price",
    "buy {brand} {product}",
    "cheap {brand} {product}",
    "{brand} {product} review",
    "{brand} {product} sale",
    "{brand} {product} deals",
    "{brand} {product} vs",
    "{product} {brand}",
    "{brand} {product} {mod} {mod2}",
    "{brand} {product} free shipping",
    "new {brand} {product} {mod}",
    "{brand} {product} warranty",
    "{product} for {mod}",
    "best {product} {mod}",
    "{product} {mod} deals",
]

# Generic queries (no brand)
GENERIC_TEMPLATES = [
    "best {product} {mod}",
    "{product} for {mod}",
    "cheap {product}",
    "{product} under 100",
    "{product} deals",
    "top rated {product}",
    "{product} review",
    "{product} vs {product2}",
    "best {product} for gaming",
    "{product} {mod} sale",
    "buy {product} online",
]


# ═══════════════════════════════════════════════════════════════
# TYPO GENERATORS — Realistic Human Error Simulation
# ═══════════════════════════════════════════════════════════════

def typo_keyboard(word: str) -> Optional[str]:
    """Replace one char with QWERTY neighbor."""
    if len(word) < 2:
        return None
    i = random.randint(0, len(word) - 1)
    ch = word[i].lower()
    if ch in QWERTY:
        neighbors = QWERTY[ch]
        replacement = random.choice(neighbors)
        if replacement.isdigit() and word.isalpha():
            # Don't inject digits into pure-alpha words (looks unnatural)
            neighbors_alpha = [c for c in neighbors if c.isalpha()]
            if neighbors_alpha:
                replacement = random.choice(neighbors_alpha)
            else:
                return None
        return word[:i] + replacement + word[i+1:]
    return None


def typo_delete(word: str) -> Optional[str]:
    """Delete one character (prefer non-first, non-last)."""
    if len(word) < 4:
        return None
    # Prefer deleting from middle (more realistic)
    i = random.randint(1, len(word) - 2)
    return word[:i] + word[i+1:]


def typo_swap(word: str) -> Optional[str]:
    """Swap two adjacent characters."""
    if len(word) < 3:
        return None
    i = random.randint(0, len(word) - 2)
    return word[:i] + word[i+1] + word[i] + word[i+2:]


def typo_insert(word: str) -> Optional[str]:
    """Insert a random lowercase letter."""
    if len(word) < 2:
        return None
    i = random.randint(0, len(word))
    ch = random.choice(string.ascii_lowercase)
    return word[:i] + ch + word[i:]


def typo_double(word: str) -> Optional[str]:
    """Double a character (common real typo: 'keyboard' -> 'keeyboard')."""
    if len(word) < 3:
        return None
    i = random.randint(0, len(word) - 1)
    return word[:i] + word[i] + word[i:]


def typo_phonetic(word: str) -> Optional[str]:
    """
    Apply phonetic confusion: replace letter/digraph with a phonetically
    similar alternative. This simulates real human errors that come from
    how words *sound*, not keyboard proximity.
    """
    if len(word) < 3:
        return None

    wl = word.lower()

    # Try digraph replacements first (higher priority)
    digraphs = ['ph', 'th', 'ck', 'ee', 'ea', 'ou', 'oo', 'ie', 'ei']
    random.shuffle(digraphs)
    for dg in digraphs:
        if dg in wl and dg in PHONETIC_CONFUSIONS:
            replacement = random.choice(PHONETIC_CONFUSIONS[dg])
            idx = wl.find(dg)
            return word[:idx] + replacement + word[idx + len(dg):]

    # Single char phonetic confusion
    indices = list(range(len(wl)))
    random.shuffle(indices)
    for i in indices:
        ch = wl[i]
        if ch in PHONETIC_CONFUSIONS:
            replacement = random.choice(PHONETIC_CONFUSIONS[ch])
            if len(replacement) == 1:
                return word[:i] + replacement + word[i+1:]
            else:
                # Multi-char replacement (e.g., 'f' -> 'ph')
                return word[:i] + replacement + word[i+1:]
    return None


def typo_omit_repeated(word: str) -> Optional[str]:
    """Remove one char from a repeated pair: 'bluetooth' -> 'bluetoth'."""
    if len(word) < 4:
        return None
    for i in range(len(word) - 1):
        if word[i] == word[i+1]:
            return word[:i] + word[i+1:]
    return None


def typo_space_error(query: str) -> Optional[str]:
    """
    Introduce space errors: merge two words or split one.
    'air pods' -> 'airpods' or 'airpods' -> 'air pods'
    """
    words = query.split()
    if len(words) < 2:
        return None

    r = random.random()
    if r < 0.5 and len(words) >= 2:
        # Merge two adjacent words
        i = random.randint(0, len(words) - 2)
        merged = words[i] + words[i+1]
        return " ".join(words[:i] + [merged] + words[i+2:])
    else:
        # Split a word (only if word is long enough)
        long_words = [(i, w) for i, w in enumerate(words) if len(w) >= 6]
        if not long_words:
            return None
        i, w = random.choice(long_words)
        split_pos = random.randint(2, len(w) - 2)
        return " ".join(words[:i] + [w[:split_pos], w[split_pos:]] + words[i+1:])


def generate_single_typo(word: str) -> Optional[str]:
    """Generate one realistic typo for a word using weighted random selection."""
    methods = [
        (typo_keyboard, 25),
        (typo_delete, 15),
        (typo_swap, 15),
        (typo_phonetic, 20),
        (typo_double, 8),
        (typo_insert, 7),
        (typo_omit_repeated, 10),
    ]
    fns, weights = zip(*methods)
    order = list(range(len(fns)))
    random.shuffle(order)

    # Try weighted random, then fallback to any
    chosen = random.choices(order, weights=[weights[i] for i in order], k=len(order))
    for idx in chosen:
        result = fns[idx](word)
        if result and result != word and len(result) >= 2:
            return result
    return None


def generate_compound_typo(word: str) -> Optional[str]:
    """Apply 2 typo operations to simulate heavily misspelled words."""
    if len(word) < 4:
        return None
    first = generate_single_typo(word)
    if not first or first == word:
        return None
    second = generate_single_typo(first)
    if not second or second == first:
        return first  # at least return single typo
    return second


def make_query_typo(query: str, n_errors: int = 1) -> Optional[str]:
    """
    Introduce typos into a multi-word query.
    n_errors: how many words to corrupt (1, 2, or 3)
    """
    words = query.split()
    eligible = [(i, w) for i, w in enumerate(words)
                if len(w) >= 3 and w.isalpha()]
    if not eligible:
        return None

    n_corrupt = min(n_errors, len(eligible))
    targets = random.sample(eligible, n_corrupt)

    for idx, w in targets:
        r = random.random()
        if r < 0.25:
            typo = generate_compound_typo(w)
        else:
            typo = generate_single_typo(w)
        if typo:
            words[idx] = typo

    result = " ".join(words)
    return result if result != query else None


# ═══════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════

def load_text_file(path: Path) -> List[str]:
    """Load a text file, skip comments and blanks."""
    terms = []
    if not path.exists():
        print(f"  [WARN] {path.name} not found")
        return terms
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("="):
                continue
            terms.append(line.lower())
    return terms


def load_typo_mappings(path: Path) -> List[Tuple[str, str]]:
    """Load typo,correct pairs."""
    mappings = []
    if not path.exists():
        return mappings
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("="):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                typo, correct = parts[0].strip().lower(), parts[1].strip().lower()
                if typo and correct and typo != correct:
                    mappings.append((typo, correct))
    return mappings


def load_csv_pairs(path: Path) -> List[Tuple[str, str]]:
    """Load noisy,clean pairs from CSV."""
    pairs = []
    if not path.exists():
        return pairs
    with open(path, encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                if "noisy" in line.lower() or "clean" in line.lower():
                    continue
            line = line.strip().strip('"')
            parts = line.split('","')
            if len(parts) == 2:
                noisy = parts[0].strip('"').lower()
                clean = parts[1].strip('"').lower()
                if noisy and clean:
                    pairs.append((noisy, clean))
            else:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    noisy = parts[0].strip().strip('"').lower()
                    clean = parts[1].strip().strip('"').lower()
                    if noisy and clean:
                        pairs.append((noisy, clean))
    return pairs


# ═══════════════════════════════════════════════════════════════
# EXAMPLE GENERATORS
# ═══════════════════════════════════════════════════════════════

def make_example(input_query: str, target_query: str, category: str) -> dict:
    return {
        "input_text": f"correct: {input_query}",
        "target_text": target_query,
        "category": category,
    }


def gen_curated_typo_examples(
    typo_mappings: List[Tuple[str, str]],
    seen: Set[Tuple[str, str]],
) -> List[dict]:
    """
    Use curated typo->correct mappings in realistic query contexts.
    Each mapping generates: standalone + multiple in-context queries.
    """
    examples = []
    correct_to_typos = defaultdict(list)
    for typo, correct in typo_mappings:
        correct_to_typos[correct].append(typo)

    for correct, typos in correct_to_typos.items():
        for typo in typos:
            # Standalone
            pair = (typo, correct)
            if pair not in seen:
                examples.append(make_example(typo, correct, "curated_standalone"))
                seen.add(pair)

            # In brand context
            if correct in BRAND_CATALOG:
                products = BRAND_CATALOG[correct]
                for product in random.sample(products, min(4, len(products))):
                    for template in random.sample(QUERY_TEMPLATES, min(3, len(QUERY_TEMPLATES))):
                        mod = random.choice(MODIFIERS)
                        mod2 = random.choice(MODIFIERS)
                        try:
                            cq = template.format(brand=correct, product=product, mod=mod, mod2=mod2)
                            tq = cq.replace(correct, typo, 1)
                        except (KeyError, IndexError):
                            continue
                        if tq != cq:
                            pair = (tq, cq)
                            if pair not in seen:
                                examples.append(make_example(tq, cq, "curated_in_context"))
                                seen.add(pair)

    return examples


def gen_brand_product_typos(
    seen: Set[Tuple[str, str]],
    count: int = 80000,
) -> List[dict]:
    """
    Generate realistic e-commerce queries with typos.
    Covers all categories in BRAND_CATALOG.
    """
    examples = []
    brands = list(BRAND_CATALOG.keys())

    for _ in range(count * 4):  # oversample then trim
        brand = random.choice(brands)
        products = BRAND_CATALOG[brand]
        product = random.choice(products)
        template = random.choice(QUERY_TEMPLATES)
        mod = random.choice(MODIFIERS)
        mod2 = random.choice(MODIFIERS)

        try:
            correct_query = template.format(
                brand=brand, product=product, mod=mod, mod2=mod2
            )
        except (KeyError, IndexError):
            continue

        # Decide typo strategy
        r = random.random()
        if r < 0.35:
            # Typo on brand name
            brand_typo = generate_single_typo(brand)
            if not brand_typo or brand_typo == brand:
                continue
            typo_query = correct_query.replace(brand, brand_typo, 1)
        elif r < 0.55:
            # Typo on product term
            product_words = product.split()
            target_word = max(product_words, key=len)  # corrupt longest word
            if len(target_word) < 3:
                continue
            word_typo = generate_single_typo(target_word)
            if not word_typo or word_typo == target_word:
                continue
            typo_query = correct_query.replace(target_word, word_typo, 1)
        elif r < 0.75:
            # 1 random word typo in the full query
            typo_query = make_query_typo(correct_query, n_errors=1)
            if not typo_query:
                continue
        elif r < 0.90:
            # 2 word typos (compound query errors)
            typo_query = make_query_typo(correct_query, n_errors=2)
            if not typo_query:
                continue
        else:
            # Compound typo on brand (severely misspelled)
            brand_typo = generate_compound_typo(brand)
            if not brand_typo or brand_typo == brand:
                continue
            typo_query = correct_query.replace(brand, brand_typo, 1)

        if typo_query == correct_query:
            continue

        pair = (typo_query, correct_query)
        if pair not in seen:
            examples.append(make_example(typo_query, correct_query, "ecom_query_typo"))
            seen.add(pair)

        if len(examples) >= count:
            break

    return examples


def gen_generic_product_typos(
    seen: Set[Tuple[str, str]],
    count: int = 30000,
) -> List[dict]:
    """Generate typos for generic (non-branded) product queries."""
    examples = []

    for _ in range(count * 4):
        template = random.choice(GENERIC_TEMPLATES)
        product = random.choice(PRODUCT_TERMS)
        product2 = random.choice(PRODUCT_TERMS)
        mod = random.choice(MODIFIERS)

        try:
            correct_query = template.format(product=product, product2=product2, mod=mod)
        except (KeyError, IndexError):
            continue

        typo_query = make_query_typo(correct_query, n_errors=random.choices([1, 2], weights=[3, 1])[0])
        if not typo_query or typo_query == correct_query:
            continue

        pair = (typo_query, correct_query)
        if pair not in seen:
            examples.append(make_example(typo_query, correct_query, "generic_product_typo"))
            seen.add(pair)

        if len(examples) >= count:
            break

    return examples


def gen_space_error_examples(
    seen: Set[Tuple[str, str]],
    count: int = 15000,
) -> List[dict]:
    """Generate space-related errors: merged words, split words."""
    examples = []

    # Known merge/split pairs in e-commerce
    KNOWN_SPACE_PAIRS = [
        ("air pods", "airpods"), ("air pod", "airpod"),
        ("play station", "playstation"), ("game pad", "gamepad"),
        ("head phones", "headphones"), ("head set", "headset"),
        ("ear buds", "earbuds"), ("ear phones", "earphones"),
        ("lap top", "laptop"), ("note book", "notebook"),
        ("key board", "keyboard"), ("mouse pad", "mousepad"),
        ("back pack", "backpack"), ("hand bag", "handbag"),
        ("sun glasses", "sunglasses"), ("sun screen", "sunscreen"),
        ("tooth brush", "toothbrush"), ("tooth paste", "toothpaste"),
        ("smart watch", "smartwatch"), ("smart phone", "smartphone"),
        ("bed room", "bedroom"), ("bath room", "bathroom"),
        ("book shelf", "bookshelf"), ("door bell", "doorbell"),
        ("blue tooth", "bluetooth"), ("wi fi", "wifi"),
        ("micro phone", "microphone"), ("web cam", "webcam"),
        ("flash drive", "flashdrive"), ("hard drive", "harddrive"),
        ("foot wear", "footwear"), ("swim wear", "swimwear"),
        ("under wear", "underwear"), ("work out", "workout"),
        ("over ear", "overear"), ("in ear", "inear"),
    ]

    # Both directions: split->correct and correct->split
    for split_form, merged_form in KNOWN_SPACE_PAIRS:
        # split is typo, merged is correct
        pair = (split_form, merged_form)
        if pair not in seen:
            examples.append(make_example(split_form, merged_form, "space_split_error"))
            seen.add(pair)

        # merged is typo, split is correct (for words that should be separate)
        # Only for cases where the correct form IS the split form
        # e.g., "air force" should stay "air force", not become "airforce"

    # Generate random space errors from brand queries
    brands = list(BRAND_CATALOG.keys())
    for _ in range(count * 3):
        brand = random.choice(brands)
        product = random.choice(BRAND_CATALOG[brand])
        correct_query = f"{brand} {product}"

        typo_query = typo_space_error(correct_query)
        if not typo_query or typo_query == correct_query:
            continue

        pair = (typo_query, correct_query)
        if pair not in seen:
            examples.append(make_example(typo_query, correct_query, "space_error"))
            seen.add(pair)

        if len(examples) >= count:
            break

    return examples


def gen_phonetic_typo_examples(
    seen: Set[Tuple[str, str]],
    count: int = 25000,
) -> List[dict]:
    """
    Generate purely phonetic typo examples.
    These are the ones v2.1 completely missed.
    """
    examples = []
    brands = list(BRAND_CATALOG.keys())

    for _ in range(count * 4):
        brand = random.choice(brands)
        product = random.choice(BRAND_CATALOG[brand])
        mod = random.choice(MODIFIERS)
        correct_query = f"{brand} {product} {mod}"

        # Apply phonetic typo to brand
        words = correct_query.split()
        eligible = [(i, w) for i, w in enumerate(words)
                    if len(w) >= 4 and w.isalpha()]
        if not eligible:
            continue

        idx, target = random.choice(eligible)
        phonetic = typo_phonetic(target)
        if not phonetic or phonetic == target:
            continue

        words[idx] = phonetic
        typo_query = " ".join(words)

        pair = (typo_query, correct_query)
        if pair not in seen:
            examples.append(make_example(typo_query, correct_query, "phonetic_typo"))
            seen.add(pair)

        if len(examples) >= count:
            break

    return examples


def gen_identity_examples(
    seen: Set[Tuple[str, str]],
    count: int = 50000,
) -> List[dict]:
    """
    Generate identity pairs (correct -> correct) to prevent overcorrection.
    Target: ~20-25% of total dataset.

    Key: include tricky cases that LOOK like typos but aren't:
    - brand names (razer, xerox, asus)
    - abbreviations (gpu, ssd, fps)
    - model numbers (rtx 4090, rx 7900)
    - proper nouns with unusual spelling
    """
    examples = []
    brands = list(BRAND_CATALOG.keys())

    # 1. Brand names as-is (these must NOT be changed)
    for brand in brands:
        pair = (brand, brand)
        if pair not in seen:
            examples.append(make_example(brand, brand, "identity_brand"))
            seen.add(pair)

    # 2. Tricky brand names that look like misspellings
    TRICKY_BRANDS = [
        "razer", "xerox", "asus", "acer", "oppo", "vivo", "poco",
        "roku", "sonos", "bose", "nzxt", "evga", "zotac", "adata",
        "skullcandy", "sennheiser", "breville", "cuisinart", "miele",
        "birkenstock", "lululemon", "patagonia", "timberland",
    ]
    for brand in TRICKY_BRANDS:
        pair = (brand, brand)
        if pair not in seen:
            examples.append(make_example(brand, brand, "identity_tricky"))
            seen.add(pair)

    # 3. Full queries (brand + product) identity
    for _ in range(count):
        brand = random.choice(brands)
        product = random.choice(BRAND_CATALOG[brand])
        template = random.choice(QUERY_TEMPLATES[:8])  # simpler templates
        mod = random.choice(MODIFIERS)
        mod2 = random.choice(MODIFIERS)
        try:
            query = template.format(brand=brand, product=product, mod=mod, mod2=mod2)
        except (KeyError, IndexError):
            continue

        pair = (query, query)
        if pair not in seen:
            examples.append(make_example(query, query, "identity_query"))
            seen.add(pair)

    # 4. Generic product queries identity
    for _ in range(count // 4):
        product = random.choice(PRODUCT_TERMS)
        mod = random.choice(MODIFIERS)
        query = f"{product} {mod}"

        pair = (query, query)
        if pair not in seen:
            examples.append(make_example(query, query, "identity_generic"))
            seen.add(pair)

    # 5. Technical abbreviations / model numbers identity
    TECH_TERMS = [
        "gpu", "cpu", "ssd", "hdd", "ram", "rgb", "led", "lcd", "oled",
        "qled", "usb", "hdmi", "wifi", "nfc", "5g", "lte", "fps", "uhd",
        "4k", "8k", "1080p", "1440p", "ddr4", "ddr5", "pcie", "nvme",
        "rtx 4090", "rtx 4080", "rtx 4070", "rtx 4060",
        "rx 7900 xtx", "rx 7800 xt", "ryzen 9 7950x",
        "core i9 14900k", "core i7 14700k", "m3 pro", "m3 max",
        "iphone 15 pro max", "galaxy s24 ultra", "pixel 8 pro",
    ]
    for term in TECH_TERMS:
        pair = (term, term)
        if pair not in seen:
            examples.append(make_example(term, term, "identity_tech"))
            seen.add(pair)

    return examples


def gen_existing_data_examples(
    typo_mappings: List[Tuple[str, str]],
    csv_pairs: List[Tuple[str, str]],
    seen: Set[Tuple[str, str]],
) -> List[dict]:
    """Incorporate existing curated data (typo_dataset.csv, etc.)."""
    examples = []

    for noisy, clean in csv_pairs:
        if noisy == clean:
            continue
        pair = (noisy, clean)
        if pair not in seen:
            examples.append(make_example(noisy, clean, "existing_csv"))
            seen.add(pair)

    return examples


def gen_vocab_typo_examples(
    vocab_terms: List[str],
    seen: Set[Tuple[str, str]],
    count: int = 20000,
) -> List[dict]:
    """Generate typos for standalone vocabulary terms."""
    examples = []
    terms = [t for t in vocab_terms if len(t) >= 4 and t.isalpha()]
    if not terms:
        return examples

    for _ in range(count * 3):
        term = random.choice(terms)
        r = random.random()
        if r < 0.3:
            typo = typo_phonetic(term)
        elif r < 0.6:
            typo = generate_single_typo(term)
        else:
            typo = generate_compound_typo(term) if len(term) >= 5 else generate_single_typo(term)

        if typo and typo != term:
            pair = (typo, term)
            if pair not in seen:
                examples.append(make_example(typo, term, "vocab_typo"))
                seen.add(pair)

        if len(examples) >= count:
            break

    return examples


# ═══════════════════════════════════════════════════════════════
# QUALITY FILTERS
# ═══════════════════════════════════════════════════════════════

def quality_filter(examples: List[dict]) -> List[dict]:
    """Remove low-quality or degenerate examples."""
    filtered = []
    for ex in examples:
        inp = ex["input_text"].replace("correct: ", "", 1)
        tgt = ex["target_text"]

        # Skip empty
        if not inp.strip() or not tgt.strip():
            continue

        # Skip if input is too long (>15 words — unrealistic queries)
        if len(inp.split()) > 15:
            continue

        # Skip if target is too long
        if len(tgt.split()) > 15:
            continue

        # Skip if input and target only differ by case
        if inp.lower() == tgt.lower() and inp != tgt:
            continue

        # Skip if edit distance is too high (>50% of word length — probably garbage)
        if inp != tgt:
            shorter = min(len(inp), len(tgt))
            if shorter > 0:
                # Simple char-level difference check
                diff = sum(1 for a, b in zip(inp, tgt) if a != b) + abs(len(inp) - len(tgt))
                if diff > shorter * 0.6:
                    continue

        filtered.append(ex)

    return filtered


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build T5-Large v3 training data")
    parser.add_argument("--target", type=int, default=400000,
                        help="Target total examples (default: 400000)")
    parser.add_argument("--identity-ratio", type=float, default=0.22,
                        help="Target identity pair ratio (default: 0.22)")
    parser.add_argument("--eval-ratio", type=float, default=0.05,
                        help="Eval split ratio (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  T5-Large v3 — Training Data Builder")
    print("  Target: {:,} examples | Identity: {:.0%} | Eval: {:.0%}".format(
        args.target, args.identity_ratio, args.eval_ratio))
    print("=" * 70)

    # ── Load existing data ──
    print("\n[1/9] Loading existing data files...")
    typo_mappings = load_typo_mappings(DATA_DIR / "typo_mappings.txt")
    brand_products = load_text_file(DATA_DIR / "brand_products.txt")
    electronics_vocab = load_text_file(DATA_DIR / "electronics_vocab.txt")
    domain_vocab = load_text_file(DATA_DIR / "domain_vocab.txt")
    ecom_corpus = load_text_file(DATA_DIR / "ecommerce_corpus.txt")
    csv_pairs = load_csv_pairs(DATA_DIR / "typo_dataset.csv")

    all_vocab = list(set(brand_products + electronics_vocab + domain_vocab + ecom_corpus))
    print(f"  Typo mappings:    {len(typo_mappings):>6,} pairs")
    print(f"  Brand products:   {len(brand_products):>6,} terms")
    print(f"  Electronics:      {len(electronics_vocab):>6,} terms")
    print(f"  Domain vocab:     {len(domain_vocab):>6,} terms")
    print(f"  E-com corpus:     {len(ecom_corpus):>6,} terms")
    print(f"  CSV pairs:        {len(csv_pairs):>6,} pairs")
    print(f"  Total vocab:      {len(all_vocab):>6,} unique terms")
    print(f"  Brand catalog:    {len(BRAND_CATALOG):>6,} brands")

    seen: Set[Tuple[str, str]] = set()
    all_examples: List[dict] = []

    # ── Curated typo mappings ──
    print("\n[2/9] Curated typo mappings in context...")
    curated = gen_curated_typo_examples(typo_mappings, seen)
    all_examples.extend(curated)
    print(f"  → {len(curated):,} examples")

    # ── Existing CSV pairs ──
    print("\n[3/9] Existing CSV data...")
    csv_ex = gen_existing_data_examples(typo_mappings, csv_pairs, seen)
    all_examples.extend(csv_ex)
    print(f"  → {len(csv_ex):,} examples")

    # ── Brand/product query typos ──
    print("\n[4/9] Brand+product query typos (all categories)...")
    brand_typos = gen_brand_product_typos(seen, count=120000)
    all_examples.extend(brand_typos)
    print(f"  → {len(brand_typos):,} examples")

    # ── Generic product typos ──
    print("\n[5/9] Generic product query typos...")
    generic = gen_generic_product_typos(seen, count=40000)
    all_examples.extend(generic)
    print(f"  → {len(generic):,} examples")

    # ── Phonetic typos ──
    print("\n[6/9] Phonetic confusion typos...")
    phonetic = gen_phonetic_typo_examples(seen, count=35000)
    all_examples.extend(phonetic)
    print(f"  → {len(phonetic):,} examples")

    # ── Space errors ──
    print("\n[7/9] Space merge/split errors...")
    space = gen_space_error_examples(seen, count=15000)
    all_examples.extend(space)
    print(f"  → {len(space):,} examples")

    # ── Vocabulary typos ──
    print("\n[8/9] Standalone vocabulary typos...")
    vocab = gen_vocab_typo_examples(all_vocab, seen, count=25000)
    all_examples.extend(vocab)
    print(f"  → {len(vocab):,} examples")

    # ── Identity pairs ──
    target_identity = int(args.target * args.identity_ratio)
    print(f"\n[9/9] Identity pairs (target: ~{target_identity:,})...")
    identity = gen_identity_examples(seen, count=target_identity)
    all_examples.extend(identity)
    print(f"  → {len(identity):,} examples")

    # ── Quality filter ──
    print(f"\n  Pre-filter:  {len(all_examples):,}")
    all_examples = quality_filter(all_examples)
    print(f"  Post-filter: {len(all_examples):,}")

    # ── Balance: trim to target ──
    random.shuffle(all_examples)

    # Calculate current identity ratio
    id_count = sum(1 for ex in all_examples
                   if ex["input_text"].replace("correct: ", "", 1) == ex["target_text"])
    corr_count = len(all_examples) - id_count
    current_ratio = id_count / len(all_examples) if all_examples else 0

    print(f"\n  Current: {len(all_examples):,} total, "
          f"{id_count:,} identity ({current_ratio:.1%}), "
          f"{corr_count:,} corrections")

    # Trim identity pairs if ratio is too high
    if current_ratio > args.identity_ratio + 0.03:
        target_id = int(corr_count * args.identity_ratio / (1 - args.identity_ratio))
        excess = id_count - target_id
        if excess > 0:
            id_indices = [i for i, ex in enumerate(all_examples)
                         if ex["input_text"].replace("correct: ", "", 1) == ex["target_text"]]
            remove = set(random.sample(id_indices, min(excess, len(id_indices))))
            all_examples = [ex for i, ex in enumerate(all_examples) if i not in remove]
            print(f"  Trimmed {len(remove):,} identity pairs")

    # Trim total if over target
    if len(all_examples) > args.target:
        all_examples = all_examples[:args.target]
        print(f"  Trimmed to target: {args.target:,}")

    # ── Split ──
    random.shuffle(all_examples)
    eval_size = int(len(all_examples) * args.eval_ratio)
    eval_data = all_examples[:eval_size]
    train_data = all_examples[eval_size:]

    # ── Final stats ──
    total = len(train_data)
    id_final = sum(1 for ex in train_data
                   if ex["input_text"].replace("correct: ", "", 1) == ex["target_text"])
    corr_final = total - id_final
    cats = Counter(ex["category"] for ex in train_data)

    print(f"\n{'=' * 70}")
    print(f"  FINAL DATASET")
    print(f"{'=' * 70}")
    print(f"  Train: {total:,}")
    print(f"    Identity:   {id_final:,} ({100 * id_final / total:.1f}%)")
    print(f"    Correction: {corr_final:,} ({100 * corr_final / total:.1f}%)")
    print(f"  Eval:  {len(eval_data):,}")
    print(f"\n  Categories:")
    for cat, cnt in cats.most_common():
        pct = 100 * cnt / total
        bar = "█" * int(pct / 2)
        print(f"    {cat:<25s} {cnt:>7,} ({pct:5.1f}%) {bar}")

    # ── Sample examples ──
    print(f"\n  --- Sample corrections ---")
    corrections = [ex for ex in train_data
                   if ex["input_text"].replace("correct: ", "", 1) != ex["target_text"]]
    for ex in random.sample(corrections, min(30, len(corrections))):
        q = ex["input_text"].replace("correct: ", "")
        t = ex["target_text"]
        print(f"    [{ex['category']:<25s}] '{q}' → '{t}'")

    # ── Save ──
    out_train = OUT_DIR / "train_v3.jsonl"
    out_eval = OUT_DIR / "eval_v3.jsonl"

    with open(out_train, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(out_eval, "w", encoding="utf-8") as f:
        for ex in eval_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ── Stats file ──
    stats = {
        "total_train": total,
        "total_eval": len(eval_data),
        "identity_count": id_final,
        "identity_ratio": round(id_final / total, 4),
        "correction_count": corr_final,
        "categories": {k: v for k, v in cats.most_common()},
        "brand_count": len(BRAND_CATALOG),
        "product_terms_count": len(PRODUCT_TERMS),
        "typo_mappings_count": len(typo_mappings),
    }
    with open(OUT_DIR / "training_stats_v3.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Saved:")
    print(f"    {out_train}")
    print(f"    {out_eval}")
    print(f"    {OUT_DIR / 'training_stats_v3.json'}")
    print(f"\n  Upload data/ folder to Google Drive for Colab training.")


if __name__ == "__main__":
    main()
