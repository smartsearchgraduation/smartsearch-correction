#!/usr/bin/env python3
"""
T5-Large / ByT5-Large v3.1 Training Data Builder
=================================================
Builds high-quality English e-commerce spelling correction training data.

v3.1 improvements over v3.0:
  - NEW: gen_price_examples (teaches price formats: $99, $1,299.99, under $500, 299 USD)
  - NEW: gen_measurement_unit_examples (GB/TB/MB, mm/cm/inch, W/V/mAh, Hz/dpi/fps, 1080p/4K)
  - NEW: gen_brand_category_mismatch_examples (e.g. "nvidia tuf chair" -> context correction)
  - NEW: gen_everyday_english_typos (realistic misspellings of frequent English words)
  - NEW: gen_aug_* generators (aug_keyboard/double/swap/phonetic/delete/insert/compound/omit_repeat)
  - NEW: gen_external_word_typos + gen_external_context_typos (Birkbeck / codespell / torinriley)
          -> with DOMAIN FILTER so out-of-domain words like "tuatera" are dropped
  - NEW: gen_identity_external (preserve correct words that LOOK like typos)
  - FIX: Levenshtein-based quality filter (replaces broken positional diff)
  - FIX: Removed trailing "{brand} {product} vs" template (generated dangling "vs" queries)
  - FIX: Removed reflexive aug_compound pairing that duplicated word pairs
  - BRAND_CATEGORIES: each brand tagged with category so mismatch generator is possible
  - Boosted identity_brand / identity_tech / identity_tricky (5-10x) to reduce
    over-correction of real brand names
  - English only (no Turkish artifacts)

Output: train_v3.jsonl, eval_v3.jsonl, training_stats_v3.json (in same data/ folder)
Format: {"input_text": "correct: corsiar keybord", "target_text": "corsair keyboard", "category": "..."}

Usage:
  python build_training_data_v3.py
  python build_training_data_v3.py --target 950000
  python build_training_data_v3.py --target 400000 --identity-ratio 0.22
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

SCRIPT_DIR = Path(__file__).parent                    # .../BYT5-T5 Large v3/data/
V3_ROOT    = SCRIPT_DIR.parent                         # .../BYT5-T5 Large v3/
CORRECTION_DIR = V3_ROOT.parent.parent                 # .../Correction/
DATA_DIR = CORRECTION_DIR / "data"                     # .../Correction/data/ (source vocab)
OUT_DIR  = SCRIPT_DIR                                  # output JSONL to same data/ folder

# ═══════════════════════════════════════════════════════════════
# LEVENSHTEIN (small helper used by the quality filter)
# ═══════════════════════════════════════════════════════════════

def levenshtein(a: str, b: str) -> int:
    """Compute true Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dlt = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur[j] = min(ins, dlt, sub)
        prev = cur
    return prev[-1]


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

PHONETIC_CONFUSIONS = {
    'a': ['e', 'u'], 'e': ['a', 'i'], 'i': ['e', 'y'],
    'o': ['u', 'a'], 'u': ['o', 'a'],
    'c': ['k', 's'], 'k': ['c'], 's': ['z', 'c'], 'z': ['s'],
    'f': ['ph', 'v'], 'ph': ['f'], 'th': ['t'], 'ck': ['k', 'c'],
    'ee': ['ea', 'ie'], 'ea': ['ee'], 'ou': ['ow', 'u'],
    'oo': ['u'], 'ie': ['ei'], 'ei': ['ie'],
}

OMISSION_PATTERNS = {
    'tion': ['tion', 'toin', 'tin', 'shon'],
    'ing': ['ing', 'in', 'ig'],
    'ment': ['ment', 'mnt'],
    'ness': ['ness', 'nss', 'nes'],
    'able': ['able', 'abl', 'ble'],
    'ible': ['ible', 'ibl', 'ble'],
    'ght': ['ght', 'gt', 'ht'],
    'ough': ['ough', 'uff', 'off'],
}


# ═══════════════════════════════════════════════════════════════
# E-COMMERCE VOCABULARY & CATEGORY MAPPING
# ═══════════════════════════════════════════════════════════════

# brand -> list of products it actually sells (used to generate in-context queries).
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

# Brand category tags (used by the brand-category mismatch generator).
BRAND_CATEGORIES = {
    # Electronics
    **{b: "electronics" for b in [
        "apple", "samsung", "sony", "nvidia", "amd", "intel", "asus", "dell", "hp",
        "lenovo", "acer", "msi", "razer", "logitech", "corsair", "steelseries",
        "hyperx", "kingston", "seagate", "sandisk", "crucial", "western digital",
        "anker", "bose", "sennheiser", "jbl", "lg", "microsoft", "google",
        "oneplus", "xiaomi", "marshall", "canon", "nikon", "gopro", "dji",
        "nintendo", "tp-link", "netgear",
    ]},
    # Appliances
    "dyson": "appliances",
    # Fashion
    **{b: "fashion" for b in [
        "nike", "adidas", "puma", "new balance", "converse", "vans", "reebok",
        "under armour", "gucci", "louis vuitton", "zara", "h&m", "uniqlo",
        "levi's", "ralph lauren", "tommy hilfiger", "calvin klein",
        "the north face", "patagonia", "columbia", "timberland", "dr. martens",
        "birkenstock", "crocs", "ray-ban", "oakley",
    ]},
    # Beauty & Personal Care
    **{b: "beauty" for b in [
        "maybelline", "l'oreal", "mac", "nyx", "clinique", "the ordinary",
        "cerave", "neutrogena", "olaplex", "dove", "old spice", "gillette",
        "oral-b", "philips", "braun",
    ]},
    # Home & Kitchen
    **{b: "home_kitchen" for b in [
        "ikea", "kitchenaid", "instant pot", "ninja", "keurig", "nespresso",
        "roomba", "cuisinart", "breville",
    ]},
    # Sports & Outdoors
    **{b: "sports" for b in [
        "yeti", "hydroflask", "garmin", "fitbit", "peloton", "theragun", "wilson",
    ]},
    # Grocery
    **{b: "grocery" for b in [
        "nestle", "coca-cola", "pepsi", "kellogg's",
    ]},
    # Automotive
    **{b: "automotive" for b in [
        "bosch", "michelin", "castrol", "meguiar's",
    ]},
}

# Products grouped by category (helps build "plausibly intended" corrections).
CATEGORY_PRODUCTS = defaultdict(set)
for brand, products in BRAND_CATALOG.items():
    cat = BRAND_CATEGORIES.get(brand, "other")
    for p in products:
        CATEGORY_PRODUCTS[cat].add(p)
CATEGORY_PRODUCTS = {k: sorted(v) for k, v in CATEGORY_PRODUCTS.items()}

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

# NOTE v3.1: removed the dangling "{brand} {product} vs" template — it produced
# trailing "vs" queries with no RHS. Swapped for a concrete "vs" template.
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
    "{product} {brand}",
    "{brand} {product} {mod} {mod2}",
    "{brand} {product} free shipping",
    "new {brand} {product} {mod}",
    "{brand} {product} warranty",
    "{product} for {mod}",
    "best {product} {mod}",
    "{product} {mod} deals",
]

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
# v3.1 NEW: PRICE / CURRENCY TEMPLATES
# ═══════════════════════════════════════════════════════════════

CURRENCY_SYMBOLS = ["$", "£", "€"]
CURRENCY_CODES = ["USD", "GBP", "EUR", "CAD", "AUD"]

PRICE_TEMPLATES_CLEAN = [
    "{product} under ${p}",
    "{product} less than ${p}",
    "{product} below ${p}",
    "{product} for ${p}",
    "{brand} {product} under ${p}",
    "{brand} {product} for ${p}",
    "{brand} {product} around ${p}",
    "best {product} under ${p}",
    "cheap {product} under ${p}",
    "{product} between ${p} and ${p2}",
    "{brand} {product} between ${p} and ${p2}",
    "{product} ${p} to ${p2}",
    "{product} {price_code}",
    "{brand} {product} {price_code}",
    "{product} ${p}.{cents}",
    "{brand} {product} ${p}.{cents}",
    "{product} ${p_big}",
    "{brand} {product} ${p_big}",
    "{product} on sale {price_code}",
    "{product} deal ${p}",
]

# Price typo rules: $ -> "dollar"/"dollars"/"dolar" typo, spacing errors,
# comma/decimal confusion. These are generated programmatically.


# ═══════════════════════════════════════════════════════════════
# v3.1 NEW: MEASUREMENT UNITS / SPEC TEMPLATES
# ═══════════════════════════════════════════════════════════════

STORAGE_UNITS = ["mb", "gb", "tb"]
FREQUENCY_UNITS = ["hz", "ghz", "mhz"]
POWER_UNITS = ["w", "kw", "v", "mah", "wh"]
LENGTH_UNITS = ["mm", "cm", "m", "inch", '"', "ft"]
WEIGHT_UNITS = ["g", "kg", "lb", "lbs", "oz"]
CAMERA_UNITS = ["mp", "fps", "dpi"]
RESOLUTIONS = ["720p", "1080p", "1440p", "2k", "4k", "5k", "8k"]

UNIT_TEMPLATES_CLEAN = [
    "{product} {n} {unit}",
    "{brand} {product} {n} {unit}",
    "{product} {n}{unit}",
    "{brand} {product} {n}{unit}",
    "{n} {unit} {product}",
    "{n}{unit} {product}",
    "{brand} {product} with {n} {unit}",
    "{product} {res}",
    "{brand} {product} {res}",
    "{product} {n} {unit} {res}",
    "{brand} {product} {n} {unit} {mod}",
    "{product} {n} {unit} for {mod}",
]


# ═══════════════════════════════════════════════════════════════
# v3.1 NEW: BRAND-CATEGORY MISMATCH EXAMPLES
# Target: "nvidia tuf chair" -> "asus tuf chair" OR "nvidia gpu" via context.
# ═══════════════════════════════════════════════════════════════

# explicit "confusion" pairs between brand sub-lines that look alike
# Only used to generate realistic in-context mismatches.
CROSS_BRAND_MODEL_CONFUSIONS = [
    # nvidia is GPUs, TUF is asus gaming line
    ("nvidia", "asus", "tuf gaming"),
    ("nvidia", "asus", "rog"),
    # ryzen is amd, intel shouldn't have it
    ("intel", "amd", "ryzen 7"),
    ("intel", "amd", "radeon"),
    # geforce is nvidia
    ("amd", "nvidia", "geforce"),
    ("amd", "nvidia", "rtx"),
    # airpods is apple
    ("samsung", "apple", "airpods"),
    # galaxy is samsung
    ("apple", "samsung", "galaxy"),
    # xbox is microsoft; playstation is sony
    ("sony", "microsoft", "xbox"),
    ("microsoft", "sony", "playstation"),
    # alienware is dell; predator is acer
    ("hp", "dell", "alienware"),
    ("dell", "acer", "predator"),
    # thinkpad is lenovo
    ("hp", "lenovo", "thinkpad"),
    ("dell", "lenovo", "thinkpad"),
    # surface is microsoft
    ("apple", "microsoft", "surface"),
    # yeezy is adidas; jordan is nike
    ("nike", "adidas", "yeezy"),
    ("adidas", "nike", "jordan"),
    # kindle is amazon but we don't have amazon in brands; skip
    # nespresso is nespresso brand
    ("keurig", "nespresso", "vertuo"),
]


# ═══════════════════════════════════════════════════════════════
# v3.1 NEW: DOMAIN FILTER — drops corpus entries not plausibly e-commerce
# ═══════════════════════════════════════════════════════════════

DOMAIN_STOPWORDS = set("""
the a an and or but of in on at for to from by with as is are was were be been being
this that these those it its they them their there here where when what who how why
i you he she we me my your his her our us do does did done doing have has had having
not no yes so very just only also more most many much some any all each every both few
""".split())

# Words to drop outright — archaic, domain-irrelevant, or potentially unsafe
# (kept conservative; e-commerce search rarely contains these).
DOMAIN_BLOCKLIST = {
    # archaic / naturalist vocabulary common in Birkbeck
    "tuatera", "herbaceously", "antediluvian", "amanuensis", "petrichor",
    "obsequious", "sesquipedalian", "pulchritudinous", "ennui",
    # anatomical / medical-only rare words
    "vicissitude", "lugubrious", "perspicacious", "recalcitrant",
    # proper nouns that aren't e-com brands
    "aberdeen", "albania", "yorkshire",
}

# Allowed-character regex for domain-plausible tokens
_DOMAIN_ALLOWED = re.compile(r"^[a-z][a-z0-9'\-]{1,}$")


def domain_plausible(word: str) -> bool:
    """Return True if word is plausibly an e-commerce search token."""
    w = word.strip().lower()
    if not w:
        return False
    if w in DOMAIN_STOPWORDS or w in DOMAIN_BLOCKLIST:
        return False
    if not _DOMAIN_ALLOWED.match(w):
        return False
    if len(w) < 3 or len(w) > 18:
        return False
    # drop weird character runs
    if re.search(r"[aeiou]{4,}", w) or re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", w):
        return False
    return True


# ═══════════════════════════════════════════════════════════════
# TYPO GENERATORS — Realistic Human Error Simulation
# ═══════════════════════════════════════════════════════════════

def typo_keyboard(word: str) -> Optional[str]:
    if len(word) < 2:
        return None
    i = random.randint(0, len(word) - 1)
    ch = word[i].lower()
    if ch in QWERTY:
        neighbors = QWERTY[ch]
        replacement = random.choice(neighbors)
        if replacement.isdigit() and word.isalpha():
            neighbors_alpha = [c for c in neighbors if c.isalpha()]
            if neighbors_alpha:
                replacement = random.choice(neighbors_alpha)
            else:
                return None
        return word[:i] + replacement + word[i+1:]
    return None


def typo_delete(word: str) -> Optional[str]:
    if len(word) < 4:
        return None
    i = random.randint(1, len(word) - 2)
    return word[:i] + word[i+1:]


def typo_swap(word: str) -> Optional[str]:
    if len(word) < 3:
        return None
    i = random.randint(0, len(word) - 2)
    return word[:i] + word[i+1] + word[i] + word[i+2:]


def typo_insert(word: str) -> Optional[str]:
    if len(word) < 2:
        return None
    i = random.randint(0, len(word))
    ch = random.choice(string.ascii_lowercase)
    return word[:i] + ch + word[i:]


def typo_double(word: str) -> Optional[str]:
    if len(word) < 3:
        return None
    i = random.randint(0, len(word) - 1)
    return word[:i] + word[i] + word[i:]


def typo_phonetic(word: str) -> Optional[str]:
    if len(word) < 3:
        return None
    wl = word.lower()
    digraphs = ['ph', 'th', 'ck', 'ee', 'ea', 'ou', 'oo', 'ie', 'ei']
    random.shuffle(digraphs)
    for dg in digraphs:
        if dg in wl and dg in PHONETIC_CONFUSIONS:
            replacement = random.choice(PHONETIC_CONFUSIONS[dg])
            idx = wl.find(dg)
            return word[:idx] + replacement + word[idx + len(dg):]
    indices = list(range(len(wl)))
    random.shuffle(indices)
    for i in indices:
        ch = wl[i]
        if ch in PHONETIC_CONFUSIONS:
            replacement = random.choice(PHONETIC_CONFUSIONS[ch])
            return word[:i] + replacement + word[i+1:]
    return None


def typo_omit_repeated(word: str) -> Optional[str]:
    if len(word) < 4:
        return None
    for i in range(len(word) - 1):
        if word[i] == word[i+1]:
            return word[:i] + word[i+1:]
    return None


def typo_space_error(query: str) -> Optional[str]:
    words = query.split()
    if len(words) < 2:
        return None
    mode = random.random()
    if mode < 0.5:
        # merge two adjacent words
        idx = random.randint(0, len(words) - 2)
        words[idx] = words[idx] + words[idx + 1]
        del words[idx + 1]
    else:
        # split a long word
        long_words = [(i, w) for i, w in enumerate(words) if len(w) >= 6]
        if not long_words:
            return None
        idx, w = random.choice(long_words)
        cut = random.randint(2, len(w) - 2)
        words[idx] = w[:cut] + " " + w[cut:]
    return " ".join(words)


def generate_single_typo(word: str) -> Optional[str]:
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
    chosen = random.choices(order, weights=[weights[i] for i in order], k=len(order))
    for idx in chosen:
        result = fns[idx](word)
        if result and result != word and len(result) >= 2:
            return result
    return None


def generate_compound_typo(word: str) -> Optional[str]:
    if len(word) < 4:
        return None
    first = generate_single_typo(word)
    if not first or first == word:
        return None
    second = generate_single_typo(first)
    if not second or second == first:
        return first
    return second


def make_query_typo(query: str, n_errors: int = 1) -> Optional[str]:
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
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip().lower() for ln in f if ln.strip()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def load_typo_mappings(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        return []
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if "," in ln:
                parts = ln.split(",", 1)
                if len(parts) == 2:
                    pairs.append((parts[0].strip().lower(), parts[1].strip().lower()))
    return pairs


def load_csv_pairs(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        return []
    pairs = []
    try:
        import csv
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    a, b = row[0].strip().lower(), row[1].strip().lower()
                    if a and b:
                        pairs.append((a, b))
    except Exception:
        pass
    return pairs


def load_external_pairs(path: Path) -> List[Tuple[str, str]]:
    """
    Load external typo corpora in a variety of formats:
      - "misspelling,correct"
      - "misspelling correct"
      - "misspelling->correct"
      - Birkbeck style (multiple typos per correct word; '$correct' marker)
    Returns list of (typo, correct) pairs.
    """
    if not path.exists():
        return []
    pairs: List[Tuple[str, str]] = []
    current_correct: Optional[str] = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if ln.startswith("$"):
                current_correct = ln[1:].strip().lower()
                continue
            for sep in ("->", "=>", "\t", "|", ",", ":"):
                if sep in ln:
                    parts = ln.split(sep, 1)
                    if len(parts) == 2:
                        a, b = parts[0].strip().lower(), parts[1].strip().lower()
                        if a and b and a != b:
                            pairs.append((a, b))
                    break
            else:
                # whitespace-separated or just a misspelling under a $marker
                if current_correct is not None:
                    typo = ln.lower()
                    if typo and typo != current_correct:
                        pairs.append((typo, current_correct))
                else:
                    parts = ln.split()
                    if len(parts) == 2:
                        a, b = parts[0].lower(), parts[1].lower()
                        if a and b and a != b:
                            pairs.append((a, b))
    return pairs


# ═══════════════════════════════════════════════════════════════
# EXAMPLE BUILDER + HELPERS
# ═══════════════════════════════════════════════════════════════

def make_example(input_query: str, target_query: str, category: str) -> dict:
    return {
        "input_text": f"correct: {input_query}",
        "target_text": target_query,
        "category": category,
    }


def _format_template(template: str, **kwargs) -> Optional[str]:
    try:
        return template.format(**kwargs).strip()
    except (KeyError, IndexError):
        return None


# ═══════════════════════════════════════════════════════════════
# GENERATORS — identical names / outputs to v3.0 where possible
# ═══════════════════════════════════════════════════════════════

def gen_curated_typo_examples(
    typo_mappings: List[Tuple[str, str]],
    seen: Set[Tuple[str, str]],
) -> List[dict]:
    """Turn static typo→correct mappings into in-context training pairs."""
    examples: List[dict] = []
    brands = list(BRAND_CATALOG.keys())
    for typo, correct in typo_mappings:
        # standalone word
        pair = (typo, correct)
        if pair not in seen:
            examples.append(make_example(typo, correct, "vocab_typo"))
            seen.add(pair)
        # in context (brand + typo) if correct is a product term
        for brand in random.sample(brands, k=min(2, len(brands))):
            products = BRAND_CATALOG[brand]
            if correct in products:
                q_correct = f"{brand} {correct}"
                q_typo = f"{brand} {typo}"
                pp = (q_typo, q_correct)
                if pp not in seen:
                    examples.append(make_example(q_typo, q_correct, "ecom_query_typo"))
                    seen.add(pp)
    return examples


def gen_existing_data_examples(
    typo_mappings: List[Tuple[str, str]],
    csv_pairs: List[Tuple[str, str]],
    seen: Set[Tuple[str, str]],
) -> List[dict]:
    """Pull in the already-collected typo_dataset.csv pairs."""
    examples: List[dict] = []
    for typo, correct in csv_pairs:
        if not typo or not correct or typo == correct:
            continue
        if len(typo.split()) > 15 or len(correct.split()) > 15:
            continue
        pair = (typo, correct)
        if pair in seen:
            continue
        examples.append(make_example(typo, correct, "ecom_query_typo"))
        seen.add(pair)
    return examples


def gen_brand_product_typos(
    seen: Set[Tuple[str, str]],
    count: int = 120000,
) -> List[dict]:
    """Typos in brand+product queries across every catalog category."""
    examples: List[dict] = []
    brands = list(BRAND_CATALOG.keys())
    attempts = 0
    max_attempts = count * 4
    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        brand = random.choice(brands)
        product = random.choice(BRAND_CATALOG[brand])
        template = random.choice(QUERY_TEMPLATES)
        mod = random.choice(MODIFIERS)
        mod2 = random.choice(MODIFIERS)
        query = _format_template(template, brand=brand, product=product, mod=mod, mod2=mod2)
        if not query:
            continue
        n_errors = random.choices([1, 2, 3], weights=[65, 28, 7])[0]
        typo_query = make_query_typo(query, n_errors=n_errors)
        if not typo_query or typo_query == query:
            continue
        pair = (typo_query, query)
        if pair in seen:
            continue
        examples.append(make_example(typo_query, query, "ecom_query_typo"))
        seen.add(pair)
    return examples


def gen_generic_product_typos(
    seen: Set[Tuple[str, str]],
    count: int = 40000,
) -> List[dict]:
    examples: List[dict] = []
    attempts = 0
    while len(examples) < count and attempts < count * 4:
        attempts += 1
        template = random.choice(GENERIC_TEMPLATES)
        p1 = random.choice(PRODUCT_TERMS)
        p2 = random.choice(PRODUCT_TERMS)
        mod = random.choice(MODIFIERS)
        query = _format_template(template, product=p1, product2=p2, mod=mod)
        if not query:
            continue
        typo_query = make_query_typo(query, n_errors=random.choice([1, 2]))
        if not typo_query or typo_query == query:
            continue
        pair = (typo_query, query)
        if pair in seen:
            continue
        examples.append(make_example(typo_query, query, "generic_product_typo"))
        seen.add(pair)
    return examples


def gen_phonetic_typo_examples(
    seen: Set[Tuple[str, str]],
    count: int = 35000,
) -> List[dict]:
    """Specifically exercise phonetic confusion patterns."""
    examples: List[dict] = []
    brands = list(BRAND_CATALOG.keys())
    attempts = 0
    while len(examples) < count and attempts < count * 4:
        attempts += 1
        brand = random.choice(brands)
        product = random.choice(BRAND_CATALOG[brand])
        # only use phonetic corruption, on brand or product word(s)
        q = f"{brand} {product}"
        words = q.split()
        # prefer to corrupt the longest eligible word
        candidates = sorted([(i, w) for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()],
                            key=lambda t: -len(t[1]))
        if not candidates:
            continue
        idx, w = candidates[0]
        typo = typo_phonetic(w)
        if not typo or typo == w:
            continue
        words[idx] = typo
        typo_query = " ".join(words)
        pair = (typo_query, q)
        if pair in seen:
            continue
        examples.append(make_example(typo_query, q, "phonetic_typo"))
        seen.add(pair)
    return examples


def gen_space_error_examples(
    seen: Set[Tuple[str, str]],
    count: int = 15000,
) -> List[dict]:
    """Merge two adjacent words or split one word."""
    examples: List[dict] = []
    brands = list(BRAND_CATALOG.keys())
    attempts = 0
    while len(examples) < count and attempts < count * 4:
        attempts += 1
        brand = random.choice(brands)
        product = random.choice(BRAND_CATALOG[brand])
        q = f"{brand} {product}"
        broken = typo_space_error(q)
        if not broken or broken == q:
            continue
        pair = (broken, q)
        if pair in seen:
            continue
        cat = "space_error"
        # subcategorize split-by-context
        if len(broken.split()) > len(q.split()):
            cat = "space_split_context"
        examples.append(make_example(broken, q, cat))
        seen.add(pair)
    return examples


def gen_vocab_typo_examples(
    vocab: List[str],
    seen: Set[Tuple[str, str]],
    count: int = 25000,
) -> List[dict]:
    """Standalone word-level typos from accumulated vocabulary."""
    examples: List[dict] = []
    valid = [w for w in vocab if w.isalpha() and 4 <= len(w) <= 15]
    random.shuffle(valid)
    for w in valid:
        if len(examples) >= count:
            break
        typo = generate_single_typo(w)
        if not typo or typo == w:
            continue
        pair = (typo, w)
        if pair in seen:
            continue
        examples.append(make_example(typo, w, "vocab_typo"))
        seen.add(pair)
    return examples


def gen_identity_examples(
    seen: Set[Tuple[str, str]],
    count: int = 50000,
) -> List[dict]:
    """
    Identity pairs — prevent over-correction. v3.1 adds 5-10x repetitions of
    tricky brand/tech entries so those are represented much more strongly.
    """
    examples: List[dict] = []
    brands = list(BRAND_CATALOG.keys())

    # 1. Brand names as-is. Repeat each several times with minor template variants.
    for brand in brands:
        variants = [brand, f"{brand} official", f"buy {brand}", f"{brand} store", f"{brand} website"]
        for v in variants:
            pair = (v, v)
            if pair not in seen:
                examples.append(make_example(v, v, "identity_brand"))
                seen.add(pair)

    # 2. Tricky brand names that look like misspellings (boosted)
    TRICKY_BRANDS = [
        "razer", "xerox", "asus", "acer", "oppo", "vivo", "poco",
        "roku", "sonos", "bose", "nzxt", "evga", "zotac", "adata",
        "skullcandy", "sennheiser", "breville", "cuisinart", "miele",
        "birkenstock", "lululemon", "patagonia", "timberland",
        "logitech", "corsair", "hyperx", "steelseries", "kingston",
    ]
    for brand in TRICKY_BRANDS:
        for template in [brand, f"{brand} official", f"buy {brand}", f"{brand} store",
                         f"{brand} products", f"new {brand}"]:
            pair = (template, template)
            if pair not in seen:
                examples.append(make_example(template, template, "identity_tricky"))
                seen.add(pair)

    # 3. Full queries (brand + product) identity
    for _ in range(count):
        brand = random.choice(brands)
        product = random.choice(BRAND_CATALOG[brand])
        template = random.choice(QUERY_TEMPLATES[:8])
        mod = random.choice(MODIFIERS)
        mod2 = random.choice(MODIFIERS)
        query = _format_template(template, brand=brand, product=product, mod=mod, mod2=mod2)
        if not query:
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

    # 5. Tech terms / model numbers (boosted 5x with template variants)
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
        for suffix in ["", " review", " price", " vs", " best", " 2024", " 2025"]:
            t = (term + suffix).strip()
            pair = (t, t)
            if pair not in seen:
                examples.append(make_example(t, t, "identity_tech"))
                seen.add(pair)

    return examples


# ─── v3.1 NEW GENERATORS ───────────────────────────────────────

def gen_aug_typo_examples(
    seen: Set[Tuple[str, str]],
    per_category: int = 55000,
) -> List[dict]:
    """
    Separately labelled augmentation pairs for each typo method, so we can
    measure per-method accuracy and keep even coverage.
    """
    fn_map = [
        ("aug_keyboard",    typo_keyboard,       70000),
        ("aug_double",      typo_double,         65000),
        ("aug_swap",        typo_swap,           58000),
        ("aug_phonetic",    typo_phonetic,       51000),
        ("aug_delete",      typo_delete,         49000),
        ("aug_insert",      typo_insert,         42000),
        ("aug_omit_repeat", typo_omit_repeated,  6000),
    ]
    examples: List[dict] = []
    brands = list(BRAND_CATALOG.keys())

    for cat_name, fn, target_n in fn_map:
        added = 0
        attempts = 0
        while added < target_n and attempts < target_n * 4:
            attempts += 1
            brand = random.choice(brands)
            product = random.choice(BRAND_CATALOG[brand])
            query = f"{brand} {product}"
            words = query.split()
            candidates = [i for i, w in enumerate(words) if len(w) >= 3 and w.isalpha()]
            if not candidates:
                continue
            idx = random.choice(candidates)
            typoed = fn(words[idx])
            if not typoed or typoed == words[idx]:
                continue
            words2 = words.copy()
            words2[idx] = typoed
            broken = " ".join(words2)
            pair = (broken, query)
            if pair in seen:
                continue
            examples.append(make_example(broken, query, cat_name))
            seen.add(pair)
            added += 1
    # Compound: apply 2 different methods on two different words
    target_compound = 16000
    added = 0
    attempts = 0
    while added < target_compound and attempts < target_compound * 4:
        attempts += 1
        brand = random.choice(brands)
        product = random.choice(BRAND_CATALOG[brand])
        query = f"{brand} {product}"
        words = query.split()
        candidates = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
        if len(candidates) < 2:
            continue
        i1, i2 = random.sample(candidates, 2)
        t1 = generate_single_typo(words[i1])
        t2 = generate_single_typo(words[i2])
        if not t1 or not t2:
            continue
        new = words.copy()
        new[i1] = t1
        new[i2] = t2
        broken = " ".join(new)
        if broken == query:
            continue
        pair = (broken, query)
        if pair in seen:
            continue
        examples.append(make_example(broken, query, "aug_compound"))
        seen.add(pair)
        added += 1
    return examples


def gen_price_examples(
    seen: Set[Tuple[str, str]],
    count: int = 40000,
) -> List[dict]:
    """
    Teach price formats: '$99', '$1,299.99', 'under $500', '299 USD',
    'between $100 and $200', 'for 29.99'. Both identity and typo'd variants.
    """
    examples: List[dict] = []
    brands = list(BRAND_CATALOG.keys())
    attempts = 0
    while len(examples) < count and attempts < count * 5:
        attempts += 1
        template = random.choice(PRICE_TEMPLATES_CLEAN)
        brand = random.choice(brands)
        product = random.choice(BRAND_CATALOG[brand] + PRODUCT_TERMS)
        p = random.randint(10, 2000)
        p2 = p + random.randint(50, 2000)
        cents = f"{random.randint(0,99):02d}"
        p_big = f"{random.randint(1,9)},{random.randint(0,999):03d}"
        price_code = f"{random.choice(CURRENCY_CODES)} {p}"
        if random.random() < 0.3:
            price_code = f"{p} {random.choice(CURRENCY_CODES)}"
        clean = _format_template(
            template, brand=brand, product=product,
            p=p, p2=p2, cents=cents, p_big=p_big, price_code=price_code,
        )
        if not clean:
            continue

        # identity example (teach the correct format)
        pair_id = (clean, clean)
        if pair_id not in seen:
            examples.append(make_example(clean, clean, "price_identity"))
            seen.add(pair_id)

        # typo'd variant: mangle currency spelling or digits lightly
        mangled_options = []
        if "$" in clean:
            mangled_options.append(clean.replace("$", "$ ", 1))
            mangled_options.append(clean.replace("$", "dolar ", 1))
            mangled_options.append(clean.replace("$", "dolars ", 1))
        for code in CURRENCY_CODES:
            if code in clean:
                lc = code.lower()
                mangled_options.append(clean.replace(code, lc))
                mangled_options.append(clean.replace(code, lc[:-1]))  # 'us' for 'usd'
                mangled_options.append(clean.replace(code, "uds"))  # swap
        if " ." in clean or "." in clean:
            mangled_options.append(clean.replace(".", " ."))
        if "," in clean:
            mangled_options.append(clean.replace(",", "."))
        if not mangled_options:
            continue
        broken = random.choice(mangled_options)
        if broken == clean or len(broken) > 80:
            continue
        pair_t = (broken, clean)
        if pair_t in seen:
            continue
        examples.append(make_example(broken, clean, "price_typo"))
        seen.add(pair_t)
    return examples


def gen_measurement_unit_examples(
    seen: Set[Tuple[str, str]],
    count: int = 40000,
) -> List[dict]:
    """
    Teach measurement unit formats for e-commerce specs.
    Produces both identity (correct) and typo'd variants.
    """
    examples: List[dict] = []
    brands = list(BRAND_CATALOG.keys())
    unit_families = [
        ("storage",   STORAGE_UNITS,   lambda: random.choice([16, 32, 64, 128, 256, 512, 1024, 2048])),
        ("freq",      FREQUENCY_UNITS, lambda: random.choice([60, 75, 90, 120, 144, 165, 240, 360])),
        ("power",     POWER_UNITS,     lambda: random.choice([5, 12, 24, 45, 65, 85, 100, 120, 1500, 2000, 3000, 5000])),
        ("length",    LENGTH_UNITS,    lambda: random.choice([5, 10, 13, 14, 15, 16, 17, 24, 27, 32])),
        ("weight",    WEIGHT_UNITS,    lambda: random.choice([100, 250, 500, 1, 2, 5, 10])),
        ("camera",    CAMERA_UNITS,    lambda: random.choice([12, 48, 50, 60, 108, 200])),
    ]
    modifiers = ["gaming", "office", "ultra", "pro", "max", "plus"]

    attempts = 0
    while len(examples) < count and attempts < count * 6:
        attempts += 1
        family_name, unit_list, n_fn = random.choice(unit_families)
        unit = random.choice(unit_list)
        n = n_fn()
        template = random.choice(UNIT_TEMPLATES_CLEAN)
        brand = random.choice(brands)
        product = random.choice(BRAND_CATALOG[brand] + PRODUCT_TERMS)
        res = random.choice(RESOLUTIONS)
        mod = random.choice(modifiers)

        unit_canonical = unit.upper() if unit not in ('"', 'inch', 'fps', 'dpi') else unit
        unit_canonical = {"MAH": "mAh", "KW": "kW", "KG": "kg", "LB": "lb", "LBS": "lbs",
                          "OZ": "oz", "MP": "MP", "FPS": "fps", "DPI": "dpi",
                          "MM": "mm", "CM": "cm", "M": "m", "FT": "ft",
                          "HZ": "Hz", "MHZ": "MHz", "GHZ": "GHz",
                          "W": "W", "V": "V", "WH": "Wh",
                          "MB": "MB", "GB": "GB", "TB": "TB",
                          "G": "g", "INCH": "inch"}.get(unit_canonical.upper(), unit_canonical)

        res_canonical = res.upper() if res[-1] in "kK" else res

        clean = _format_template(
            template, brand=brand, product=product,
            n=n, unit=unit_canonical, res=res_canonical, mod=mod,
        )
        if not clean:
            continue

        pair_id = (clean, clean)
        if pair_id not in seen:
            examples.append(make_example(clean, clean, "unit_identity"))
            seen.add(pair_id)

        # typo variants: lowercase the unit, add/remove space, swap digits
        mangled_options = [
            clean.replace(unit_canonical, unit.lower()),
            clean.replace(f" {unit_canonical}", unit_canonical.lower()),
            clean.replace(f"{n} {unit_canonical}", f"{n}{unit.lower()}"),
            clean.replace(res_canonical, res.lower()),
        ]
        if res_canonical in clean and res_canonical.endswith("K"):
            mangled_options.append(clean.replace(res_canonical, f"{res_canonical[:-1]} k"))
            mangled_options.append(clean.replace(res_canonical, f"{res_canonical[:-1]}K"))
        mangled_options = [m for m in mangled_options if m and m != clean]
        if not mangled_options:
            continue
        broken = random.choice(mangled_options)
        pair_t = (broken, clean)
        if pair_t in seen:
            continue
        examples.append(make_example(broken, clean, "unit_typo"))
        seen.add(pair_t)

    return examples


def gen_brand_category_mismatch_examples(
    seen: Set[Tuple[str, str]],
    count: int = 15000,
) -> List[dict]:
    """
    Teach the model to fix brand-model mismatches via context.
    e.g. "nvidia tuf gaming chair" -> "asus tuf gaming chair"
    (nvidia is GPUs; TUF is an ASUS gaming sub-line).
    """
    examples: List[dict] = []
    # For each confusion triple, generate many product contexts
    attempts = 0
    while len(examples) < count and attempts < count * 8:
        attempts += 1
        wrong_brand, right_brand, model_line = random.choice(CROSS_BRAND_MODEL_CONFUSIONS)
        # Pick a product that plausibly goes with the model_line
        right_products = BRAND_CATALOG.get(right_brand, []) + PRODUCT_TERMS
        product = random.choice(right_products)

        # Build wrong variant: "wrong_brand model_line product"
        # Build right variant: "right_brand model_line product"
        wrong = f"{wrong_brand} {model_line} {product}".strip()
        right = f"{right_brand} {model_line} {product}".strip()

        if wrong == right or len(wrong) > 80:
            continue
        pair = (wrong, right)
        if pair in seen:
            continue
        examples.append(make_example(wrong, right, "brand_category_mismatch"))
        seen.add(pair)

        # Also: identity of the right version (reinforce correct form)
        id_pair = (right, right)
        if id_pair not in seen:
            examples.append(make_example(right, right, "brand_category_identity"))
            seen.add(id_pair)

    return examples


# Everyday English words (1000+ common misspellings' source words)
EVERYDAY_WORDS = [
    "receive", "definitely", "occurred", "separate", "necessary", "occasionally",
    "accommodate", "acquire", "believe", "calendar", "cemetery", "changeable",
    "collectible", "committed", "conscientious", "conscience", "consensus",
    "daiquiri", "discipline", "drunkenness", "embarrass", "equipment",
    "exceed", "existence", "experience", "foreign", "friend", "government",
    "grateful", "guarantee", "harass", "height", "hierarchy", "humorous",
    "immediately", "independent", "indispensable", "inoculate", "intelligence",
    "jewelry", "judgment", "knowledge", "leisure", "library", "license",
    "lightning", "maintenance", "maneuver", "millennium", "miniature",
    "miscellaneous", "mischievous", "misspell", "neighbor", "noticeable",
    "occurrence", "paid", "pastime", "perseverance", "personnel", "playwright",
    "possession", "precede", "principal", "privilege", "professor", "pronunciation",
    "publicly", "questionnaire", "receipt", "recommend", "reference", "referred",
    "relevant", "restaurant", "rhyme", "rhythm", "schedule", "science",
    "secretary", "sergeant", "sincerely", "sophomore", "sovereign", "succeed",
    "supersede", "suppress", "tendency", "threshold", "tomorrow", "transferred",
    "truly", "twelfth", "tyranny", "unanimous", "until", "useful", "vacuum",
    "vehicle", "visible", "weather", "whether", "which", "writing", "written",
    "yield", "address", "amateur", "argument", "athletic", "beautiful",
    "beginning", "business", "category", "commitment", "committee", "competent",
    "conscious", "deceive", "desperate", "disastrous", "eighth", "eligible",
    "embarrassment", "environment", "especially", "exaggerate", "excellent",
    "explanation", "familiar", "fascinate", "February", "finally", "fluorescent",
    "forty", "frequent", "gauge", "genius", "guidance", "happily", "height",
    "humorous", "idiosyncrasy", "immediate", "innocuous", "irrelevant",
    "liaison", "marriage", "medicine", "memento", "minuscule", "necessary",
    "ninety", "noticeable", "ninth", "opinion", "opportunity", "ordinary",
    "parallel", "particular", "peculiar", "persistent", "perspective",
    "pleasant", "possess", "potato", "practice", "preceding", "preference",
    "preferred", "prejudice", "prevalent", "procedure", "proceed", "profession",
    "prominent", "pursue", "quantity", "quarrel", "quiet", "realize",
    "really", "recognize", "reminisce", "repetition", "reservoir", "ridiculous",
    "safety", "salary", "scheme", "separate", "siege", "similar", "simile",
    "simultaneous", "soliloquy", "souvenir", "specific", "strategy", "strength",
    "subtle", "successful", "summary", "surprise", "susceptible", "technique",
    "temperature", "temporarily", "theater", "theory", "threshold", "thorough",
    "thought", "through", "together", "tomorrow", "traffic", "tragedy",
    "typical", "usage", "usual", "vacuum", "variety", "various", "vegetable",
    "vehicle", "villain", "visible", "voluntary", "warranty", "weather",
    "weird", "welfare", "wholly", "width", "willful", "wilful", "wisdom",
    "wonderful", "workplace", "yesterday",
]


def gen_everyday_english_typos(
    seen: Set[Tuple[str, str]],
    count: int = 35000,
) -> List[dict]:
    """
    Teach typo correction of common English words (not brand/product names).
    Each word passes through the domain_plausible filter.
    """
    examples: List[dict] = []
    words = [w for w in EVERYDAY_WORDS if domain_plausible(w)]
    attempts = 0
    while len(examples) < count and attempts < count * 8:
        attempts += 1
        word = random.choice(words)
        typo_fn = random.choice([
            typo_keyboard, typo_double, typo_swap,
            typo_phonetic, typo_delete, typo_insert, typo_omit_repeated,
        ])
        typoed = typo_fn(word)
        if not typoed or typoed == word or len(typoed) < 2:
            continue
        # Sometimes embed in a natural context phrase
        if random.random() < 0.4:
            template = random.choice([
                "{word} review", "best {word}", "{word} for sale",
                "cheap {word}", "{word} online", "buy {word}",
                "{word} near me", "top {word}", "new {word}",
            ])
            broken = template.replace("{word}", typoed)
            correct = template.replace("{word}", word)
        else:
            broken = typoed
            correct = word
        pair = (broken, correct)
        if pair in seen:
            continue
        examples.append(make_example(broken, correct, "everyday_english"))
        seen.add(pair)
    return examples


def gen_external_word_pairs(
    seen: Set[Tuple[str, str]],
    external_pairs: List[Tuple[str, str]],
    count: int = 30000,
) -> List[dict]:
    """
    Use external misspelling corpora (Birkbeck / codespell / torinriley).
    Only keep pairs where the target word passes the domain filter.
    """
    examples: List[dict] = []
    if not external_pairs:
        return examples
    random.shuffle(external_pairs)
    for wrong, right in external_pairs:
        if len(examples) >= count:
            break
        wrong = wrong.strip().lower()
        right = right.strip().lower()
        if not wrong or not right or wrong == right:
            continue
        if not domain_plausible(right):
            continue
        if levenshtein(wrong, right) > max(3, len(right) // 3):
            continue
        pair = (wrong, right)
        if pair in seen:
            continue
        examples.append(make_example(wrong, right, "external_word"))
        seen.add(pair)
    return examples


def gen_external_context_typos(
    seen: Set[Tuple[str, str]],
    external_pairs: List[Tuple[str, str]],
    count: int = 25000,
) -> List[dict]:
    """Embed external corpus typos in e-commerce search contexts."""
    examples: List[dict] = []
    if not external_pairs:
        return examples
    plausible_pairs = [(w, r) for w, r in external_pairs
                       if domain_plausible(r.strip().lower())
                       and levenshtein(w.strip().lower(), r.strip().lower()) <= max(3, len(r) // 3)]
    if not plausible_pairs:
        return examples
    templates = [
        "{x} review", "best {x}", "buy {x}", "{x} online",
        "cheap {x}", "top {x}", "{x} near me",
    ]
    attempts = 0
    while len(examples) < count and attempts < count * 6:
        attempts += 1
        wrong, right = random.choice(plausible_pairs)
        template = random.choice(templates)
        broken = template.replace("{x}", wrong.strip().lower())
        correct = template.replace("{x}", right.strip().lower())
        if broken == correct:
            continue
        pair = (broken, correct)
        if pair in seen:
            continue
        examples.append(make_example(broken, correct, "external_context"))
        seen.add(pair)
    return examples


def gen_identity_external(
    seen: Set[Tuple[str, str]],
    external_pairs: List[Tuple[str, str]],
    count: int = 15000,
) -> List[dict]:
    """
    Preserve correctly-spelled words that LOOK like typos.
    Uses only the 'right' side of external pairs filtered by domain_plausible.
    """
    examples: List[dict] = []
    if not external_pairs:
        return examples
    corrects = list({r.strip().lower() for _, r in external_pairs if domain_plausible(r.strip().lower())})
    random.shuffle(corrects)
    templates = [
        "{x}", "{x} review", "best {x}", "buy {x}",
        "{x} online", "top {x}",
    ]
    attempts = 0
    while len(examples) < count and attempts < count * 6:
        attempts += 1
        word = random.choice(corrects)
        tpl = random.choice(templates)
        s = tpl.replace("{x}", word)
        pair = (s, s)
        if pair in seen:
            continue
        examples.append(make_example(s, s, "identity_external"))
        seen.add(pair)
    return examples


# ═══════════════════════════════════════════════════════════════
# v3.1 QUALITY FILTER (true Levenshtein, replaces positional diff)
# ═══════════════════════════════════════════════════════════════

def quality_filter(examples: List[dict]) -> Tuple[List[dict], Dict[str, int]]:
    """
    Drop pairs that look corrupted:
      - empty / whitespace-only
      - identical except where category says "identity" (those are OK)
      - non-ASCII characters (we want English only)
      - edit distance too large (d / max(len) > 0.5)
      - lengths wildly different
      - duplicate (input, target) pair
    """
    seen_pairs: Set[Tuple[str, str]] = set()
    keep: List[dict] = []
    stats = {
        "total_in": len(examples),
        "dropped_empty": 0,
        "dropped_nonascii": 0,
        "dropped_edit_distance": 0,
        "dropped_length_ratio": 0,
        "dropped_duplicate": 0,
        "dropped_conflict": 0,
    }
    target_for_input: Dict[str, str] = {}
    for ex in examples:
        inp = ex["input_text"]
        tgt = ex["target_text"]
        cat = ex.get("category", "")
        if not inp.strip() or not tgt.strip():
            stats["dropped_empty"] += 1
            continue
        try:
            inp.encode("ascii")
            tgt.encode("ascii")
        except UnicodeEncodeError:
            stats["dropped_nonascii"] += 1
            continue
        # identity categories allowed to have inp == tgt
        is_identity = "identity" in cat
        if inp == tgt and not is_identity:
            stats["dropped_empty"] += 1
            continue
        if not is_identity:
            d = levenshtein(inp, tgt)
            longer = max(len(inp), len(tgt))
            if longer > 0 and d / longer > 0.5:
                stats["dropped_edit_distance"] += 1
                continue
            if len(inp) > 3 and len(tgt) > 3:
                ratio = min(len(inp), len(tgt)) / max(len(inp), len(tgt))
                if ratio < 0.4:
                    stats["dropped_length_ratio"] += 1
                    continue
        key = (inp, tgt)
        if key in seen_pairs:
            stats["dropped_duplicate"] += 1
            continue
        # detect conflicting targets for same input (keep first)
        if inp in target_for_input and target_for_input[inp] != tgt:
            stats["dropped_conflict"] += 1
            continue
        target_for_input[inp] = tgt
        seen_pairs.add(key)
        keep.append(ex)
    stats["total_out"] = len(keep)
    return keep, stats


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def _discover_external_corpora() -> List[Path]:
    """Scan known dirs for external typo corpora."""
    scan_dirs = [DATA_DIR, DATA_DIR / "external", V3_ROOT / "data" / "external",
                 SCRIPT_DIR / "external"]
    found = []
    for d in scan_dirs:
        if not d.exists():
            continue
        for ext in ("*.csv", "*.tsv", "*.arrow", "*.txt", "*.jsonl"):
            for p in d.glob(ext):
                # skip our own training outputs
                nm = p.name.lower()
                if any(x in nm for x in ("train_v3", "eval_v3", "training_stats",
                                          "training_data_v3", "eval_data_v3")):
                    continue
                found.append(p)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=950000,
                        help="target total dataset size (before filter)")
    parser.add_argument("--identity-ratio", type=float, default=0.22,
                        help="fraction of dataset that should be identity pairs")
    parser.add_argument("--eval-size", type=int, default=25000,
                        help="size of eval split")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR),
                        help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BYT5-LARGE v3.1 TRAINING DATA BUILDER")
    print("=" * 70)
    print(f"Target size:      {args.target:,}")
    print(f"Identity ratio:   {args.identity_ratio}")
    print(f"Eval size:        {args.eval_size:,}")
    print(f"Output dir:       {out_dir}")
    print("=" * 70)

    seen: Set[Tuple[str, str]] = set()
    all_examples: List[dict] = []

    # Load inputs
    print("\n── Loading source corpora from", DATA_DIR)
    typo_mappings = load_typo_mappings(DATA_DIR / "typo_mappings.txt")
    csv_pairs     = load_csv_pairs(DATA_DIR / "typo_dataset.csv")
    vocab_list: List[str] = []
    for vocab_file in ("expanded_common_words.txt", "curated_common_words.txt",
                        "common_words.txt", "domain_vocab.txt",
                        "electronics_vocab.txt", "ecommerce_corpus.txt"):
        p = DATA_DIR / vocab_file
        if p.exists():
            vocab_list.extend(load_text_file(p))
    # Deduplicate, keep short/plausible
    vocab_list = list({v.strip().lower() for v in vocab_list
                       if v and v.strip() and 3 <= len(v.strip()) <= 18})
    print(f"   typo_mappings:  {len(typo_mappings):,}")
    print(f"   csv_pairs:      {len(csv_pairs):,}")
    print(f"   vocab_list:     {len(vocab_list):,}")

    # ── 1. Curated typos
    print("\n[1/14] gen_curated_typo_examples ...")
    ex = gen_curated_typo_examples(typo_mappings, seen)
    print(f"       + {len(ex):,}")
    all_examples.extend(ex)

    # ── 2. Existing training data (if any)
    print("[2/14] gen_existing_data_examples ...")
    ex = gen_existing_data_examples(typo_mappings, csv_pairs, seen)
    print(f"       + {len(ex):,}")
    all_examples.extend(ex)

    # ── 3. Brand/product typos
    print("[3/14] gen_brand_product_typos ...")
    ex = gen_brand_product_typos(seen, count=120000)
    print(f"       + {len(ex):,}")
    all_examples.extend(ex)

    # ── 4. Generic product typos
    print("[4/14] gen_generic_product_typos ...")
    ex = gen_generic_product_typos(seen, count=60000)
    print(f"       + {len(ex):,}")
    all_examples.extend(ex)

    # ── 5. Phonetic
    print("[5/14] gen_phonetic_typo_examples ...")
    ex = gen_phonetic_typo_examples(seen, count=40000)
    print(f"       + {len(ex):,}")
    all_examples.extend(ex)

    # ── 6. Space errors
    print("[6/14] gen_space_error_examples ...")
    ex = gen_space_error_examples(seen, count=25000)
    print(f"       + {len(ex):,}")
    all_examples.extend(ex)

    # ── 7. Vocab typos
    print("[7/14] gen_vocab_typo_examples ...")
    ex = gen_vocab_typo_examples(vocab_list, seen, count=25000)
    print(f"       + {len(ex):,}")
    all_examples.extend(ex)

    # ── 8. Identity
    identity_target = int(args.target * args.identity_ratio)
    print(f"[8/14] gen_identity_examples (target {identity_target:,}) ...")
    ex = gen_identity_examples(seen, count=identity_target)
    print(f"       + {len(ex):,}")
    all_examples.extend(ex)

    # ── 9. Augmented (keyboard/double/swap/phonetic/delete/insert/compound/omit_repeat)
    print("[9/14] gen_aug_typo_examples ...")
    ex = gen_aug_typo_examples(seen)
    print(f"       + {len(ex):,}")
    all_examples.extend(ex)

    # ── 10. Price
    print("[10/14] gen_price_examples ...")
    ex = gen_price_examples(seen, count=40000)
    print(f"        + {len(ex):,}")
    all_examples.extend(ex)

    # ── 11. Units
    print("[11/14] gen_measurement_unit_examples ...")
    ex = gen_measurement_unit_examples(seen, count=40000)
    print(f"        + {len(ex):,}")
    all_examples.extend(ex)

    # ── 12. Brand-category mismatch
    print("[12/14] gen_brand_category_mismatch_examples ...")
    ex = gen_brand_category_mismatch_examples(seen, count=15000)
    print(f"        + {len(ex):,}")
    all_examples.extend(ex)

    # ── 13. Everyday English
    print("[13/14] gen_everyday_english_typos ...")
    ex = gen_everyday_english_typos(seen, count=35000)
    print(f"        + {len(ex):,}")
    all_examples.extend(ex)

    # ── 14. External corpora (auto-discover)
    print("[14/14] External corpora (auto-discover) ...")
    corpora_paths = _discover_external_corpora()
    all_ext: List[Tuple[str, str]] = []
    for p in corpora_paths:
        pairs = load_external_pairs(p)
        print(f"        loaded {len(pairs):,} pairs from {p.name}")
        all_ext.extend(pairs)
    if all_ext:
        ex1 = gen_external_word_pairs(seen, all_ext, count=30000)
        ex2 = gen_external_context_typos(seen, all_ext, count=25000)
        ex3 = gen_identity_external(seen, all_ext, count=15000)
        print(f"        + {len(ex1):,} external_word")
        print(f"        + {len(ex2):,} external_context")
        print(f"        + {len(ex3):,} identity_external")
        all_examples.extend(ex1)
        all_examples.extend(ex2)
        all_examples.extend(ex3)
    else:
        print("        (no external corpora found — skipping)")

    print(f"\n─── Raw total: {len(all_examples):,} examples ───")

    # Quality filter
    print("Running quality filter ...")
    all_examples, qstats = quality_filter(all_examples)
    print(f"  total_in         : {qstats['total_in']:,}")
    print(f"  dropped_empty    : {qstats['dropped_empty']:,}")
    print(f"  dropped_nonascii : {qstats['dropped_nonascii']:,}")
    print(f"  dropped_edit_dist: {qstats['dropped_edit_distance']:,}")
    print(f"  dropped_len_ratio: {qstats['dropped_length_ratio']:,}")
    print(f"  dropped_duplicate: {qstats['dropped_duplicate']:,}")
    print(f"  dropped_conflict : {qstats['dropped_conflict']:,}")
    print(f"  kept             : {qstats['total_out']:,}")

    # Shuffle + split
    random.shuffle(all_examples)
    eval_split = all_examples[:args.eval_size]
    train_split = all_examples[args.eval_size:]
    # Verify no overlap
    eval_keys = {(e["input_text"], e["target_text"]) for e in eval_split}
    train_split = [e for e in train_split if (e["input_text"], e["target_text"]) not in eval_keys]

    # Category counts
    from collections import Counter
    cat_train = Counter(e["category"] for e in train_split)
    cat_eval = Counter(e["category"] for e in eval_split)

    # Write
    train_file = out_dir / "train_v3.jsonl"
    eval_file = out_dir / "eval_v3.jsonl"
    stats_file = out_dir / "training_stats_v3.json"

    print(f"\nWriting {len(train_split):,} -> {train_file.name}")
    with open(train_file, "w", encoding="utf-8") as f:
        for ex in train_split:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Writing {len(eval_split):,} -> {eval_file.name}")
    with open(eval_file, "w", encoding="utf-8") as f:
        for ex in eval_split:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    stats = {
        "version": "v3.1",
        "train_size": len(train_split),
        "eval_size": len(eval_split),
        "train_categories": dict(cat_train.most_common()),
        "eval_categories": dict(cat_eval.most_common()),
        "quality_filter": qstats,
        "overlap_check": 0,
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Writing stats -> {stats_file.name}")
    print("\n=== TOP 15 CATEGORIES (train) ===")
    for name, c in cat_train.most_common(15):
        print(f"  {name:32s} {c:>8,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
