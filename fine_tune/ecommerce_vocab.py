"""
E-Commerce Domain Vocabulary Builder

Bu modül, farklı e-ticaret kategorileri için vocabulary ve typo mapping'leri yönetir.
Şu an elektronik cihazlar ağırlıklı, ama ileride diğer kategoriler eklenebilir.

Categories:
- electronics: Elektronik cihazlar, markalar, modeller
- fashion: Giyim, ayakkabı, aksesuar (TODO)
- home: Ev & Yaşam (TODO)
- sports: Spor & Outdoor (TODO)

Usage:
    from ecommerce_vocab import ECommerceVocab
    
    vocab = ECommerceVocab()
    vocab.add_category("electronics")
    vocab.add_category("fashion")  # Future
    
    training_data = vocab.generate_training_data(augment=True)
"""

import os
import re
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class Category:
    """E-commerce category with vocabulary and typo mappings."""
    name: str
    brands: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    abbreviations: Dict[str, str] = field(default_factory=dict)  # abbr -> full
    typo_mappings: Dict[str, str] = field(default_factory=dict)  # typo -> correct
    common_terms: List[str] = field(default_factory=list)


# =============================================================================
# ELECTRONICS CATEGORY - Kapsamlı Elektronik Cihaz Vocabulary
# =============================================================================

ELECTRONICS_BRANDS = {
    # Smartphone & Mobile
    "apple": ["iphone", "ipad", "macbook", "imac", "mac mini", "mac studio", "mac pro", "airpods", "apple watch", "homepod", "apple tv", "vision pro", "magsafe"],
    "samsung": ["galaxy", "galaxy s", "galaxy a", "galaxy m", "galaxy z", "galaxy tab", "galaxy watch", "galaxy buds", "odyssey", "qled", "neo qled", "frame", "freestyle", "smart monitor"],
    "google": ["pixel", "pixel pro", "pixel a", "pixel fold", "pixel tablet", "pixel watch", "pixel buds", "nest", "chromecast", "fitbit"],
    "xiaomi": ["redmi", "poco", "mi", "redmi note", "poco x", "poco f", "poco m", "mi band", "mi tv", "mi box", "mi stick", "robot vacuum"],
    "huawei": ["mate", "pura", "p series", "nova", "watch", "freebuds", "matepad", "matebook"],
    "oneplus": ["nord", "ace", "open", "watch", "buds", "pad"],
    "oppo": ["find", "reno", "a series", "enco", "pad"],
    "vivo": ["x series", "v series", "y series", "tws", "iqoo"],
    "realme": ["gt", "narzo", "c series", "buds", "pad", "book"],
    "motorola": ["moto g", "moto e", "razr", "edge", "thinkphone"],
    "sony": ["xperia", "wh-1000xm", "wf-1000xm", "inzone", "bravia", "playstation", "alpha", "cybershot", "walkman"],
    "nothing": ["phone", "ear", "cmf"],
    
    # Laptop/PC Brands
    "dell": ["xps", "inspiron", "latitude", "precision", "alienware", "vostro", "optiplex", "g series", "monitor"],
    "hp": ["spectre", "envy", "pavilion", "omen", "victus", "elitebook", "probook", "zbook", "dragonfly", "laserjet", "deskjet", "inktank"],
    "lenovo": ["thinkpad", "ideapad", "legion", "loq", "yoga", "thinkcentre", "thinkstation", "tab", "thinkbook"],
    "asus": ["rog", "tuf", "zenbook", "vivobook", "proart", "expertbook", "rog strix", "rog zephyrus", "rog ally", "prime", "tuf gaming"],
    "acer": ["predator", "nitro", "aspire", "swift", "spin", "chromebook", "travelmate"],
    "msi": ["stealth", "raider", "vector", "titan", "crosshair", "katana", "pulse", "cyborg", "prestige", "modern", "summit", "suprim", "ventus", "gaming x"],
    "razer": ["blade", "book", "kraken", "blackwidow", "deathadder", "viper", "basilisk", "huntsman", "barracuda", "hammerhead"],
    "microsoft": ["surface", "surface pro", "surface laptop", "surface go", "surface book", "surface studio", "xbox", "xbox series", "intellimouse"],
    "gigabyte": ["aorus", "aero", "eagle", "gaming oc", "windforce"],
    
    # Component Brands
    "nvidia": ["geforce", "rtx", "gtx", "quadro", "tesla", "shield"],
    "amd": ["ryzen", "radeon", "threadripper", "epyc", "athlon", "freesync"],
    "intel": ["core", "xeon", "celeron", "pentium", "arc", "nuc", "evo"],
    "western digital": ["wd blue", "wd black", "wd red", "wd green", "wd purple", "elements", "my passport", "sn850", "sn770"],
    "seagate": ["barracuda", "ironwolf", "skyhawk", "firecuda", "expansion", "one touch"],
    "sandisk": ["extreme", "ultra", "cruzer", "clip"],
    "kingston": ["fury", "beast", "renegade", "canvas", "datatraveler"],
    "crucial": ["mx500", "bx500", "p3", "p5", "ballistix"],
    "corsair": ["vengeance", "dominator", "force", "mp600", "k70", "k100", "harpoon", "void", "virtuoso", "hs80", "icue", "elgato"],
    "g.skill": ["trident", "ripjaws", "flare"],
    
    # Peripheral & Accessory Brands
    "logitech": ["g pro", "g502", "mx master", "mx keys", "g915", "g733", "streamcam", "c920", "c922", "g29", "g923", "pebble", "lift"],
    "steelseries": ["arctis", "apex", "rival", "aerox", "prime", "qck"],
    "hyperx": ["cloud", "pulsefire", "alloy", "quadcast", "fury"],
    "anker": ["soundcore", "eufy", "nebula", "powercore", "powerport", "nano", "maggo", "roav"],
    "baseus": ["blade", "gan", "encok"],
    "ugreen": ["nexode", "hitune"],
    "belkin": ["boostcharge", "soundform"],
    "tp-link": ["archer", "deco", "tapo", "kasa", "omada"],
    "ubiquiti": ["unifi", "amplifi", "dream machine"],
    "netgear": ["nighthawk", "orbi"],
    
    # Audio & Visual
    "bose": ["quietcomfort", "soundlink", "soundbar", "smart soundbar", "nc700"],
    "jbl": ["flip", "charge", "pulse", "tune", "live", "reflect", "partybox", "go", "clip", "quantum"],
    "sennheiser": ["momentum", "hd", "ie", "gsp", "accentum"],
    "audio-technica": ["ath-m50", "ath-m40", "at2020", "at2040"],
    "lg": ["ultragear", "ultrafine", "oled", "nanocell", "gram", "cinebeam", "tone free", "xboom"],
    "benq": ["zowie", "mobiuz", "pd", "ew", "tk", "th"],
    "viewsonic": ["elite", "vx", "xg", "m1", "m2"],
    "epson": ["ecotank", "workforce", "cinema", "home cinema", "epson"],
    "canon": ["eos", "pixma", "powershot", "imageclass"],
    "nikon": ["z series", "d series", "coolpix"],
    "fujifilm": ["instax", "x series", "gfx"],
    "gopro": ["hero", "max"],
    "dji": ["mavic", "mini", "air", "avata", "osmo", "ronin"],
    "nintendo": ["switch", "switch oled", "switch lite", "joy-con", "pro controller"],
    "valve": ["steam deck", "index"],
}

ELECTRONICS_PRODUCT_TYPES = [
    # Mobile & Wearables
    "smartphone", "phone", "tablet", "smartwatch", "watch", "fitness tracker", "smart band",
    "earbuds", "earphones", "headphones", "headset", "tws", "iem",
    "charger", "cable", "case", "screen protector", "power bank", "wireless charger", "magsafe",
    
    # Computer & Office
    "laptop", "notebook", "ultrabook", "chromebook", "desktop", "pc", "workstation", "all-in-one", "mini pc",
    "monitor", "keyboard", "mouse", "mousepad", "webcam", "microphone", "speaker", "soundbar",
    "printer", "scanner", "inkjet", "laser printer", "3d printer", "projector", "mini projector",
    "usb hub", "docking station", "card reader", "kvm switch", "ups",
    
    # Components
    "gpu", "graphics card", "video card", "cpu", "processor", "motherboard", "mainboard",
    "ram", "memory", "ddr4", "ddr5", "dimm", "sodimm",
    "ssd", "hdd", "hard drive", "internal hard drive", "external hard drive", "nvme", "m.2", "sata",
    "psu", "power supply", "case", "chassis", "tower",
    "cooler", "cpu cooler", "liquid cooler", "aio", "case fan", "thermal paste",
    "capture card", "sound card", "network card",
    
    # Gaming
    "gaming laptop", "gaming pc", "gaming mouse", "gaming keyboard", "gaming headset", 
    "gaming monitor", "gaming chair", "gaming desk",
    "controller", "gamepad", "joystick", "steering wheel", "flight stick",
    "console", "handheld console", "vr headset",
    
    # Networking & Smart Home
    "router", "modem", "access point", "wifi adapter", "ethernet cable", "network switch", "mesh system", "wifi extender",
    "smart bulb", "smart plug", "smart switch", "security camera", "ip camera", "video doorbell", "smart lock",
    "robot vacuum", "air purifier",
    
    # Camera & Photo
    "camera", "digital camera", "mirrorless camera", "dslr", "action camera", "drone", "gimbal",
    "lens", "tripod", "ring light", "softbox", "memory card", "sd card", "microsd",
    
    # Accessories
    "adapter", "converter", "cable organizer", "laptop stand", "monitor arm", "monitor mount",
    "backpack", "sleeve", "cleaning kit", "thermal pad",
]

# Model Number Patterns (önemli - kullanıcılar bunları yazar)
ELECTRONICS_MODEL_PATTERNS = {
    # Phones
    "iphone": ["15", "15 pro", "15 pro max", "15 plus", "14", "14 pro", "14 pro max", "13", "13 mini", "12", "11", "xr", "se"],
    "galaxy": ["s24", "s24+", "s24 ultra", "s23", "s23 fe", "s22", "z fold5", "z flip5", "a55", "a35", "a25", "a15", "m34", "f34"],
    "pixel": ["8", "8 pro", "8a", "7", "7 pro", "7a", "6", "6 pro", "fold"],
    "redmi": ["note 13", "note 13 pro", "note 13 pro+", "13c", "12", "note 12"],
    
    # Laptops
    "macbook": ["air m1", "air m2", "air m3", "pro m1", "pro m2", "pro m3", "pro 14", "pro 16"],
    "thinkpad": ["x1 carbon", "x1 yoga", "t14", "e14", "e15", "l14", "z13", "p16"],
    "rog": ["zephyrus g14", "zephyrus g16", "strix g16", "strix scar 18", "flow x13", "ally"],
    "legion": ["pro 7i", "pro 5i", "slim 7", "slim 5", "go"],
    
    # Components
    "rtx": ["4090", "4080", "4070 ti super", "4070 super", "4070", "4060 ti", "4060", "3060", "3050"],
    "rx": ["7900 xtx", "7900 xt", "7800 xt", "7700 xt", "7600", "6700 xt", "6600"],
    "ryzen": ["9 7950x3d", "7 7800x3d", "7 5800x3d", "5 7600x", "5 5600", "9 9950x", "7 9700x"],
    "core": ["i9-14900k", "i7-14700k", "i5-14600k", "i9-13900k", "i7-13700k", "i5-13400f", "ultra 7", "ultra 9"],
    
    # Consoles
    "playstation": ["5", "5 slim", "5 pro", "4", "4 pro", "vr2", "portal"],
    "xbox": ["series x", "series s", "one x", "one s", "wireless controller"],
    "switch": ["oled", "lite", "v2"],
    "steam": ["deck", "deck oled"],
    
    # Others
    "gopro": ["hero 12", "hero 11", "hero 10", "max"],
    "dji": ["mini 4 pro", "mini 3", "air 3", "mavic 3", "avata 2", "osmo action 4", "pocket 3"],
    "ipad": ["pro 12.9", "pro 11", "air 5", "mini 6", "10th gen", "9th gen"],
    "watch": ["series 9", "ultra 2", "se 2", "series 8", "ultra"],
    "monitor": ["24 inch", "27 inch", "32 inch", "34 inch", "49 inch", "144hz", "165hz", "240hz", "360hz", "540hz", "oled", "ips", "va"],
}

# Common abbreviations in electronics
ELECTRONICS_ABBREVIATIONS = {
    "gpu": "graphics processing unit",
    "cpu": "central processing unit",
    "ram": "random access memory",
    "ssd": "solid state drive",
    "hdd": "hard disk drive",
    "nvme": "non-volatile memory express",
    "psu": "power supply unit",
    "aio": "all in one",
    "rgb": "red green blue",
    "fps": "frames per second",
    "hz": "hertz",
    "tb": "terabyte",
    "gb": "gigabyte",
    "mb": "megabyte",
    "usb": "universal serial bus",
    "hdmi": "high definition multimedia interface",
    "dp": "displayport",
    "tws": "true wireless stereo",
    "anc": "active noise cancellation",
    "ips": "in-plane switching",
    "oled": "organic light emitting diode",
    "qled": "quantum dot led",
    "lcd": "liquid crystal display",
    "led": "light emitting diode",
    "wifi": "wireless fidelity",
    "bt": "bluetooth",
    "nfc": "near field communication",
    "5g": "fifth generation",
    "lte": "long term evolution",
    "os": "operating system",
    "ui": "user interface",
}

# Common typo patterns for electronics
ELECTRONICS_TYPO_PATTERNS = {
    # Brand typos
    "iphnoe": "iphone", "iphoen": "iphone", "iphne": "iphone", "ihpone": "iphone",
    "samsng": "samsung", "samsnug": "samsung", "samsug": "samsung", "sumsung": "samsung",
    "macbok": "macbook", "mackbook": "macbook", "mcbook": "macbook",
    "airpod": "airpods", "airpds": "airpods", "aripods": "airpods",
    "nvidai": "nvidia", "nvidea": "nvidia", "nividia": "nvidia",
    "geforc": "geforce", "gefroce": "geforce", "gerforce": "geforce",
    "lenvo": "lenovo", "lenvoo": "lenovo", "lenova": "lenovo",
    "thinkpd": "thinkpad", "thnikpad": "thinkpad", "thinpad": "thinkpad",
    "logitec": "logitech", "logitek": "logitech",
    "corsiar": "corsair", "corsar": "corsair", "crosair": "corsair",
    "razr": "razer", "razzer": "razer",
    "surfce": "surface", "surfac": "surface",
    "blutooth": "bluetooth", "bluethooth": "bluetooth", "bluetoth": "bluetooth",
    "wireles": "wireless", "wirelss": "wireless", "wirless": "wireless",
    "chargr": "charger", "charger": "charger", "chager": "charger",
    "labtop": "laptop", "laptpp": "laptop", "laptp": "laptop",
    "moniter": "monitor", "monitr": "monitor", "mointor": "monitor",
    "keybord": "keyboard", "keybaord": "keyboard", "keybard": "keyboard",
    "headphons": "headphones", "headfones": "headphones", "hedphones": "headphones",
    "speakr": "speaker", "speeker": "speaker", "speker": "speaker",
    
    # Product type typos
    "smartfone": "smartphone", "samartphone": "smartphone",
    "tablt": "tablet", "tablat": "tablet",
    "earbusd": "earbuds", "earbudd": "earbuds",
    "webcma": "webcam", "wbcam": "webcam",
    "microphne": "microphone", "micropone": "microphone",
    
    # Technical term typos
    "procesor": "processor", "processer": "processor",
    "memroy": "memory", "memmory": "memory", "memry": "memory",
    "graphis": "graphics", "graphix": "graphics",
    "gamign": "gaming", "gaiming": "gaming",
}


def build_electronics_category() -> Category:
    """Build comprehensive electronics category."""
    cat = Category(name="electronics")
    
    # Add all brands
    cat.brands = list(ELECTRONICS_BRANDS.keys())
    
    # Add all products from brand product lines
    for brand, products in ELECTRONICS_BRANDS.items():
        cat.products.extend(products)
        # Also add brand + product combinations
        for product in products:
            cat.products.append(f"{brand} {product}")
    
    # Add product types
    cat.common_terms = ELECTRONICS_PRODUCT_TYPES.copy()
    
    # Add model patterns
    for base, models in ELECTRONICS_MODEL_PATTERNS.items():
        for model in models:
            cat.models.append(f"{base} {model}")
    
    # Add abbreviations
    cat.abbreviations = ELECTRONICS_ABBREVIATIONS.copy()
    
    # Add typo mappings
    cat.typo_mappings = ELECTRONICS_TYPO_PATTERNS.copy()
    
    return cat


# =============================================================================
# FUTURE CATEGORIES (Placeholder)
# =============================================================================

def build_fashion_category() -> Category:
    """Fashion category - TODO: Implement when needed."""
    cat = Category(name="fashion")
    cat.brands = ["nike", "adidas", "puma", "zara", "h&m", "uniqlo", "gucci", "louis vuitton"]
    cat.common_terms = ["shirt", "pants", "dress", "shoes", "sneakers", "jacket", "coat"]
    cat.typo_mappings = {
        "addidas": "adidas",
        "nikee": "nike",
        "snekers": "sneakers",
        "shrit": "shirt",
    }
    return cat


def build_home_category() -> Category:
    """Home & Living category - TODO: Implement when needed."""
    cat = Category(name="home")
    cat.brands = ["ikea", "dyson", "philips", "bosch", "electrolux"]
    cat.common_terms = ["vacuum", "blender", "coffee maker", "air fryer", "toaster"]
    return cat


# =============================================================================
# CATEGORY REGISTRY
# =============================================================================

CATEGORY_BUILDERS = {
    "electronics": build_electronics_category,
    "fashion": build_fashion_category,
    "home": build_home_category,
}


class ECommerceVocab:
    """E-commerce vocabulary manager for multiple categories."""
    
    def __init__(self):
        self.categories: Dict[str, Category] = {}
    
    def add_category(self, name: str):
        """Add a category by name."""
        if name in CATEGORY_BUILDERS:
            self.categories[name] = CATEGORY_BUILDERS[name]()
            print(f"✓ Added category: {name}")
        else:
            print(f"✗ Unknown category: {name}")
            print(f"  Available: {list(CATEGORY_BUILDERS.keys())}")
    
    def get_all_vocabulary(self) -> Set[str]:
        """Get all vocabulary terms from all categories."""
        vocab = set()
        for cat in self.categories.values():
            vocab.update(cat.brands)
            vocab.update(cat.products)
            vocab.update(cat.models)
            vocab.update(cat.common_terms)
            vocab.update(cat.abbreviations.keys())
        return vocab
    
    def get_all_typo_mappings(self) -> Dict[str, str]:
        """Get all typo mappings from all categories."""
        mappings = {}
        for cat in self.categories.values():
            mappings.update(cat.typo_mappings)
        return mappings
    
    def get_category_stats(self) -> Dict:
        """Get statistics for all categories."""
        stats = {}
        for name, cat in self.categories.items():
            stats[name] = {
                "brands": len(cat.brands),
                "products": len(cat.products),
                "models": len(cat.models),
                "terms": len(cat.common_terms),
                "abbreviations": len(cat.abbreviations),
                "typo_mappings": len(cat.typo_mappings),
            }
        return stats


# =============================================================================
# MAIN - Test
# =============================================================================

if __name__ == "__main__":
    vocab = ECommerceVocab()
    vocab.add_category("electronics")
    
    stats = vocab.get_category_stats()
    print("\n📊 Category Statistics:")
    for cat_name, cat_stats in stats.items():
        print(f"\n{cat_name.upper()}:")
        for key, value in cat_stats.items():
            print(f"  {key}: {value}")
    
    print(f"\n📝 Total vocabulary: {len(vocab.get_all_vocabulary())}")
    print(f"🔤 Total typo mappings: {len(vocab.get_all_typo_mappings())}")
