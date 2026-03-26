"""
Universal E-Commerce Vocabulary Builder — ALL Categories
=========================================================

Her e-ticaret kategorisinde markalar, ürünler, model numaraları,
kısaltmalar, birimler, para birimleri ve typo mapping'leri.

Categories:
    - electronics    → Tam (mevcut + genişletilmiş)
    - fashion        → Tam
    - beauty         → Tam
    - home           → Tam
    - sports         → Tam
    - toys           → Tam
    - automotive     → Tam
    - grocery        → Tam
    - books          → Tam
    - office         → Tam

Usage:
    from universal_ecommerce_vocab import UniversalVocab

    vocab = UniversalVocab()
    vocab.load_all()

    all_brands = vocab.get_all_brands()
    all_typos  = vocab.get_all_typo_mappings()
    all_vocab  = vocab.get_all_vocabulary()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import random

# ======================================================================
# Data Class
# ======================================================================

@dataclass
class Category:
    """E-commerce category with full vocabulary."""
    name: str
    brands: Dict[str, List[str]] = field(default_factory=dict)        # brand -> sub-brands/lines
    products: List[str] = field(default_factory=list)
    model_patterns: Dict[str, List[str]] = field(default_factory=dict) # base -> variants
    abbreviations: Dict[str, str] = field(default_factory=dict)        # abbr -> full
    typo_mappings: Dict[str, str] = field(default_factory=dict)        # typo -> correct
    common_terms: List[str] = field(default_factory=list)
    units: List[str] = field(default_factory=list)                     # protected units


# ======================================================================
# GLOBAL: Units, Currencies, Abbreviations (cross-category)
# ======================================================================

GLOBAL_UNITS = [
    # Weight
    "kg", "kgs", "g", "gm", "gms", "mg", "lbs", "lb", "oz", "ounce", "ounces",
    "ton", "tonnes",
    # Length
    "cm", "mm", "m", "km", "in", "inch", "inches", "ft", "feet", "yd", "yards",
    # Volume
    "ml", "l", "liter", "liters", "litre", "litres", "fl oz", "gal", "gallon",
    "qt", "quart", "pt", "pint", "cc", "cl",
    # Digital
    "kb", "mb", "gb", "tb", "pb", "kbps", "mbps", "gbps",
    "mhz", "ghz", "hz", "khz",
    # Screen / Display
    "px", "dpi", "ppi", "fps", "rpm",
    # Temperature
    "°c", "°f", "celsius", "fahrenheit",
    # Electrical
    "w", "watt", "watts", "v", "volt", "volts", "mah", "ah", "amp", "amps",
    "kwh",
]

GLOBAL_CURRENCIES = [
    "$", "€", "£", "₺", "¥", "₹", "₩", "₽", "zł", "kr",
    "usd", "eur", "gbp", "try", "tl", "jpy", "cny", "inr", "krw",
    "aud", "cad", "chf", "sek", "nok", "dkk", "pln", "brl", "mxn",
]

GLOBAL_QUANTITY_ABBR = [
    "pcs", "pc", "ea", "qty", "doz", "dozen", "pk", "pack", "ct", "count",
    "set", "pair", "pairs", "box", "carton", "roll", "sheet", "sheets",
    "bundle", "lot",
]

GLOBAL_COMMON_SHORTHAND = [
    "w/", "b/c", "approx", "misc", "incl", "excl", "max", "min",
    "vs", "etc", "avg", "est", "qty", "ref", "no.", "nr",
    "orig", "auth", "cert", "compat", "adj", "recond", "refurb",
]

GLOBAL_SIZE_TERMS = [
    "xs", "s", "m", "l", "xl", "xxl", "xxxl", "2xl", "3xl", "4xl", "5xl",
    "small", "medium", "large", "extra large",
    "one size", "os", "free size", "regular", "slim", "wide", "narrow",
    "petite", "plus", "tall", "short", "long",
    "us", "uk", "eu",
]

# Common English filler/connector words that should NOT be corrected
PROTECTED_COMMON_WORDS = [
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "shall", "may", "might", "can", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "with", "from", "by", "of", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over",
    "and", "or", "but", "not", "no", "yes", "if", "then", "than", "so",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "some", "any",
    "much", "many", "lot", "lots",
    "new", "old", "big", "small", "good", "best", "great", "top",
    "cheap", "budget", "premium", "pro", "lite", "ultra", "mini", "max",
    "free", "fast", "quick", "original", "genuine", "authentic",
    "buy", "get", "need", "want", "find", "show", "looking", "search",
    "price", "cost", "deal", "sale", "discount", "offer", "coupon",
    "review", "rating", "compare", "alternative", "similar",
    "black", "white", "red", "blue", "green", "yellow", "pink", "purple",
    "orange", "brown", "grey", "gray", "silver", "gold", "rose gold",
    "beige", "navy", "teal", "coral", "maroon", "ivory", "cream",
    "color", "colour",
]

GLOBAL_ECOMMERCE_TERMS = [
    "warranty", "guarantee", "refurbished", "renewed", "pre-owned", "used",
    "brand new", "sealed", "open box", "clearance", "limited edition",
    "bundle", "combo", "kit", "set", "accessory", "accessories",
    "replacement", "compatible", "universal", "generic", "oem",
    "shipping", "delivery", "express", "same day", "next day",
    "return", "refund", "exchange", "track", "order",
    "in stock", "out of stock", "preorder", "pre-order", "backorder",
    "bestseller", "trending", "popular", "recommended", "featured",
    "unisex", "men", "women", "kids", "boys", "girls", "baby", "toddler",
    "adult", "teen", "junior", "senior",
]


# ======================================================================
# ELECTRONICS
# ======================================================================

def build_electronics() -> Category:
    cat = Category(name="electronics")

    cat.brands = {
        # Smartphone / Mobile
        "apple": ["iphone", "ipad", "macbook", "imac", "mac mini", "mac studio", "mac pro",
                   "airpods", "apple watch", "homepod", "apple tv", "vision pro", "magsafe", "airtag"],
        "samsung": ["galaxy", "galaxy s", "galaxy a", "galaxy m", "galaxy z", "galaxy tab",
                     "galaxy watch", "galaxy buds", "odyssey", "qled", "neo qled", "the frame",
                     "freestyle", "smart monitor", "galaxy ring"],
        "google": ["pixel", "pixel pro", "pixel a", "pixel fold", "pixel tablet",
                    "pixel watch", "pixel buds", "nest", "chromecast", "fitbit"],
        "xiaomi": ["redmi", "poco", "mi", "redmi note", "poco x", "poco f", "poco m",
                    "mi band", "mi tv", "mi box", "mi stick", "robot vacuum", "14 ultra"],
        "huawei": ["mate", "pura", "p series", "nova", "watch", "freebuds", "matepad", "matebook"],
        "oneplus": ["nord", "ace", "open", "watch", "buds", "pad", "12", "13"],
        "oppo": ["find", "reno", "a series", "enco", "pad", "find x", "find n"],
        "vivo": ["x series", "v series", "y series", "tws", "iqoo"],
        "realme": ["gt", "narzo", "c series", "buds", "pad", "book"],
        "motorola": ["moto g", "moto e", "razr", "edge", "thinkphone"],
        "sony": ["xperia", "wh-1000xm", "wf-1000xm", "inzone", "bravia", "playstation",
                  "alpha", "cybershot", "walkman", "pulse 3d"],
        "nothing": ["phone", "ear", "cmf"],
        "honor": ["magic", "x", "pad"],
        "tecno": ["camon", "spark", "pova", "phantom"],
        "infinix": ["note", "hot", "zero", "smart"],

        # Laptop / PC
        "dell": ["xps", "inspiron", "latitude", "precision", "alienware", "vostro",
                  "optiplex", "g series"],
        "hp": ["spectre", "envy", "pavilion", "omen", "victus", "elitebook", "probook",
                "zbook", "dragonfly", "laserjet", "deskjet", "inktank"],
        "lenovo": ["thinkpad", "ideapad", "legion", "loq", "yoga", "thinkcentre",
                    "thinkstation", "tab", "thinkbook"],
        "asus": ["rog", "tuf", "zenbook", "vivobook", "proart", "expertbook",
                  "rog strix", "rog zephyrus", "rog ally", "prime", "tuf gaming"],
        "acer": ["predator", "nitro", "aspire", "swift", "spin", "chromebook", "travelmate"],
        "msi": ["stealth", "raider", "vector", "titan", "crosshair", "katana", "pulse",
                 "cyborg", "prestige", "modern", "summit", "suprim", "ventus", "gaming x"],
        "razer": ["blade", "book", "kraken", "blackwidow", "deathadder", "viper",
                   "basilisk", "huntsman", "barracuda", "hammerhead"],
        "microsoft": ["surface", "surface pro", "surface laptop", "surface go",
                       "surface book", "surface studio", "xbox", "xbox series", "intellimouse"],
        "gigabyte": ["aorus", "aero", "eagle", "gaming oc", "windforce"],
        "framework": ["laptop 13", "laptop 16"],

        # Components
        "nvidia": ["geforce", "rtx", "gtx", "quadro", "tesla", "shield"],
        "amd": ["ryzen", "radeon", "threadripper", "epyc", "athlon", "freesync"],
        "intel": ["core", "xeon", "celeron", "pentium", "arc", "nuc", "evo", "core ultra"],
        "western digital": ["wd blue", "wd black", "wd red", "wd green", "wd purple",
                             "elements", "my passport", "sn850", "sn770", "sn580"],
        "seagate": ["barracuda", "ironwolf", "skyhawk", "firecuda", "expansion", "one touch"],
        "sandisk": ["extreme", "ultra", "cruzer", "clip"],
        "kingston": ["fury", "beast", "renegade", "canvas", "datatraveler", "nv2"],
        "crucial": ["mx500", "bx500", "p3", "p5", "ballistix", "pro"],
        "corsair": ["vengeance", "dominator", "force", "mp600", "k70", "k100",
                     "harpoon", "void", "virtuoso", "hs80", "icue", "elgato"],
        "g.skill": ["trident", "ripjaws", "flare"],

        # Peripherals
        "logitech": ["g pro", "g502", "mx master", "mx keys", "g915", "g733",
                      "streamcam", "c920", "c922", "g29", "g923", "pebble", "lift", "mx anywhere"],
        "steelseries": ["arctis", "apex", "rival", "aerox", "prime", "qck"],
        "hyperx": ["cloud", "pulsefire", "alloy", "quadcast", "fury"],
        "anker": ["soundcore", "eufy", "nebula", "powercore", "powerport", "nano",
                   "maggo", "roav"],
        "baseus": ["blade", "gan", "encok"],
        "ugreen": ["nexode", "hitune", "revodok"],
        "belkin": ["boostcharge", "soundform"],
        "tp-link": ["archer", "deco", "tapo", "kasa", "omada"],
        "ubiquiti": ["unifi", "amplifi", "dream machine"],
        "netgear": ["nighthawk", "orbi"],

        # Audio
        "bose": ["quietcomfort", "soundlink", "soundbar", "smart soundbar", "nc700", "ultra open"],
        "jbl": ["flip", "charge", "pulse", "tune", "live", "reflect", "partybox",
                 "go", "clip", "quantum", "xtreme"],
        "sennheiser": ["momentum", "hd", "ie", "gsp", "accentum"],
        "audio-technica": ["ath-m50", "ath-m40", "at2020", "at2040"],
        "marshall": ["major", "minor", "emberton", "stanmore", "acton", "woburn"],
        "sonos": ["one", "beam", "arc", "move", "roam", "era", "sub"],

        # Display
        "lg": ["ultragear", "ultrafine", "oled", "nanocell", "gram", "cinebeam",
                "tone free", "xboom", "ergo"],
        "benq": ["zowie", "mobiuz", "pd", "ew", "tk", "th"],
        "viewsonic": ["elite", "vx", "xg", "m1", "m2"],

        # Camera
        "canon": ["eos", "pixma", "powershot", "imageclass", "eos r", "eos m"],
        "nikon": ["z series", "d series", "coolpix"],
        "fujifilm": ["instax", "x series", "gfx", "x-t5", "x-s20"],
        "gopro": ["hero", "max"],
        "dji": ["mavic", "mini", "air", "avata", "osmo", "ronin", "pocket", "action"],
        "epson": ["ecotank", "workforce", "home cinema"],

        # Gaming
        "nintendo": ["switch", "switch oled", "switch lite", "joy-con", "pro controller"],
        "valve": ["steam deck", "index"],
        "playstation": ["ps5", "ps4", "dualsense", "pulse 3d"],
    }

    cat.products = [
        "smartphone", "phone", "tablet", "smartwatch", "watch", "fitness tracker", "smart band",
        "earbuds", "earphones", "headphones", "headset", "tws", "iem",
        "charger", "cable", "case", "screen protector", "power bank", "wireless charger",
        "laptop", "notebook", "ultrabook", "chromebook", "desktop", "pc", "workstation",
        "all-in-one", "mini pc",
        "monitor", "keyboard", "mouse", "mousepad", "webcam", "microphone", "speaker",
        "soundbar", "printer", "scanner", "projector", "usb hub", "docking station",
        "gpu", "graphics card", "cpu", "processor", "motherboard", "ram", "memory",
        "ssd", "hdd", "hard drive", "nvme", "psu", "power supply",
        "cooler", "cpu cooler", "liquid cooler", "aio", "case fan", "thermal paste",
        "gaming laptop", "gaming pc", "gaming mouse", "gaming keyboard", "gaming headset",
        "gaming monitor", "gaming chair",
        "controller", "gamepad", "console", "vr headset",
        "router", "modem", "wifi extender", "mesh system",
        "smart bulb", "smart plug", "security camera", "video doorbell", "robot vacuum",
        "camera", "mirrorless camera", "dslr", "action camera", "drone", "gimbal",
        "lens", "tripod", "ring light", "memory card", "sd card", "microsd",
        "adapter", "converter", "laptop stand", "monitor arm",
    ]

    cat.model_patterns = {
        "iphone": ["15", "15 pro", "15 pro max", "15 plus", "14", "14 pro", "14 pro max",
                    "13", "13 mini", "12", "11", "xr", "se", "16", "16 pro", "16 pro max"],
        "galaxy": ["s24", "s24+", "s24 ultra", "s23", "s23 fe", "s22", "z fold5", "z fold6",
                    "z flip5", "z flip6", "a55", "a35", "a25", "a15"],
        "pixel": ["8", "8 pro", "8a", "9", "9 pro", "7", "7 pro", "7a", "fold"],
        "redmi": ["note 13", "note 13 pro", "note 13 pro+", "13c", "12", "note 12"],
        "macbook": ["air m1", "air m2", "air m3", "pro m1", "pro m2", "pro m3",
                     "pro m3 max", "pro m3 pro", "pro 14", "pro 16"],
        "thinkpad": ["x1 carbon", "x1 yoga", "t14", "e14", "e15", "l14", "z13", "p16"],
        "rog": ["zephyrus g14", "zephyrus g16", "strix g16", "strix scar 18", "flow x13", "ally"],
        "legion": ["pro 7i", "pro 5i", "slim 7", "slim 5", "go"],
        "rtx": ["5090", "5080", "5070 ti", "5070", "5060", "4090", "4080", "4070 ti super",
                 "4070 super", "4070", "4060 ti", "4060", "3060", "3050"],
        "rx": ["9070 xt", "9070", "7900 xtx", "7900 xt", "7800 xt", "7700 xt", "7600"],
        "ryzen": ["9 9950x", "9 9900x", "7 9700x", "5 9600x", "9 7950x3d", "7 7800x3d",
                   "5 7600x", "5 5600"],
        "core": ["i9-14900k", "i7-14700k", "i5-14600k", "ultra 9", "ultra 7", "ultra 5",
                  "i9-13900k", "i7-13700k", "i5-13400f"],
        "ipad": ["pro 12.9", "pro 11", "air", "mini", "10th gen"],
        "playstation": ["5", "5 slim", "5 pro", "4"],
        "xbox": ["series x", "series s"],
        "switch": ["oled", "lite", "2"],
        "steam": ["deck", "deck oled"],
    }

    cat.abbreviations = {
        "gpu": "graphics processing unit", "cpu": "central processing unit",
        "ram": "random access memory", "ssd": "solid state drive",
        "hdd": "hard disk drive", "nvme": "non-volatile memory express",
        "psu": "power supply unit", "aio": "all in one",
        "rgb": "red green blue", "fps": "frames per second",
        "tb": "terabyte", "gb": "gigabyte", "mb": "megabyte",
        "usb": "universal serial bus", "hdmi": "high definition multimedia interface",
        "dp": "displayport", "tws": "true wireless stereo",
        "anc": "active noise cancellation", "ips": "in-plane switching",
        "oled": "organic light emitting diode", "qled": "quantum dot led",
        "lcd": "liquid crystal display", "led": "light emitting diode",
        "wifi": "wireless fidelity", "bt": "bluetooth",
        "nfc": "near field communication", "os": "operating system",
        "4k": "3840x2160", "1080p": "1920x1080", "1440p": "2560x1440",
    }

    cat.typo_mappings = {
        "iphnoe": "iphone", "iphoen": "iphone", "iphne": "iphone", "ihpone": "iphone",
        "iphon": "iphone", "ipone": "iphone", "iponhe": "iphone",
        "samsng": "samsung", "samsnug": "samsung", "samsug": "samsung", "sumsung": "samsung",
        "samsunng": "samsung", "samaung": "samsung", "smasung": "samsung",
        "macbok": "macbook", "mackbook": "macbook", "mcbook": "macbook", "macbbok": "macbook",
        "airpod": "airpods", "airpds": "airpods", "aripods": "airpods",
        "nvidai": "nvidia", "nvidea": "nvidia", "nividia": "nvidia", "nvdia": "nvidia",
        "geforc": "geforce", "gefroce": "geforce", "gerforce": "geforce", "gefore": "geforce",
        "lenvo": "lenovo", "lenvoo": "lenovo", "lenova": "lenovo", "lnovo": "lenovo",
        "thinkpd": "thinkpad", "thnikpad": "thinkpad", "thinpad": "thinkpad",
        "logitec": "logitech", "logitek": "logitech", "logitch": "logitech",
        "corsiar": "corsair", "corsar": "corsair", "crosair": "corsair",
        "razr": "razer", "razzer": "razer", "raserr": "razer",
        "surfce": "surface", "surfac": "surface", "srface": "surface",
        "blutooth": "bluetooth", "bluethooth": "bluetooth", "bluetoth": "bluetooth",
        "wireles": "wireless", "wirelss": "wireless", "wirless": "wireless",
        "labtop": "laptop", "laptpp": "laptop", "laptp": "laptop", "latop": "laptop",
        "moniter": "monitor", "monitr": "monitor", "mointor": "monitor",
        "keybord": "keyboard", "keybaord": "keyboard", "keybard": "keyboard", "kyboard": "keyboard",
        "headphons": "headphones", "headfones": "headphones", "hedphones": "headphones",
        "speakr": "speaker", "speeker": "speaker", "speker": "speaker",
        "smartfone": "smartphone", "samartphone": "smartphone",
        "tablt": "tablet", "tablat": "tablet", "tblet": "tablet",
        "earbusd": "earbuds", "earbudd": "earbuds", "erbds": "earbuds",
        "webcma": "webcam", "wbcam": "webcam",
        "procesor": "processor", "processer": "processor",
        "memroy": "memory", "memmory": "memory", "memry": "memory",
        "graphis": "graphics", "graphix": "graphics", "grahics": "graphics",
        "gamign": "gaming", "gaiming": "gaming", "gmng": "gaming",
        "chargr": "charger", "chager": "charger", "chrger": "charger",
        "microphne": "microphone", "micropone": "microphone",
        "nintedo": "nintendo", "nintndo": "nintendo",
        "playstaton": "playstation", "playstaion": "playstation",
        "xbok": "xbox", "xbos": "xbox",
    }

    return cat


# ======================================================================
# FASHION
# ======================================================================

def build_fashion() -> Category:
    cat = Category(name="fashion")

    cat.brands = {
        "nike": ["air max", "air force", "air jordan", "dunk", "blazer", "cortez",
                  "pegasus", "vomero", "flyknit", "react", "zoom", "tech fleece", "acg"],
        "adidas": ["ultraboost", "nmd", "stan smith", "superstar", "gazelle", "samba",
                    "yeezy", "forum", "campus", "ozweego", "terrex", "predator", "copa"],
        "puma": ["suede", "rs-x", "cali", "future rider", "roma", "clyde", "mayze"],
        "new balance": ["990", "574", "550", "327", "530", "2002r", "1906r", "fresh foam"],
        "reebok": ["classic", "club c", "nano", "floatride", "question"],
        "converse": ["chuck taylor", "chuck 70", "one star", "run star"],
        "vans": ["old skool", "sk8-hi", "authentic", "era", "slip-on", "ultrarange"],
        "asics": ["gel-kayano", "gel-nimbus", "gel-1130", "gt-2000", "novablast"],
        "hoka": ["bondi", "clifton", "speedgoat", "mach", "arahi", "rincon"],
        "on": ["cloud", "cloudmonster", "cloudswift", "cloudrunner", "roger"],
        "saucony": ["triumph", "ride", "guide", "endorphin", "peregrine"],
        "zara": ["basic", "premium", "studio", "srpls", "origins"],
        "h&m": ["divided", "trend", "premium quality", "conscious"],
        "uniqlo": ["heattech", "airism", "ut", "jw anderson", "u"],
        "shein": [],
        "temu": [],
        "gap": ["gap", "old navy", "banana republic", "athleta"],
        "levi's": ["501", "502", "505", "511", "512", "514", "517", "711", "720"],
        "gucci": ["ace", "dionysus", "marmont", "gg", "horsebit"],
        "louis vuitton": ["neverfull", "speedy", "alma", "keepall", "pochette"],
        "chanel": ["classic flap", "boy", "coco", "gabrielle", "2.55"],
        "prada": ["re-nylon", "galleria", "saffiano"],
        "ralph lauren": ["polo", "purple label", "rlx", "lauren"],
        "tommy hilfiger": ["essential", "signature", "sport", "denim"],
        "calvin klein": ["ck one", "eternity", "obsessed"],
        "the north face": ["nuptse", "thermoball", "mcmurdo", "1996", "borealis"],
        "patagonia": ["nano puff", "better sweater", "torrentshell", "baggies"],
        "columbia": ["omni-heat", "silver ridge", "bugaboo"],
        "under armour": ["hovr", "charged", "ua", "project rock", "coldgear", "heatgear"],
        "lululemon": ["align", "wunder under", "define", "scuba", "abc"],
        "birkenstock": ["arizona", "boston", "gizeh", "madrid", "kyoto"],
        "dr. martens": ["1460", "1461", "jadon", "audrick", "platform"],
        "timberland": ["6-inch", "euro sprint", "greyfield"],
        "crocs": ["classic clog", "bayaband", "literide", "bistro"],
        "skechers": ["go walk", "d'lites", "arch fit", "max cushioning"],
        "fila": ["disruptor", "ray tracer", "grant hill"],
        "champion": ["reverse weave", "powerblend", "classic"],
    }

    cat.products = [
        "sneakers", "running shoes", "basketball shoes", "tennis shoes", "hiking boots",
        "dress shoes", "loafers", "sandals", "flip flops", "slides", "slippers",
        "boots", "ankle boots", "chelsea boots", "combat boots", "rain boots",
        "t-shirt", "tee", "shirt", "blouse", "polo", "tank top", "crop top", "henley",
        "hoodie", "sweatshirt", "sweater", "cardigan", "pullover", "fleece",
        "jacket", "coat", "blazer", "bomber jacket", "puffer jacket", "windbreaker",
        "parka", "trench coat", "denim jacket", "leather jacket", "vest",
        "jeans", "pants", "trousers", "chinos", "joggers", "sweatpants", "cargo pants",
        "shorts", "skirt", "dress", "jumpsuit", "romper", "leggings", "tights",
        "suit", "tuxedo", "formal wear",
        "underwear", "boxers", "briefs", "bra", "sports bra", "socks", "ankle socks",
        "hat", "cap", "beanie", "bucket hat", "snapback", "visor",
        "scarf", "gloves", "belt", "tie", "bow tie", "suspenders",
        "bag", "backpack", "tote", "crossbody", "clutch", "wallet", "purse",
        "handbag", "duffel bag", "messenger bag", "fanny pack",
        "watch", "sunglasses", "jewelry", "necklace", "bracelet", "ring", "earrings",
    ]

    cat.typo_mappings = {
        "addidas": "adidas", "adiddas": "adidas", "addidas": "adidas", "adidias": "adidas",
        "nikee": "nike", "nkie": "nike", "nikey": "nike",
        "pumma": "puma", "pmua": "puma",
        "sketchers": "skechers", "skecherss": "skechers",
        "convrse": "converse", "convrese": "converse",
        "levis": "levi's", "leviss": "levi's",
        "guucci": "gucci", "guccci": "gucci",
        "pradda": "prada", "prda": "prada",
        "shien": "shein", "shien": "shein", "sheinn": "shein",
        "zaara": "zara", "zra": "zara",
        "uniqloo": "uniqlo", "uniklo": "uniqlo",
        "birkenstock": "birkenstock", "birkenstok": "birkenstock",
        "timberlnd": "timberland", "timberlan": "timberland",
        "newbalance": "new balance", "new ballance": "new balance",
        "northface": "the north face", "north fase": "the north face",
        "underarmor": "under armour", "under armor": "under armour",
        "lululmon": "lululemon", "lulemon": "lululemon",
        "calvinklein": "calvin klein", "calvin clein": "calvin klein",
        "tommyhilfiger": "tommy hilfiger", "tommy hillfiger": "tommy hilfiger",
        "snekers": "sneakers", "sneekers": "sneakers", "snakers": "sneakers",
        "hodie": "hoodie", "hoddie": "hoodie", "hooyde": "hoodie",
        "jackt": "jacket", "jaket": "jacket",
        "tshirt": "t-shirt", "tshrt": "t-shirt",
        "jenas": "jeans", "jeens": "jeans",
        "sweatr": "sweater", "sweeter": "sweater",
        "leggins": "leggings", "leging": "leggings",
        "backpak": "backpack", "bakpack": "backpack",
        "sandels": "sandals", "sandles": "sandals",
        "sunglases": "sunglasses", "sunglass": "sunglasses",
        "jewlry": "jewelry", "jewlery": "jewelry", "jwelry": "jewelry",
        "necklase": "necklace", "neckles": "necklace",
        "braclet": "bracelet", "bracelat": "bracelet",
    }

    cat.model_patterns = {
        "air max": ["90", "95", "97", "1", "270", "720", "plus", "dn", "tn"],
        "air jordan": ["1", "3", "4", "5", "6", "11", "12", "13", "retro", "low", "mid", "high"],
        "dunk": ["low", "high", "sb", "retro"],
        "ultraboost": ["22", "23", "light", "1.0", "5.0"],
        "yeezy": ["350", "500", "700", "slide", "foam runner"],
        "990": ["v3", "v4", "v5", "v6"],
        "550": ["white green", "white grey", "white red", "white navy"],
        "1460": ["smooth", "nappa", "wintergrip", "bex"],
        "classic clog": ["", "lined", "platform", "tie dye"],
    }

    cat.common_terms = GLOBAL_SIZE_TERMS.copy()
    cat.units = ["us", "uk", "eu", "cm"]

    return cat


# ======================================================================
# BEAUTY & PERSONAL CARE
# ======================================================================

def build_beauty() -> Category:
    cat = Category(name="beauty")

    cat.brands = {
        "mac": ["studio fix", "ruby woo", "prep + prime", "fix+", "strobe cream"],
        "nars": ["radiant", "orgasm", "sheer glow", "creamy concealer"],
        "maybelline": ["fit me", "superstay", "lash sensational", "sky high", "instant age rewind"],
        "l'oreal": ["true match", "voluminous", "revitalift", "elvive", "paris"],
        "nyx": ["butter gloss", "soft matte", "born to glow", "can't stop won't stop"],
        "clinique": ["dramatically different", "moisture surge", "even better", "happy"],
        "estee lauder": ["double wear", "advanced night repair", "pure color"],
        "charlotte tilbury": ["pillow talk", "flawless filter", "magic cream", "airbrush"],
        "fenty beauty": ["pro filt'r", "gloss bomb", "match stix", "eaze drop"],
        "rare beauty": ["soft pinch", "positive light", "always an optimist"],
        "the ordinary": ["niacinamide", "hyaluronic acid", "retinol", "aha bha", "glycolic acid"],
        "cerave": ["moisturizing cream", "foaming cleanser", "hydrating cleanser", "sa cleanser",
                    "pm moisturizer", "am moisturizer", "healing ointment"],
        "la roche-posay": ["effaclar", "toleriane", "anthelios", "cicaplast", "hyalu b5"],
        "neutrogena": ["hydro boost", "rapid wrinkle repair", "ultra sheer"],
        "olaplex": ["no. 3", "no. 4", "no. 5", "no. 7", "no. 8"],
        "dyson": ["airwrap", "supersonic", "corrale", "airstrait"],
        "ghd": ["platinum+", "gold", "helios", "curve", "max"],
        "revlon": ["colorstay", "super lustrous", "photoready"],
        "urban decay": ["naked", "all nighter", "eyeshadow primer potion"],
        "too faced": ["born this way", "better than sex", "sweet peach", "lip injection"],
        "benefit": ["brow", "hoola", "porefessional", "they're real"],
        "tatcha": ["dewy skin", "water cream", "silk canvas", "rice wash"],
        "drunk elephant": ["protini", "baby facial", "lala retro", "c-firma"],
        "glossier": ["boy brow", "cloud paint", "milky jelly", "you", "balm dotcom"],
        "paula's choice": ["bha exfoliant", "retinol", "vitamin c", "niacinamide"],
        "kiehl's": ["ultra facial", "midnight recovery", "calendula"],
        "dove": ["beauty bar", "body wash", "deodorant", "shampoo"],
        "nivea": ["creme", "soft", "sun", "men"],
        "bath & body works": ["body mist", "shower gel", "body cream", "candle"],
    }

    cat.products = [
        "lipstick", "lip gloss", "lip liner", "lip balm", "lip stain", "lip oil",
        "foundation", "concealer", "powder", "setting powder", "primer", "bb cream",
        "cc cream", "tinted moisturizer", "contour", "highlighter", "bronzer", "blush",
        "eyeshadow", "eyeshadow palette", "eyeliner", "mascara", "eyebrow pencil",
        "lashes", "false lashes", "lash glue",
        "setting spray", "makeup remover", "micellar water", "cleansing balm",
        "cleanser", "face wash", "toner", "serum", "moisturizer", "night cream",
        "eye cream", "face mask", "sheet mask", "peel", "exfoliant", "scrub",
        "sunscreen", "spf", "body lotion", "body butter", "hand cream",
        "shampoo", "conditioner", "hair mask", "hair oil", "leave-in conditioner",
        "hair spray", "hair gel", "dry shampoo", "heat protectant",
        "hair dryer", "flat iron", "curling iron", "hair straightener",
        "perfume", "cologne", "fragrance", "body mist", "eau de toilette", "eau de parfum",
        "deodorant", "antiperspirant",
        "nail polish", "gel polish", "nail art", "nail file", "cuticle oil",
        "razor", "shaving cream", "aftershave", "trimmer", "electric razor",
        "toothbrush", "toothpaste", "mouthwash", "floss", "whitening strips",
        "makeup bag", "makeup brush", "beauty blender", "mirror",
    ]

    cat.typo_mappings = {
        "maybelline": "maybelline", "maybeline": "maybelline", "maybellne": "maybelline",
        "loreal": "l'oreal", "l'oreal": "l'oreal", "lorel": "l'oreal",
        "clinque": "clinique", "cliniqe": "clinique",
        "cerave": "cerave", "cerav": "cerave", "cereve": "cerave",
        "olapex": "olaplex", "olapelx": "olaplex",
        "dysen": "dyson", "dyosn": "dyson",
        "ghdd": "ghd",
        "ordnary": "ordinary", "ordinry": "ordinary",
        "feenty": "fenty", "fenty beuty": "fenty beauty",
        "charlote tilbury": "charlotte tilbury", "charlotte tilbry": "charlotte tilbury",
        "estee laudr": "estee lauder", "este lauder": "estee lauder",
        "neutragena": "neutrogena", "neutrogna": "neutrogena",
        "la roche posay": "la roche-posay", "laroche posay": "la roche-posay",
        "moisturzer": "moisturizer", "moisturisr": "moisturizer",
        "foundaton": "foundation", "fondation": "foundation",
        "mascra": "mascara", "masacara": "mascara",
        "shampo": "shampoo", "shamppo": "shampoo", "shmpoo": "shampoo",
        "condtioner": "conditioner", "conditionr": "conditioner",
        "sunscren": "sunscreen", "sunscrean": "sunscreen",
        "conceler": "concealer", "concealr": "concealer",
        "perfum": "perfume", "parfume": "perfume",
        "lipstik": "lipstick", "lipstic": "lipstick",
        "eyliner": "eyeliner", "eye liner": "eyeliner",
    }

    cat.units = ["ml", "oz", "fl oz", "g", "spf"]

    return cat


# ======================================================================
# HOME & KITCHEN
# ======================================================================

def build_home() -> Category:
    cat = Category(name="home")

    cat.brands = {
        "ikea": ["kallax", "malm", "billy", "hemnes", "poang", "lack", "besta", "alex",
                  "raskog", "detolf", "markus"],
        "dyson": ["v15", "v12", "v8", "v7", "big ball", "purifier", "humidifier",
                   "hot+cool", "pure cool"],
        "roomba": ["j7", "i7", "s9", "combo", "e5"],
        "irobot": ["braava", "roomba"],
        "shark": ["navigator", "rotator", "vertex", "ionflex", "wandvac"],
        "kitchenaid": ["artisan", "classic", "professional", "hand mixer", "food processor"],
        "instant pot": ["duo", "duo plus", "ultra", "pro", "max"],
        "ninja": ["foodi", "air fryer", "blender", "creami", "thirsti", "woodfire"],
        "cuisinart": ["food processor", "coffee maker", "toaster oven"],
        "breville": ["barista express", "smart oven", "juicer", "bambino"],
        "keurig": ["k-slim", "k-elite", "k-supreme", "k-duo", "k-cafe"],
        "nespresso": ["vertuo", "original", "lattissima", "essenza"],
        "philips": ["airfryer", "sonicare", "hue", "norelco", "wake-up light"],
        "bosch": ["dishwasher", "washing machine", "dryer", "oven"],
        "samsung": ["bespoke", "family hub"],
        "lg": ["thinq", "styler"],
        "whirlpool": ["cabrio", "duet"],
        "vitamix": ["explorian", "a3500", "e310"],
        "crock-pot": ["slow cooker", "express"],
        "rubbermaid": ["brilliance", "takealongs"],
        "yeti": ["rambler", "tundra", "hopper", "loadout"],
        "stanley": ["quencher", "adventure", "classic"],
        "tervis": ["tumbler", "water bottle"],
        "simplehuman": ["sensor can", "dish rack", "mirror"],
        "ring": ["doorbell", "camera", "alarm", "floodlight"],
        "nest": ["thermostat", "cam", "doorbell", "wifi"],
        "ecobee": ["thermostat", "smartsensor"],
        "hisense": ["tv", "mini fridge", "dehumidifier"],
        "tcl": ["roku tv", "google tv"],
    }

    cat.products = [
        "vacuum", "vacuum cleaner", "robot vacuum", "handheld vacuum", "steam mop",
        "air purifier", "humidifier", "dehumidifier", "fan", "space heater",
        "air conditioner", "portable ac",
        "blender", "mixer", "food processor", "juicer", "hand blender",
        "coffee maker", "espresso machine", "french press", "kettle", "electric kettle",
        "toaster", "toaster oven", "microwave", "oven", "air fryer",
        "slow cooker", "pressure cooker", "rice cooker", "instant pot",
        "dishwasher", "washing machine", "dryer",
        "refrigerator", "fridge", "mini fridge", "freezer",
        "pan", "pot", "skillet", "wok", "baking sheet", "cutting board",
        "knife set", "kitchen scale", "measuring cups", "spatula", "tongs",
        "water bottle", "tumbler", "mug", "thermos", "lunch box",
        "storage container", "trash can", "paper towel holder",
        "bed frame", "mattress", "pillow", "duvet", "blanket", "sheets",
        "couch", "sofa", "recliner", "armchair", "ottoman",
        "desk", "office chair", "bookshelf", "nightstand", "dresser",
        "tv stand", "coffee table", "dining table", "bar stool",
        "curtains", "blinds", "rug", "carpet", "lamp", "light bulb",
        "candle", "diffuser", "wall art", "picture frame", "mirror",
        "doorbell", "thermostat", "smart lock", "security system",
    ]

    cat.typo_mappings = {
        "ikea": "ikea", "ikae": "ikea",
        "dyosn": "dyson", "dysen": "dyson",
        "roomba": "roomba", "romba": "roomba", "rumba": "roomba",
        "kitchneaid": "kitchenaid", "kitcheniad": "kitchenaid",
        "instantpot": "instant pot", "instan pot": "instant pot",
        "nnja": "ninja", "ninaj": "ninja",
        "kurig": "keurig", "keuirg": "keurig",
        "nespreso": "nespresso", "nespreso": "nespresso",
        "vaccum": "vacuum", "vacum": "vacuum", "vaccuum": "vacuum", "vacuume": "vacuum",
        "blendr": "blender", "blendor": "blender",
        "toastr": "toaster", "toster": "toaster",
        "matress": "mattress", "matres": "mattress", "mattres": "mattress",
        "pillw": "pillow", "pillow": "pillow",
        "curtians": "curtains", "curtans": "curtains",
        "refridgerator": "refrigerator", "refridgrator": "refrigerator",
        "microwve": "microwave", "microave": "microwave",
        "dishwashr": "dishwasher", "dishwsher": "dishwasher",
        "humidifer": "humidifier", "humidifyer": "humidifier",
    }

    return cat


# ======================================================================
# SPORTS & OUTDOORS
# ======================================================================

def build_sports() -> Category:
    cat = Category(name="sports")

    cat.brands = {
        "nike": ["dri-fit", "pro", "aeroswift"],
        "adidas": ["aeroready", "terrex", "predator", "copa"],
        "under armour": ["hovr", "charged", "project rock", "coldgear", "heatgear"],
        "the north face": ["nuptse", "thermoball", "mcmurdo", "borealis"],
        "patagonia": ["nano puff", "better sweater", "torrentshell"],
        "columbia": ["omni-heat", "silver ridge", "bugaboo"],
        "yeti": ["rambler", "tundra", "hopper"],
        "coleman": ["tent", "cooler", "lantern", "stove"],
        "osprey": ["atmos", "exos", "talon", "manta", "sirrus"],
        "salomon": ["speedcross", "x ultra", "sense ride", "xt-6"],
        "garmin": ["forerunner", "fenix", "venu", "instinct", "edge", "inreach"],
        "fitbit": ["charge", "versa", "sense", "luxe", "inspire"],
        "peloton": ["bike", "tread", "guide", "row"],
        "bowflex": ["selecttech", "max trainer", "velocore"],
        "rogue": ["echo bike", "barbell", "rack", "bumper plates"],
        "theragun": ["pro", "elite", "prime", "mini", "relief"],
        "hydro flask": ["wide mouth", "standard mouth", "coffee mug"],
        "camelbak": ["podium", "eddy", "chute"],
        "black diamond": ["headlamp", "carabiner", "harness"],
        "rei": ["co-op", "trailbreak"],
        "decathlon": ["quechua", "domyos", "kalenji", "btwin"],
        "wilson": ["blade", "clash", "pro staff", "ultra"],
        "callaway": ["epic", "rogue", "paradym", "big bertha"],
        "titleist": ["pro v1", "tsi", "t-series"],
        "spalding": ["nba", "tf"],
        "everlast": ["pro style", "powerlock", "elite"],
    }

    cat.products = [
        "running shoes", "trail shoes", "hiking boots", "cleats",
        "yoga mat", "exercise mat", "resistance bands", "jump rope",
        "dumbbell", "kettlebell", "barbell", "weight plates", "bench press",
        "treadmill", "exercise bike", "elliptical", "rowing machine",
        "pull up bar", "squat rack", "power rack", "smith machine",
        "foam roller", "massage gun", "compression sleeves",
        "sports bra", "athletic shorts", "compression tights", "tank top",
        "swim goggles", "swim cap", "swimsuit", "wetsuit", "snorkel",
        "bicycle", "bike helmet", "bike lock", "cycling jersey",
        "tent", "sleeping bag", "camping stove", "cooler", "lantern",
        "backpack", "hiking backpack", "hydration pack", "day pack",
        "water bottle", "insulated bottle", "protein shaker",
        "fishing rod", "tackle box", "fishing reel",
        "golf club", "golf ball", "golf bag", "golf glove",
        "tennis racket", "tennis balls", "badminton racket",
        "basketball", "football", "soccer ball", "volleyball", "baseball",
        "skateboard", "longboard", "scooter", "roller skates",
        "ski jacket", "ski pants", "ski goggles", "snowboard",
        "kayak", "paddle board", "life jacket",
        "fitness tracker", "sports watch", "gps watch", "heart rate monitor",
    ]

    cat.typo_mappings = {
        "tredmill": "treadmill", "treadmil": "treadmill",
        "dumbel": "dumbbell", "dumbell": "dumbbell", "dumbell": "dumbbell",
        "kettelbell": "kettlebell", "kettlebal": "kettlebell",
        "yogamat": "yoga mat", "yga mat": "yoga mat",
        "bycicle": "bicycle", "bicylce": "bicycle",
        "tenis": "tennis", "teniss": "tennis",
        "basketbal": "basketball", "baskeball": "basketball",
        "footbal": "football", "fotball": "football",
        "skatbord": "skateboard", "skateborad": "skateboard",
        "garmin": "garmin", "garmn": "garmin",
        "fitbt": "fitbit", "fitbi": "fitbit",
        "salomn": "salomon", "soloman": "salomon",
        "colmbia": "columbia", "columba": "columbia",
        "ospreey": "osprey", "ospery": "osprey",
        "campng": "camping", "campin": "camping",
        "hikng": "hiking", "hking": "hiking",
        "runnng": "running", "runing": "running",
        "swimmin": "swimming", "swiming": "swimming",
    }

    return cat


# ======================================================================
# TOYS & GAMES
# ======================================================================

def build_toys() -> Category:
    cat = Category(name="toys")

    cat.brands = {
        "lego": ["city", "technic", "creator", "star wars", "harry potter", "friends",
                  "ninjago", "marvel", "dc", "architecture", "duplo", "icons"],
        "barbie": ["dreamhouse", "fashionistas", "color reveal", "extra"],
        "hot wheels": ["track builder", "monster trucks", "city", "id", "mario kart"],
        "nerf": ["elite 2.0", "ultra", "rival", "mega", "fortnite", "halo"],
        "hasbro": ["transformers", "g.i. joe", "monopoly", "play-doh", "furreal"],
        "mattel": ["uno", "scrabble", "pictionary", "matchbox"],
        "funko": ["pop", "pop vinyl", "mystery mini"],
        "playmobil": ["city life", "country", "pirates", "knights"],
        "fisher-price": ["laugh & learn", "little people", "imaginext"],
        "melissa & doug": ["puzzle", "play food", "art supplies"],
        "vtech": ["kidizoom", "toot-toot", "leapfrog"],
        "nintendo": ["mario", "zelda", "pokemon", "animal crossing", "kirby"],
        "pokemon": ["trading cards", "plush", "action figure"],
        "disney": ["frozen", "moana", "encanto", "princess", "pixar", "marvel"],
        "ravensburger": ["puzzle", "gravitrax", "memory"],
        "catan": ["base game", "expansion", "traders"],
        "magic the gathering": ["booster", "commander", "draft"],
        "monopoly": ["classic", "junior", "electronic banking"],
    }

    cat.products = [
        "action figure", "doll", "stuffed animal", "plush", "teddy bear",
        "building blocks", "building set", "construction set",
        "rc car", "remote control car", "rc drone", "rc helicopter",
        "board game", "card game", "strategy game", "party game",
        "puzzle", "jigsaw puzzle", "3d puzzle", "rubik's cube",
        "play set", "dollhouse", "toy kitchen", "play tent",
        "toy car", "toy truck", "train set", "play mat",
        "water gun", "foam dart gun", "toy sword",
        "art supplies", "crayons", "colored pencils", "markers", "paint set",
        "slime", "kinetic sand", "play-doh", "modeling clay",
        "science kit", "microscope", "telescope", "robot kit",
        "educational toy", "learning tablet", "interactive book",
        "baby toy", "rattle", "teething toy", "stacking toy",
        "outdoor toy", "swing set", "trampoline", "sandbox",
        "ride-on toy", "balance bike", "scooter", "tricycle",
        "trading cards", "booster pack", "starter deck",
        "video game", "game cartridge", "gaming accessory",
    ]

    cat.typo_mappings = {
        "lego": "lego", "leggo": "lego", "lgeo": "lego",
        "barbi": "barbie", "barbee": "barbie",
        "hotwheels": "hot wheels", "hot weels": "hot wheels",
        "nerff": "nerf", "nref": "nerf",
        "funco": "funko", "fnko": "funko",
        "pokemn": "pokemon", "pokimon": "pokemon", "pokmon": "pokemon",
        "monopoli": "monopoly", "monoply": "monopoly",
        "puzzel": "puzzle", "puzle": "puzzle",
        "tramploine": "trampoline", "trampolin": "trampoline",
        "playdoh": "play-doh", "playdough": "play-doh",
    }

    return cat


# ======================================================================
# AUTOMOTIVE
# ======================================================================

def build_automotive() -> Category:
    cat = Category(name="automotive")

    cat.brands = {
        "bosch": ["icon", "evolution", "envision"],
        "michelin": ["pilot sport", "primacy", "defender", "latitude", "crossclimate"],
        "bridgestone": ["potenza", "turanza", "blizzak", "ecopia"],
        "goodyear": ["eagle", "assurance", "wrangler"],
        "continental": ["extremecontact", "purecontact", "truecontact"],
        "castrol": ["edge", "gtx", "magnatec"],
        "mobil": ["1", "super", "delvac"],
        "shell": ["rotella", "pennzoil", "helix"],
        "3m": ["headlight restoration", "auto body repair", "adhesive"],
        "weathertech": ["floor mat", "cargo liner", "deflector"],
        "meguiar's": ["gold class", "ultimate", "hybrid ceramic"],
        "chemical guys": ["butter wet wax", "honeydew", "vss"],
        "thule": ["roof rack", "cargo box", "bike rack"],
        "garmin": ["dash cam", "drivemart", "dezl"],
        "pioneer": ["dmh", "avh", "deh", "sph"],
        "kenwood": ["excelon", "dpx", "dmx"],
        "denso": ["spark plug", "wiper", "air filter"],
        "k&n": ["air filter", "cold air intake", "oil filter"],
        "ngk": ["spark plug", "ignition coil"],
    }

    cat.products = [
        "tire", "tires", "all season tire", "winter tire", "summer tire",
        "brake pad", "brake rotor", "brake caliper", "brake fluid",
        "oil filter", "air filter", "cabin filter", "fuel filter",
        "motor oil", "synthetic oil", "transmission fluid", "coolant",
        "spark plug", "ignition coil", "alternator", "starter",
        "battery", "car battery", "jumper cables", "battery charger",
        "wiper blade", "headlight", "tail light", "fog light", "led bulb",
        "floor mat", "seat cover", "steering wheel cover", "sun shade",
        "dash cam", "backup camera", "gps navigator", "car stereo",
        "car charger", "phone mount", "bluetooth adapter",
        "car wash", "car wax", "polish", "clay bar", "detailing spray",
        "jack", "lug wrench", "torque wrench", "obd2 scanner",
        "roof rack", "cargo carrier", "bike rack", "trailer hitch",
        "bumper", "fender", "hood", "mirror", "door handle",
    ]

    cat.typo_mappings = {
        "michellin": "michelin", "michlin": "michelin",
        "bridgston": "bridgestone", "bridgston": "bridgestone",
        "goodyar": "goodyear", "goodyer": "goodyear",
        "castrl": "castrol", "castol": "castrol",
        "meguirs": "meguiar's", "meguiars": "meguiar's",
        "weathertec": "weathertech", "wethertec": "weathertech",
        "brakepads": "brake pads", "brak pad": "brake pad",
        "tiree": "tire", "tyer": "tire",
        "headlght": "headlight", "hedlight": "headlight",
        "stearing": "steering", "steeling": "steering",
        "battrey": "battery", "baterry": "battery",
    }

    return cat


# ======================================================================
# GROCERY & FOOD
# ======================================================================

def build_grocery() -> Category:
    cat = Category(name="grocery")

    cat.brands = {
        "organic": [],
        "keto": [],
        "vegan": [],
        "gluten-free": [],
        "nestle": ["nescafe", "kitkat", "maggi", "milo"],
        "kraft": ["mac & cheese", "heinz", "philadelphia"],
        "general mills": ["cheerios", "nature valley", "yoplait"],
        "kellogg's": ["frosted flakes", "corn flakes", "special k", "pringles"],
        "coca-cola": ["coke", "sprite", "fanta", "dasani", "minute maid"],
        "pepsi": ["mountain dew", "gatorade", "aquafina", "doritos", "lays"],
        "red bull": ["original", "sugar free", "edition"],
        "monster": ["energy", "ultra", "juice", "rehab"],
        "starbucks": ["pike place", "veranda", "dark roast", "cold brew"],
        "nespresso": ["vertuo", "original", "melozio", "stormio"],
        "optimum nutrition": ["gold standard", "serious mass", "bcaa"],
        "myprotein": ["impact whey", "creatine", "clear whey"],
        "garden of life": ["raw organic", "sport", "meal replacement"],
        "rxbar": ["chocolate sea salt", "blueberry", "peanut butter"],
        "clif": ["bar", "builder", "kid"],
        "kind": ["bar", "granola", "nut bar"],
    }

    cat.products = [
        "protein powder", "whey protein", "creatine", "pre-workout", "bcaa",
        "protein bar", "energy bar", "granola bar", "snack bar",
        "coffee beans", "ground coffee", "instant coffee", "coffee pods",
        "tea", "green tea", "herbal tea", "matcha",
        "energy drink", "sports drink", "electrolyte",
        "vitamins", "multivitamin", "vitamin d", "vitamin c", "omega 3",
        "fish oil", "probiotics", "collagen", "melatonin",
        "nut butter", "peanut butter", "almond butter",
        "olive oil", "coconut oil", "avocado oil",
        "rice", "pasta", "quinoa", "oats", "cereal",
        "flour", "sugar", "salt", "pepper", "spices",
        "sauce", "hot sauce", "soy sauce", "bbq sauce", "ketchup", "mustard",
        "snacks", "chips", "crackers", "popcorn", "nuts", "dried fruit",
        "chocolate", "candy", "gummy", "cookies",
        "water", "sparkling water", "coconut water", "juice",
    ]

    cat.typo_mappings = {
        "protien": "protein", "protine": "protein", "protin": "protein",
        "creatne": "creatine", "creatin": "creatine",
        "vitamn": "vitamin", "vitamen": "vitamin",
        "cofee": "coffee", "coffe": "coffee", "coffie": "coffee",
        "chocolat": "chocolate", "chocolte": "chocolate",
        "granla": "granola", "granloa": "granola",
        "orgainc": "organic", "orgnic": "organic",
        "glutn free": "gluten-free", "gluten fre": "gluten-free",
        "probiotic": "probiotics", "prbiotic": "probiotics",
        "electrolite": "electrolyte", "electrolyt": "electrolyte",
    }

    cat.units = ["g", "kg", "oz", "lbs", "ml", "l", "fl oz", "serving", "servings",
                 "calories", "cal", "kcal"]

    return cat


# ======================================================================
# BOOKS & MEDIA
# ======================================================================

def build_books() -> Category:
    cat = Category(name="books")

    cat.brands = {
        "kindle": ["paperwhite", "oasis", "scribe", "kids"],
        "audible": ["premium plus", "plus"],
        "amazon": ["echo", "fire tablet", "fire tv"],
        "kobo": ["libra", "clara", "sage", "elipsa"],
    }

    cat.products = [
        "paperback", "hardcover", "ebook", "audiobook", "e-reader",
        "textbook", "workbook", "notebook", "journal", "planner", "diary",
        "novel", "manga", "comic book", "graphic novel",
        "coloring book", "activity book", "sticker book",
        "cookbook", "recipe book",
        "self-help book", "business book", "biography",
        "children's book", "picture book", "board book",
        "dvd", "blu-ray", "vinyl record", "cd",
        "magazine", "subscription",
    ]

    cat.typo_mappings = {
        "paperbak": "paperback", "paprback": "paperback",
        "hardcovr": "hardcover", "hardcoer": "hardcover",
        "audiobk": "audiobook", "audiobok": "audiobook",
        "kindel": "kindle", "kindl": "kindle",
        "textbok": "textbook", "textbk": "textbook",
        "notbook": "notebook", "notebk": "notebook",
    }

    return cat


# ======================================================================
# OFFICE & SCHOOL SUPPLIES
# ======================================================================

def build_office() -> Category:
    cat = Category(name="office")

    cat.brands = {
        "hp": ["laserjet", "officejet", "deskjet"],
        "epson": ["ecotank", "workforce"],
        "brother": ["mfc", "hl"],
        "canon": ["pixma", "imageclass"],
        "dymo": ["labelwriter", "letratag"],
        "avery": ["labels", "dividers", "binder"],
        "post-it": ["notes", "flags", "tabs"],
        "sharpie": ["fine point", "ultra fine", "permanent", "s-gel"],
        "pilot": ["g2", "frixion", "precise"],
        "bic": ["cristal", "atlantis", "velocity"],
        "staedtler": ["mars", "noris", "triplus"],
        "moleskine": ["classic", "smart", "cahier"],
        "leuchtturm1917": ["a5", "bullet journal"],
    }

    cat.products = [
        "printer", "ink cartridge", "toner", "paper", "copy paper",
        "pen", "pencil", "mechanical pencil", "highlighter", "marker",
        "eraser", "ruler", "scissors", "tape", "glue", "stapler", "staples",
        "binder", "folder", "file cabinet", "organizer", "dividers",
        "envelope", "labels", "sticky notes", "index cards",
        "desk", "office chair", "desk lamp", "monitor stand",
        "whiteboard", "corkboard", "easel",
        "calculator", "shredder", "laminator",
        "backpack", "messenger bag", "briefcase", "laptop bag",
        "planner", "calendar", "notebook", "legal pad",
    ]

    cat.typo_mappings = {
        "printr": "printer", "priner": "printer",
        "cartrige": "cartridge", "cartrdge": "cartridge",
        "highliter": "highlighter", "highlghter": "highlighter",
        "stapler": "stapler", "staplr": "stapler",
        "scissers": "scissors", "scisors": "scissors",
        "calcualtor": "calculator", "calulator": "calculator",
        "shredr": "shredder", "shreder": "shredder",
    }

    return cat


# ======================================================================
# Category Registry
# ======================================================================

CATEGORY_BUILDERS = {
    "electronics": build_electronics,
    "fashion": build_fashion,
    "beauty": build_beauty,
    "home": build_home,
    "sports": build_sports,
    "toys": build_toys,
    "automotive": build_automotive,
    "grocery": build_grocery,
    "books": build_books,
    "office": build_office,
}


# ======================================================================
# Main Class
# ======================================================================

class UniversalVocab:
    """Universal e-commerce vocabulary across ALL categories."""

    def __init__(self):
        self.categories: Dict[str, Category] = {}

    def load_all(self):
        """Load all available categories."""
        for name in CATEGORY_BUILDERS:
            self.add_category(name)

    def add_category(self, name: str):
        if name in CATEGORY_BUILDERS:
            self.categories[name] = CATEGORY_BUILDERS[name]()

    def get_all_brands(self) -> Set[str]:
        brands = set()
        for cat in self.categories.values():
            brands.update(cat.brands.keys())
            for sub_brands in cat.brands.values():
                brands.update(sub_brands)
        return brands

    def get_all_brand_names_flat(self) -> Set[str]:
        """All brand names as a flat set (for RAG lookup)."""
        names = set()
        for cat in self.categories.values():
            for brand, subs in cat.brands.items():
                names.add(brand.lower())
                for s in subs:
                    names.add(s.lower())
        return names

    def get_all_products(self) -> Set[str]:
        products = set()
        for cat in self.categories.values():
            products.update(cat.products)
        return products

    def get_all_vocabulary(self) -> Set[str]:
        vocab = set()
        for cat in self.categories.values():
            vocab.update(cat.brands.keys())
            for subs in cat.brands.values():
                vocab.update(subs)
            vocab.update(cat.products)
            for base, variants in cat.model_patterns.items():
                for v in variants:
                    vocab.add(f"{base} {v}")
                vocab.add(base)
            vocab.update(cat.abbreviations.keys())
            vocab.update(cat.common_terms)
        # Add globals
        vocab.update(GLOBAL_UNITS)
        vocab.update(GLOBAL_CURRENCIES)
        vocab.update(GLOBAL_QUANTITY_ABBR)
        vocab.update(GLOBAL_COMMON_SHORTHAND)
        vocab.update(GLOBAL_SIZE_TERMS)
        vocab.update(PROTECTED_COMMON_WORDS)
        vocab.update(GLOBAL_ECOMMERCE_TERMS)
        return vocab

    def get_all_typo_mappings(self) -> Dict[str, str]:
        mappings = {}
        for cat in self.categories.values():
            mappings.update(cat.typo_mappings)
        return mappings

    def get_all_model_patterns(self) -> Dict[str, List[str]]:
        patterns = {}
        for cat in self.categories.values():
            patterns.update(cat.model_patterns)
        return patterns

    def get_protected_tokens(self) -> Set[str]:
        """Tokens that should NEVER be corrected."""
        protected = set()
        protected.update(GLOBAL_UNITS)
        protected.update(GLOBAL_CURRENCIES)
        protected.update(GLOBAL_QUANTITY_ABBR)
        protected.update(GLOBAL_COMMON_SHORTHAND)
        protected.update(GLOBAL_SIZE_TERMS)
        protected.update(self.get_all_brand_names_flat())
        for cat in self.categories.values():
            protected.update(cat.abbreviations.keys())
        return protected

    def get_stats(self) -> Dict:
        stats = {"categories": {}}
        for name, cat in self.categories.items():
            stats["categories"][name] = {
                "brands": len(cat.brands),
                "sub_brands": sum(len(v) for v in cat.brands.values()),
                "products": len(cat.products),
                "model_patterns": sum(len(v) for v in cat.model_patterns.values()),
                "typo_mappings": len(cat.typo_mappings),
                "abbreviations": len(cat.abbreviations),
            }
        stats["total_vocabulary"] = len(self.get_all_vocabulary())
        stats["total_brands"] = len(self.get_all_brands())
        stats["total_typo_mappings"] = len(self.get_all_typo_mappings())
        stats["total_protected_tokens"] = len(self.get_protected_tokens())
        return stats


# ======================================================================
# Main - Test
# ======================================================================

if __name__ == "__main__":
    vocab = UniversalVocab()
    vocab.load_all()

    stats = vocab.get_stats()
    print("\n" + "=" * 60)
    print("  Universal E-Commerce Vocabulary Stats")
    print("=" * 60)

    for cat_name, cat_stats in stats["categories"].items():
        total = sum(cat_stats.values())
        print(f"\n  {cat_name.upper()}: {total} items")
        for k, v in cat_stats.items():
            print(f"    {k}: {v}")

    print(f"\n  {'=' * 40}")
    print(f"  Total Vocabulary:      {stats['total_vocabulary']:,}")
    print(f"  Total Brands:          {stats['total_brands']:,}")
    print(f"  Total Typo Mappings:   {stats['total_typo_mappings']:,}")
    print(f"  Total Protected:       {stats['total_protected_tokens']:,}")
