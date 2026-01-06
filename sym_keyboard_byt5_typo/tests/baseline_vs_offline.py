#!/usr/bin/env python3
"""
Baseline vs Offline SymSpell Comparison

Baseline: IMPROVED SymSpell with all optimizations (typo_mappings + brand_products + preprocessing)
Offline: Our optimized SymSpell (same as baseline now)

Both should achieve ~73%+ accuracy
"""

import sys
import time
import re
sys.path.insert(0, '.')

from symspellpy import SymSpell, Verbosity
import os

# ============================================
# HELPER FUNCTIONS
# ============================================

def normalize_repeated_chars(token: str) -> str:
    """
    Normalize excessively repeated characters in a token.
    Preserves numbers and handles e-commerce specific cases.
    """
    # Skip pure numbers (like "1000", "256", "512")
    if token.isdigit():
        return token
    
    # Skip tokens with numbers mixed in (like "i7", "m3", "4090")
    if any(c.isdigit() for c in token):
        return token
    
    # Collapse 3+ repeated chars to 2, then try single
    result = re.sub(r'(.)\1{2,}', r'\1\1', token)
    # If still has doubles, try collapsing to single for common patterns
    result = re.sub(r'(.)\1+', r'\1', result)
    return result


# Protected patterns - NEVER change these
PROTECTED_PATTERNS = [
    r'^\d+gb$',      # 16gb, 32gb, etc.
    r'^\d+tb$',      # 1tb, 2tb, etc.
    r'^\d+mb$',      # 512mb, etc.
    r'^\d+hz$',      # 60hz, 144hz, etc.
    r'^\d+ghz$',     # 3.5ghz, etc.
    r'^\d+mhz$',     # 3200mhz, etc.
    r'^\d+w$',       # 650w, etc.
    r'^\d+mp$',      # 12mp, 48mp, etc.
    r'^\d+k$',       # 4k, 8k, etc.
    r'^i\d+$',       # i5, i7, i9
    r'^m\d+$',       # m1, m2, m3
    r'^[a-z]\d+$',   # g15, s23, a54, etc.
    r'^rtx\d+$',     # rtx3060, rtx4090
    r'^gtx\d+$',     # gtx1080, etc.
    r'^rx\d+$',      # rx6800, rx7900
    r'^ps\d+$',      # ps4, ps5
    r'^x\d+$',       # x360, x1
    r'^\d+mm$',      # 360mm, etc.
]


def is_protected(token: str) -> bool:
    """Check if token should be protected from correction."""
    token_lower = token.lower()
    
    # Pure numbers
    if re.match(r'^\d+(\.\d+)?$', token):
        return True
    
    # Single character
    if len(token) == 1:
        return True
    
    # Common units
    protected_units = {"gb", "tb", "hz", "mb", "kb", "ghz", "mhz", "fps", "mp", "mm", 
                       "4k", "8k", "5g", "4g", "usb", "rgb", "lcd", "led", "hdr", 
                       "ssd", "hdd", "nvme", "ddr4", "ddr5", "aio", "oled", "qhd", "fhd"}
    if token_lower in protected_units:
        return True
    
    # Check patterns
    for pattern in PROTECTED_PATTERNS:
        if re.match(pattern, token_lower):
            return True
    
    return False

# ============================================
# BASELINE SYMSPELL (NOW FULLY OPTIMIZED)
# ============================================

class BaselineSymSpell:
    """
    IMPROVED Baseline SymSpell with all optimizations:
    - Typo mappings for direct correction
    - Brand/product vocabulary  
    - Expanded common words
    - Normalize repeated chars preprocessing
    """
    
    def __init__(self):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        self.typo_mappings = {}  # Direct typo -> correct mappings
        
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        
        # 1. Load typo mappings FIRST (highest priority)
        typo_path = os.path.join(data_dir, 'typo_mappings.txt')
        if os.path.exists(typo_path):
            with open(typo_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            typo = parts[0].strip().lower()
                            correct = parts[1].strip().lower()
                            if typo and correct:
                                self.typo_mappings[typo] = correct
            print(f"  Loaded {len(self.typo_mappings)} typo mappings")
        
        # 2. Load brand products (high frequency)
        brand_path = os.path.join(data_dir, 'brand_products.txt')
        if os.path.exists(brand_path):
            count = 0
            with open(brand_path, 'r') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.sym_spell.create_dictionary_entry(word, 1000)  # High priority
                        count += 1
            print(f"  Loaded {count} brand/product terms")
        
        # 3. Load expanded common words
        expanded_path = os.path.join(data_dir, 'expanded_common_words.txt')
        if os.path.exists(expanded_path):
            count = 0
            with open(expanded_path, 'r') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.sym_spell.create_dictionary_entry(word, 100)
                        count += 1
            print(f"  Loaded {count} expanded common words")
        
        # 4. Load common words
        common_path = os.path.join(data_dir, 'common_words.txt')
        if os.path.exists(common_path):
            count = 0
            with open(common_path, 'r') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.sym_spell.create_dictionary_entry(word, 50)
                        count += 1
            print(f"  Loaded {count} common words")
        
        # 5. Load NLTK words (lowest priority)
        nltk_path = os.path.join(data_dir, 'nltk_words.txt')
        if os.path.exists(nltk_path):
            count = 0
            with open(nltk_path, 'r') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.sym_spell.create_dictionary_entry(word, 1)
                        count += 1
            print(f"  Loaded {count} NLTK words")
        
        # 6. Load domain vocab
        domain_path = os.path.join(data_dir, 'domain_vocab.txt')
        if os.path.exists(domain_path):
            count = 0
            with open(domain_path, 'r') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.sym_spell.create_dictionary_entry(word, 500)
                        count += 1
            print(f"  Loaded {count} domain vocab words")
        
        # 7. Load electronics vocab
        electronics_path = os.path.join(data_dir, 'electronics_vocab.txt')
        if os.path.exists(electronics_path):
            count = 0
            with open(electronics_path, 'r') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.sym_spell.create_dictionary_entry(word, 500)
                        count += 1
            print(f"  Loaded {count} electronics vocab words")
        
        print(f"Baseline loaded with {self.sym_spell.word_count} total dictionary words")
    
    def correct_query(self, query: str) -> str:
        """Optimized SymSpell correction with preprocessing and typo mappings"""
        tokens = query.lower().split()
        corrected = []
        
        for token in tokens:
            # Skip empty tokens
            if not token:
                continue
            
            # 1. Check typo mappings FIRST (exact match) - even before protected check
            if token in self.typo_mappings:
                corrected.append(self.typo_mappings[token])
                continue
            
            # 2. Skip protected patterns (numbers, units, specs)
            if is_protected(token):
                corrected.append(token)
                continue
            
            # 3. Try normalized version in typo mappings
            normalized = normalize_repeated_chars(token)
            if normalized != token and normalized in self.typo_mappings:
                corrected.append(self.typo_mappings[normalized])
                continue
            
            # 4. SymSpell lookup on original token
            suggestions = self.sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=3)
            if suggestions:
                best = suggestions[0]
                # Accept if distance <= 2, or if in dictionary with distance 3
                if best.distance <= 2:
                    corrected.append(best.term)
                    continue
                elif best.distance == 3 and len(token) > 5:
                    # Only accept distance 3 for longer words
                    corrected.append(best.term)
                    continue
            
            # 5. SymSpell lookup on normalized token
            if normalized != token:
                suggestions = self.sym_spell.lookup(normalized, Verbosity.CLOSEST, max_edit_distance=3)
                if suggestions and suggestions[0].distance <= 2:
                    corrected.append(suggestions[0].term)
                    continue
            
            # 6. Fallback: keep original token
            corrected.append(token)
        
        return ' '.join(corrected)


# ============================================
# TEST QUERIES - 100 Total (20 Easy, 50 Medium, 30 Hard)
# ============================================

EASY_QUERIES = [
    ("iphone 16 pro max 256gb blakc", "iphone 16 pro max 256gb black"),
    ("samsung galaxy s24 ultra 512g", "samsung galaxy s24 ultra 512gb"),
    ("macbook pro 14 m3 512gb spce", "macbook pro 14 m3 512gb space"),
    ("sony wh-1000xm5 balck", "sony wh-1000xm5 black"),
    ("logitech mx master 3s whit", "logitech mx master 3s white"),
    ("dell xps 15 i7 16gb 512g", "dell xps 15 i7 16gb 512gb"),
    ("airpods pro 2 usb-c whte", "airpods pro 2 usb-c white"),
    ("nvidia rtx 4090 foundrs", "nvidia rtx 4090 founders"),
    ("asus rog strix rtx 4080 blck", "asus rog strix rtx 4080 black"),
    ("razer blackwidow v4 pro blk", "razer blackwidow v4 pro black"),
    ("google pixel 9 pro obsidan", "google pixel 9 pro obsidian"),
    ("lenovo thinkpad x1 carbon i7", "lenovo thinkpad x1 carbon i7"),
    ("samsung 990 pro 2tb nvme ssd", "samsung 990 pro 2tb nvme ssd"),
    ("corsair k100 rgb mechancal", "corsair k100 rgb mechanical"),
    ("bose quietcomfort ultra blak", "bose quietcomfort ultra black"),
    ("amd ryzen 9 7950x3d procsor", "amd ryzen 9 7950x3d processor"),
    ("intel core i9-14900k cpu", "intel core i9-14900k cpu"),
    ("lg ultragear 27gp950 4k monitr", "lg ultragear 27gp950 4k monitor"),
    ("steelseries apex pro tkl keybord", "steelseries apex pro tkl keyboard"),
    ("hp envy 16 rtx 4060 i9 laptpo", "hp envy 16 rtx 4060 i9 laptop"),
]

MEDIUM_QUERIES = [
    ("samsugn galaxy z fold 5 512gb crem", "samsung galaxy z fold 5 512gb cream"),
    ("appel macbook air 15 m3 midnght", "apple macbook air 15 m3 midnight"),
    ("logitec g pro x superlight 2 whte", "logitech g pro x superlight 2 white"),
    ("nvdia geforce rtx 4080 super fe", "nvidia geforce rtx 4080 super fe"),
    ("razer deathadder v3 pro wirelss", "razer deathadder v3 pro wireless"),
    ("sony wf-1000xm5 earbuds silvr", "sony wf-1000xm5 earbuds silver"),
    ("assu rog zephyrus g14 ryzen 9", "asus rog zephyrus g14 ryzen 9"),
    ("corsiar vengeance ddr5 32gb 6000", "corsair vengeance ddr5 32gb 6000"),
    ("gigabyt aorus rtx 4090 mastr", "gigabyte aorus rtx 4090 master"),
    ("dell alienwar m18 rtx 4090 i9", "dell alienware m18 rtx 4090 i9"),
    ("xiaomi 14 ulrta 512gb leica blk", "xiaomi 14 ultra 512gb leica black"),
    ("oneplus 12 256gb flowy emrald", "oneplus 12 256gb flowy emerald"),
    ("googel pixel 8 pro 256gb hazl", "google pixel 8 pro 256gb hazel"),
    ("huawei mate 60 pro 512gb blck", "huawei mate 60 pro 512gb black"),
    ("nothign phone 2 256gb whte", "nothing phone 2 256gb white"),
    ("msi katna 15 rtx 4060 i7 144hz", "msi katana 15 rtx 4060 i7 144hz"),
    ("acer predtor helios 16 i9 rtx", "acer predator helios 16 i9 rtx"),
    ("lenvo legion pro 7 rtx 4080 qhd", "lenovo legion pro 7 rtx 4080 qhd"),
    ("hp specre x360 14 oled i7 16gb", "hp spectre x360 14 oled i7 16gb"),
    ("asus zenbokk 14 oled ryzen 7", "asus zenbook 14 oled ryzen 7"),
    ("keychon q1 pro wireless mechncl", "keychron q1 pro wireless mechanical"),
    ("ducky one 3 sf rgb gateron yello", "ducky one 3 sf rgb gateron yellow"),
    ("sennheiser momentum 4 wireles blk", "sennheiser momentum 4 wireless black"),
    ("jabr elite 85t anc earbuds navy", "jabra elite 85t anc earbuds navy"),
    ("beats studoi pro wireless blck", "beats studio pro wireless black"),
    ("samsung odyssy g9 49 oled 240hz", "samsung odyssey g9 49 oled 240hz"),
    ("dell ultrashrap u2723qe 4k usbc", "dell ultrasharp u2723qe 4k usbc"),
    ("lg ultrawide 34wk95u 5k thundrbt", "lg ultrawide 34wk95u 5k thunderbolt"),
    ("asus proart pa32ucg 4k hdr creatv", "asus proart pa32ucg 4k hdr creative"),
    ("benq mobiuz ex2710q 165hz gamng", "benq mobiuz ex2710q 165hz gaming"),
    ("amd radoen rx 7900 xtx sapphir", "amd radeon rx 7900 xtx sapphire"),
    ("g.skil trident z5 rgb 64gb ddr5", "g.skill trident z5 rgb 64gb ddr5"),
    ("kignston fury beast ddr5 32gb", "kingston fury beast ddr5 32gb"),
    ("wd blakc sn850x 2tb nvme gen4", "wd black sn850x 2tb nvme gen4"),
    ("seagaet barracuda 8tb hdd nas", "seagate barracuda 8tb hdd nas"),
    ("asus rog maxmus z790 hero wifi", "asus rog maximus z790 hero wifi"),
    ("msi meg z790 ace ddr5 wifi7", "msi meg z790 ace ddr5 wifi7"),
    ("corsiar rm1000x 1000w 80+ gold", "corsair rm1000x 1000w 80+ gold"),
    ("seasnic prime tx-1000 titanium", "seasonic prime tx-1000 titanium"),
    ("lian li o11 dynmic evo white rgb", "lian li o11 dynamic evo white rgb"),
    ("nzxt h9 fow white tempered glas", "nzxt h9 flow white tempered glass"),
    ("fractla design torrent rgb black", "fractal design torrent rgb black"),
    ("noctua nh-d15 chromx black cooler", "noctua nh-d15 chromax black cooler"),
    ("corsiar h150i elite lcd 360mm aio", "corsair h150i elite lcd 360mm aio"),
    ("deepcol ls720 360mm argb cooler", "deepcool ls720 360mm argb cooler"),
    ("canon pixam ts8350 wireless printr", "canon pixma ts8350 wireless printer"),
    ("epson ecotnak et-8550 photo printr", "epson ecotank et-8550 photo printer"),
    ("logtiech brio 500 4k webcam strmng", "logitech brio 500 4k webcam streaming"),
    ("elgato facecm pro 4k streamng cam", "elgato facecam pro 4k streaming cam"),
    ("sandisk extrme pro 2tb portable ssd", "sandisk extreme pro 2tb portable ssd"),
]

HARD_QUERIES = [
    ("samsugn galxy z flp 5 256gb creem", "samsung galaxy z flip 5 256gb cream"),
    ("appel macbok pro 16 m3 mxa 1tb spce", "apple macbook pro 16 m3 max 1tb space"),
    ("logiteh g pro x superlgiht 2 wht", "logitech g pro x superlight 2 white"),
    ("nvida gefroce rtx 4090 foundrs editn", "nvidia geforce rtx 4090 founders edition"),
    ("razer huntsmn v3 pro anlaog keybrd", "razer huntsman v3 pro analog keyboard"),
    ("sonny wh-1000xm5 noice canceling blk", "sony wh-1000xm5 noise canceling black"),
    ("assu rog zephrus g16 ryezn 9 rtx", "asus rog zephyrus g16 ryzen 9 rtx"),
    ("corsiar dominator platnum ddr5 64gb", "corsair dominator platinum ddr5 64gb"),
    ("gigabyt aourus z790 mastr ddr5 wifi", "gigabyte aorus z790 master ddr5 wifi"),
    ("dell alienwre m16 r2 rtx 4080 qhd", "dell alienware m16 r2 rtx 4080 qhd"),
    ("xiaom 14 ulrta 512g leika camra blk", "xiaomi 14 ultra 512gb leica camera black"),
    ("onepls opne 512gb flowy emrald foldbl", "oneplus open 512gb flowy emerald foldable"),
    ("googel pixle 8 por 256gb obsidan ai", "google pixel 8 pro 256gb obsidian ai"),
    ("huwaie mate 60 pro plsu 1tb blck", "huawei mate 60 pro plus 1tb black"),
    ("nothng phoen 2a 256gb whte glpyh", "nothing phone 2a 256gb white glyph"),
    ("msi stealt 17 stuido rtx 4080 4k oled", "msi stealth 17 studio rtx 4080 4k oled"),
    ("acer predtor helis 18 i9 rtx 4090", "acer predator helios 18 i9 rtx 4090"),
    ("lenvo yoag 9i 14 oled i7 evo platfrm", "lenovo yoga 9i 14 oled i7 evo platform"),
    ("hp spectere x360 16 oled i7 32gb 1tb", "hp spectre x360 16 oled i7 32gb 1tb"),
    ("asus zenbookk pro 16x oled ryezn 9", "asus zenbook pro 16x oled ryzen 9"),
    ("keychorn q3 max wirless mecahnical rgb", "keychron q3 max wireless mechanical rgb"),
    ("wooting 60he mechancial analog keybrd", "wooting 60he mechanical analog keyboard"),
    ("sennheser momentum 4 wirless anc blk", "sennheiser momentum 4 wireless anc black"),
    ("jabr elite 10 anc earbuds titanum", "jabra elite 10 anc earbuds titanium"),
    ("samsnug odyssey ark 55 curvd 4k 165hz", "samsung odyssey ark 55 curved 4k 165hz"),
    ("lg ultrager oled 27gr95qe 240hz qhd", "lg ultragear oled 27gr95qe 240hz qhd"),
    ("amd radoen rx 7900 xrx sapphre nitro", "amd radeon rx 7900 xtx sapphire nitro"),
    ("be queit dark powr pro 13 1000w pltnm", "be quiet dark power pro 13 1000w platinum"),
    ("nxzt krakn z73 rgb 360mm aio lcd", "nzxt kraken z73 rgb 360mm aio lcd"),
    ("corsiar 5000d airfow black temepered", "corsair 5000d airflow black tempered"),
]


def run_comparison():
    print("="*90)
    print("🔬 BASELINE vs OFFLINE SymSpell Comparison")
    print("="*90)
    print()
    
    # Load baseline
    print("Loading Baseline SymSpell...")
    baseline = BaselineSymSpell()
    
    # Load offline (our optimized version)
    print("Loading Offline SymSpell (optimized)...")
    from app.typo_corrector import TypoCorrector
    offline = TypoCorrector()
    
    print()
    print("="*90)
    
    all_queries = [
        ("EASY", EASY_QUERIES),
        ("MEDIUM", MEDIUM_QUERIES),
        ("HARD", HARD_QUERIES),
    ]
    
    baseline_results = {"EASY": 0, "MEDIUM": 0, "HARD": 0}
    offline_results = {"EASY": 0, "MEDIUM": 0, "HARD": 0}
    
    detailed_results = []
    
    query_num = 1
    
    for level, queries in all_queries:
        print(f"\n{'='*90}")
        print(f"📊 {level} - {len(queries)} queries")
        print("="*90)
        
        for typo, expected in queries:
            # Baseline correction
            baseline_result = baseline.correct_query(typo)
            baseline_correct = baseline_result == expected
            
            # Offline correction
            offline_result = offline.correct_query(typo)
            offline_correct = offline_result == expected
            
            if baseline_correct:
                baseline_results[level] += 1
            if offline_correct:
                offline_results[level] += 1
            
            # Determine status
            if baseline_correct and offline_correct:
                status = "✅✅"  # Both correct
            elif not baseline_correct and offline_correct:
                status = "❌✅"  # Only offline correct
            elif baseline_correct and not offline_correct:
                status = "✅❌"  # Only baseline correct (regression!)
            else:
                status = "❌❌"  # Both wrong
            
            print(f"\n{query_num:3}. {status} (Baseline|Offline)")
            print(f"     Query:    \"{typo}\"")
            print(f"     Expected: \"{expected}\"")
            
            if not baseline_correct or not offline_correct:
                if not baseline_correct:
                    print(f"     Baseline: \"{baseline_result}\"")
                if not offline_correct:
                    print(f"     Offline:  \"{offline_result}\"")
            
            detailed_results.append({
                "level": level,
                "query": typo,
                "expected": expected,
                "baseline": baseline_result,
                "offline": offline_result,
                "baseline_correct": baseline_correct,
                "offline_correct": offline_correct
            })
            
            query_num += 1
    
    # Summary
    print()
    print("="*90)
    print("📈 FINAL COMPARISON")
    print("="*90)
    print()
    print(f"{'Level':<10} {'Baseline':>15} {'Offline':>15} {'Improvement':>15}")
    print("-"*60)
    
    total_baseline = 0
    total_offline = 0
    
    for level, queries in all_queries:
        total = len(queries)
        b_score = baseline_results[level]
        o_score = offline_results[level]
        total_baseline += b_score
        total_offline += o_score
        
        b_pct = b_score / total * 100
        o_pct = o_score / total * 100
        improvement = o_pct - b_pct
        
        print(f"{level:<10} {b_score:>6}/{total} ({b_pct:>5.1f}%) {o_score:>6}/{total} ({o_pct:>5.1f}%) {improvement:>+10.1f}%")
    
    print("-"*60)
    total = len(EASY_QUERIES) + len(MEDIUM_QUERIES) + len(HARD_QUERIES)
    b_pct = total_baseline / total * 100
    o_pct = total_offline / total * 100
    improvement = o_pct - b_pct
    
    print(f"{'TOTAL':<10} {total_baseline:>6}/{total} ({b_pct:>5.1f}%) {total_offline:>6}/{total} ({o_pct:>5.1f}%) {improvement:>+10.1f}%")
    print("="*90)
    
    # Breakdown
    print()
    print("📊 DETAILED BREAKDOWN")
    print("-"*60)
    
    both_correct = sum(1 for r in detailed_results if r['baseline_correct'] and r['offline_correct'])
    only_offline = sum(1 for r in detailed_results if not r['baseline_correct'] and r['offline_correct'])
    only_baseline = sum(1 for r in detailed_results if r['baseline_correct'] and not r['offline_correct'])
    both_wrong = sum(1 for r in detailed_results if not r['baseline_correct'] and not r['offline_correct'])
    
    print(f"  ✅✅ Both Correct:      {both_correct:>3} ({both_correct/total*100:.1f}%)")
    print(f"  ❌✅ Only Offline:      {only_offline:>3} ({only_offline/total*100:.1f}%) ← Improvement")
    print(f"  ✅❌ Only Baseline:     {only_baseline:>3} ({only_baseline/total*100:.1f}%) ← Regression")
    print(f"  ❌❌ Both Wrong:        {both_wrong:>3} ({both_wrong/total*100:.1f}%)")
    print("="*90)
    
    # Show regressions if any
    if only_baseline > 0:
        print()
        print("⚠️  REGRESSIONS (Baseline correct but Offline wrong):")
        print("-"*60)
        for r in detailed_results:
            if r['baseline_correct'] and not r['offline_correct']:
                print(f"  Query:    {r['query']}")
                print(f"  Expected: {r['expected']}")
                print(f"  Baseline: {r['baseline']} ✅")
                print(f"  Offline:  {r['offline']} ❌")
                print()


if __name__ == "__main__":
    run_comparison()
