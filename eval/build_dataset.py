"""
build_dataset.py
================
Generate a 500-query categorized evaluation dataset for the SmartSearch
Correction module. Designed to *separate* models — easy queries every
half-decent corrector should pass, but hard / adversarial / dense-error
queries that crack open the differences between ByT5-Small, ByT5-Base,
ByT5-Large-V3, T5-Large-V2.1 and Qwen-3.5-2B.

Schema (each row):
    {
        "id": "L3-MED-021",
        "query": "...",            # input the model sees
        "expected": "...",         # gold corrected output
        "length": "short|medium|long",
        "difficulty": "L1|L2|L3|L4|L5",
        "category": "...",
        "notes": "..."
    }

Difficulty:
    L1 — clean trap (correctly spelled, MUST NOT be changed) — measures FP rate
    L2 — easy: single ED-1 typo on a common token
    L3 — medium: 2 typos OR transposition OR missing letter on multi-token
    L4 — hard: dense multi-edit, brand corruption traps, phonetic, ED-2
    L5 — very hard: heavily corrupted compound queries with mixed errors,
        long compound product names with brand+spec+model, ED-3+ typos

Length:
    short  — 1-2 tokens
    medium — 3-5 tokens
    long   — 6+ tokens
"""

import json
import os

OUT = os.path.join(os.path.dirname(__file__), "dataset.json")


def row(id_, query, expected, length, difficulty, category, notes=""):
    return {
        "id": id_,
        "query": query,
        "expected": expected,
        "length": length,
        "difficulty": difficulty,
        "category": category,
        "notes": notes,
    }


rows = []

# ---------------------------------------------------------------------------
# L1 — CLEAN TRAPS (must not be changed). 90 queries.
# ---------------------------------------------------------------------------

L1_SHORT = [
    "iphone", "samsung", "logitech mouse", "razer keyboard", "bose headphones",
    "sony speaker", "dell laptop", "asus monitor", "hp printer", "lenovo thinkpad",
    "macbook", "airpods", "kindle", "ps5 controller", "xbox controller",
    "nvidia gpu", "amd cpu", "intel cpu", "msi monitor", "corsair ram",
    "ssd 1tb", "hdd 2tb", "wifi router", "smart tv", "wireless charger",
]
for i, q in enumerate(L1_SHORT):
    rows.append(row(f"L1-SHO-{i:03d}", q, q, "short", "L1", "clean", "must NOT change"))

L1_MEDIUM = [
    "apple iphone 15 pro max",
    "samsung galaxy s24 ultra",
    "logitech mx master 3s mouse",
    "razer deathadder v3 pro",
    "bose quietcomfort ultra headphones",
    "sony wh-1000xm5 wireless",
    "dell xps 15 oled laptop",
    "asus rog strix monitor",
    "hp laserjet pro printer",
    "lenovo thinkpad x1 carbon",
    "macbook pro m3 max",
    "airpods pro 2 with usb-c",
    "kindle paperwhite signature edition",
    "playstation 5 dualsense controller",
    "xbox series x wireless controller",
    "nvidia rtx 4090 founders edition",
    "amd ryzen 9 7950x cpu",
    "intel core i9 14900k",
    "msi mag 274qrf monitor",
    "corsair vengeance 32gb ddr5",
    "samsung 980 pro 1tb ssd",
    "wd black 2tb hdd",
    "asus rt-ax88u wifi router",
    "lg c3 65 inch oled tv",
    "anker magsafe wireless charger",
    "logitech g pro x superlight",
    "razer huntsman v3 pro tkl",
    "sennheiser hd 660s2 headphones",
    "shure sm7b microphone",
    "elgato stream deck mk2",
]
for i, q in enumerate(L1_MEDIUM):
    rows.append(row(f"L1-MED-{i:03d}", q, q, "medium", "L1", "clean", "must NOT change"))

L1_LONG = [
    "apple macbook pro 16 inch m3 max 64gb 2tb space black",
    "samsung galaxy s24 ultra 512gb titanium black unlocked",
    "asus rog strix scar 18 g834 gaming laptop rtx 4090",
    "razer blade 18 2024 intel core i9 rtx 4080 240hz qhd",
    "lg ultragear 27 inch 4k oled gaming monitor 240hz",
    "logitech mx master 3s for mac performance wireless mouse",
    "sony wh 1000xm5 wireless noise canceling headphones black",
    "bose quietcomfort ultra wireless noise cancelling earbuds",
    "corsair hs80 max wireless gaming headset rgb dolby atmos",
    "nvidia geforce rtx 4090 24gb gddr6x graphics card founders",
    "amd ryzen 9 7950x3d 16 core 32 thread desktop processor",
    "intel core i9 14900k unlocked desktop processor 24 cores",
    "samsung 990 pro 2tb pcie 4.0 nvme m 2 internal ssd",
    "western digital black sn850x 1tb nvme internal gaming ssd",
    "asus rog strix x670e creator wifi atx motherboard ddr5",
    "lg c3 65 inch oled evo 4k smart tv with self lit pixels",
    "sony bravia xr 75 inch class a95l qd oled 4k hdr smart tv",
    "anker 737 power bank 24000mah 140w usb c portable charger",
    "ugreen nexode 100w usb c gan charger 4 port fast charging",
    "satechi usb c hub multiport adapter pro for macbook pro",
    "razer basilisk v3 pro wireless ergonomic gaming mouse 30k dpi",
    "logitech g pro x superlight 2 wireless gaming mouse hero 2",
    "corsair k70 rgb pro mechanical gaming keyboard cherry mx red",
    "sennheiser hd 800 s reference open back headphones audiophile",
    "shure mv7 plus podcast dynamic microphone usb xlr",
    "elgato cam link 4k external video capture card hdmi",
    "rode procaster broadcast quality dynamic microphone xlr",
    "blue yeti x professional condenser usb microphone for streaming",
    "audio technica at2020 cardioid condenser studio xlr microphone",
    "fifine ampligame a8 usb gaming microphone rgb cardioid",
    "samsung odyssey neo g9 57 inch dual 4k 240hz curved monitor",
    "asus rog swift pg27aqdm 26.5 inch 1440p oled 240hz monitor",
    "lg 27gp950 b ultragear 27 inch 4k nano ips 144hz gaming monitor",
    "alienware aw3423dwf 34 inch curved qd oled 165hz ultrawide",
]
for i, q in enumerate(L1_LONG):
    rows.append(row(f"L1-LNG-{i:03d}", q, q, "long", "L1", "clean", "must NOT change"))


# ---------------------------------------------------------------------------
# L2 — EASY: single ED-1 typo on common token. 100 queries.
# ---------------------------------------------------------------------------

L2_SHORT = [
    ("iphne",      "iphone"),
    ("ipone",      "iphone"),
    ("samung",     "samsung"),
    ("sumsung",    "samsung"),
    ("laptp",      "laptop"),
    ("laptopp",    "laptop"),
    ("moniter",    "monitor"),
    ("kyboard",    "keyboard"),
    ("kerboard",   "keyboard"),
    ("mose",       "mouse"),
    ("hedphones",  "headphones"),
    ("headfones",  "headphones"),
    ("speker",     "speaker"),
    ("camra",      "camera"),
    ("camera ",    "camera"),
    ("printr",     "printer"),
    ("rooter",     "router"),
    ("tablt",      "tablet"),
    ("smarphone",  "smartphone"),
    ("chrger",     "charger"),
    ("cabel",      "cable"),
    ("bluetoth",   "bluetooth"),
    ("wifii",      "wifi"),
    ("airpod",     "airpods"),
    ("kindel",     "kindle"),
    ("powebank",   "powerbank"),
    ("microphne",  "microphone"),
    ("webcm",      "webcam"),
    ("controler",  "controller"),
    ("contoller",  "controller"),
]
for i, (q, e) in enumerate(L2_SHORT):
    rows.append(row(f"L2-SHO-{i:03d}", q, e, "short", "L2", "single_edit_typo", "ED-1"))

L2_MEDIUM = [
    ("aple iphone 15",                   "apple iphone 15"),
    ("samsng galaxy s24",                "samsung galaxy s24"),
    ("logitec mx master",                "logitech mx master"),
    ("razerr deathadder v3",             "razer deathadder v3"),
    ("bose quitcomfort ultra",           "bose quietcomfort ultra"),
    ("sony wireless hedphones",          "sony wireless headphones"),
    ("dell xps 15 laptp",                "dell xps 15 laptop"),
    ("asus rog moniter",                 "asus rog monitor"),
    ("hp laserjet printr",               "hp laserjet printer"),
    ("lenovo thinkpadd x1",              "lenovo thinkpad x1"),
    ("macbookk pro m3",                  "macbook pro m3"),
    ("airpods pro 2 usbc",               "airpods pro 2 usb-c"),
    ("kindle papperwhite",               "kindle paperwhite"),
    ("playstation 5 controler",          "playstation 5 controller"),
    ("xbox seriess x",                   "xbox series x"),
    ("nvidia rtx 4090 founders",         "nvidia rtx 4090 founders"),
    ("amd ryzn 9 7950x",                 "amd ryzen 9 7950x"),
    ("intel core i9 14900k",             "intel core i9 14900k"),
    ("msi mag 274qrf moniter",           "msi mag 274qrf monitor"),
    ("corsair vengeance 32gb ddr5",      "corsair vengeance 32gb ddr5"),
    ("samsung 980 pro 1tb sdd",          "samsung 980 pro 1tb ssd"),
    ("wd black 2tb harddrive",           "wd black 2tb harddrive"),
    ("asus rt-ax88u wifii router",       "asus rt-ax88u wifi router"),
    ("lg c3 65 inch oledd tv",           "lg c3 65 inch oled tv"),
    ("anker magsafe wirless charger",    "anker magsafe wireless charger"),
    ("logitech g pro superlite",         "logitech g pro superlight"),
    ("razer huntsman v3 pro tkll",       "razer huntsman v3 pro tkl"),
    ("sennheiser hd 660 hedphones",      "sennheiser hd 660 headphones"),
    ("shure sm7b microphne",             "shure sm7b microphone"),
    ("elgato stream dec mk2",            "elgato stream deck mk2"),
    ("aples watch series 9",             "apple watch series 9"),
    ("samsung galaxy bud pro 2",         "samsung galaxy buds pro 2"),
    ("steelseries arctis nova prro",     "steelseries arctis nova pro"),
    ("hyperx cloud aplha s",             "hyperx cloud alpha s"),
    ("anker soundcore liberti 4",        "anker soundcore liberty 4"),
]
for i, (q, e) in enumerate(L2_MEDIUM):
    rows.append(row(f"L2-MED-{i:03d}", q, e, "medium", "L2", "single_edit_typo", "ED-1 in one token"))

L2_LONG = [
    ("apple macbook pro 16 inch m3 max 64gb 2tb space blak",
     "apple macbook pro 16 inch m3 max 64gb 2tb space black"),
    ("samsung galaxy s24 ultra 512gb titanium black unloked",
     "samsung galaxy s24 ultra 512gb titanium black unlocked"),
    ("asus rog strix scar 18 gaming laptp rtx 4090",
     "asus rog strix scar 18 gaming laptop rtx 4090"),
    ("razer blade 18 2024 intel core i9 rtx 4080 240hz qhdd",
     "razer blade 18 2024 intel core i9 rtx 4080 240hz qhd"),
    ("lg ultragear 27 inch 4k oled gamming monitor 240hz",
     "lg ultragear 27 inch 4k oled gaming monitor 240hz"),
    ("logitech mx master 3s for mac perfomance wireless mouse",
     "logitech mx master 3s for mac performance wireless mouse"),
    ("sony wh 1000xm5 wireless noise cancling headphones black",
     "sony wh 1000xm5 wireless noise canceling headphones black"),
    ("bose quietcomfort ultra wireless noise cancelling earbuds",
     "bose quietcomfort ultra wireless noise cancelling earbuds"),
    ("corsair hs80 max wireless gaming headsett rgb dolby atmos",
     "corsair hs80 max wireless gaming headset rgb dolby atmos"),
    ("nvidia geforce rtx 4090 24gb gddr6x graphic card founders",
     "nvidia geforce rtx 4090 24gb gddr6x graphics card founders"),
    ("amd ryzen 9 7950x3d 16 core 32 thread desktop procesor",
     "amd ryzen 9 7950x3d 16 core 32 thread desktop processor"),
    ("intel core i9 14900k unlocked desktop processor 24 corres",
     "intel core i9 14900k unlocked desktop processor 24 cores"),
    ("samsung 990 pro 2tb pcie 4.0 nvme m 2 internal sdd",
     "samsung 990 pro 2tb pcie 4.0 nvme m 2 internal ssd"),
    ("western digital black sn850x 1tb nvme internal gaming sdd",
     "western digital black sn850x 1tb nvme internal gaming ssd"),
    ("asus rog strix x670e creator wifii atx motherboard ddr5",
     "asus rog strix x670e creator wifi atx motherboard ddr5"),
    ("lg c3 65 inch oled evo 4k smart tv with self lit pixells",
     "lg c3 65 inch oled evo 4k smart tv with self lit pixels"),
    ("sony bravia xr 75 inch class a95l qd oled 4k hdr smart tvv",
     "sony bravia xr 75 inch class a95l qd oled 4k hdr smart tv"),
    ("anker 737 power bank 24000mah 140w usb c portable charer",
     "anker 737 power bank 24000mah 140w usb c portable charger"),
    ("ugreen nexode 100w usb c gan chargr 4 port fast charging",
     "ugreen nexode 100w usb c gan charger 4 port fast charging"),
    ("satechi usb c hub multport adapter pro for macbook pro",
     "satechi usb c hub multiport adapter pro for macbook pro"),
    ("razer basilisk v3 pro wireless ergonmic gaming mouse 30k dpi",
     "razer basilisk v3 pro wireless ergonomic gaming mouse 30k dpi"),
    ("logitech g pro x superlight 2 wirless gaming mouse hero 2",
     "logitech g pro x superlight 2 wireless gaming mouse hero 2"),
    ("corsair k70 rgb pro mechanikal gaming keyboard cherry mx red",
     "corsair k70 rgb pro mechanical gaming keyboard cherry mx red"),
    ("sennheiser hd 800 s reference open back headphons audiophile",
     "sennheiser hd 800 s reference open back headphones audiophile"),
    ("shure mv7 plus podcast dynamic microphne usb xlr",
     "shure mv7 plus podcast dynamic microphone usb xlr"),
    ("elgato cam link 4k externall video capture card hdmi",
     "elgato cam link 4k external video capture card hdmi"),
    ("rode procaster broadcast qualty dynamic microphone xlr",
     "rode procaster broadcast quality dynamic microphone xlr"),
    ("blue yeti x profesional condenser usb microphone for streaming",
     "blue yeti x professional condenser usb microphone for streaming"),
    ("audio technica at2020 cardiod condenser studio xlr microphone",
     "audio technica at2020 cardioid condenser studio xlr microphone"),
    ("fifine ampligame a8 usb gaming microphne rgb cardioid",
     "fifine ampligame a8 usb gaming microphone rgb cardioid"),
    ("samsung odyssey neo g9 57 inch dual 4k 240hz curvd monitor",
     "samsung odyssey neo g9 57 inch dual 4k 240hz curved monitor"),
    ("asus rog swift pg27aqdm 26.5 inch 1440p oled 240hzz monitor",
     "asus rog swift pg27aqdm 26.5 inch 1440p oled 240hz monitor"),
    ("lg 27gp950 b ultragear 27 inch 4k nano ips 144hz gamming monitor",
     "lg 27gp950 b ultragear 27 inch 4k nano ips 144hz gaming monitor"),
    ("alienware aw3423dwf 34 inch curvd qd oled 165hz ultrawide",
     "alienware aw3423dwf 34 inch curved qd oled 165hz ultrawide"),
    ("apple ipad pro 12.9 inch m4 wifi cellular 1tb space gry",
     "apple ipad pro 12.9 inch m4 wifi cellular 1tb space gray"),
]
for i, (q, e) in enumerate(L2_LONG):
    rows.append(row(f"L2-LNG-{i:03d}", q, e, "long", "L2", "single_edit_typo", "ED-1 in one token"))


# ---------------------------------------------------------------------------
# L3 — MEDIUM: 2 typos OR transposition OR missing letter, multi-token. 110.
# ---------------------------------------------------------------------------

L3_SHORT = [
    ("hedphone",       "headphone"),
    ("micrphne",       "microphone"),
    ("wirelees",       "wireless"),
    ("blutooth",       "bluetooth"),
    ("recievr",        "receiver"),
    ("acessory",       "accessory"),
    ("nottebook",      "notebook"),
    ("wireles mouse",  "wireless mouse"),
    ("smart watche",   "smart watch"),
    ("gamming pc",     "gaming pc"),
    ("hdmii cable",    "hdmi cable"),
    ("usbb cable",     "usb cable"),
    ("wirless earbud", "wireless earbud"),
    ("blutoot speaker","bluetooth speaker"),
    ("noisee canceling","noise canceling"),
    ("ergonomik mouse","ergonomic mouse"),
    ("mecanical kybd", "mechanical keyboard"),
    ("ultrawide moniter","ultrawide monitor"),
    ("4k televison",   "4k television"),
    ("solid stat drive","solid state drive"),
    ("graphic crd",    "graphics card"),
    ("portable charer","portable charger"),
    ("gamng laptp",    "gaming laptop"),
    ("smart tvv",      "smart tv"),
    ("powr bank",      "power bank"),
]
for i, (q, e) in enumerate(L3_SHORT):
    rows.append(row(f"L3-SHO-{i:03d}", q, e, "short", "L3", "double_edit_or_transpose", ""))

L3_MEDIUM = [
    ("aple ipone 15 pro",                "apple iphone 15 pro"),
    ("samsng galxy s24 ultra",           "samsung galaxy s24 ultra"),
    ("logitec mx mater 3s",              "logitech mx master 3s"),
    ("razerr deathadderr v3 pro",        "razer deathadder v3 pro"),
    ("bose quitcomfor ultra",            "bose quietcomfort ultra"),
    ("sony wh1000xm5 wirless",           "sony wh-1000xm5 wireless"),
    ("dell xps 15 oledd laptp",          "dell xps 15 oled laptop"),
    ("asus rog strix moniter",           "asus rog strix monitor"),
    ("hp laserjett pro printr",          "hp laserjet pro printer"),
    ("lenovo thinkpadd x1 carbn",        "lenovo thinkpad x1 carbon"),
    ("macboook pro m3 max",              "macbook pro m3 max"),
    ("airpod pro 2 usb-c",               "airpods pro 2 usb-c"),
    ("kindel paperwhitee signature",     "kindle paperwhite signature"),
    ("playstation 5 dualsense controler","playstation 5 dualsense controller"),
    ("xbox sereies x wireles controller","xbox series x wireless controller"),
    ("nvidia rtx 4090 founders editon",  "nvidia rtx 4090 founders edition"),
    ("amd ryzn 9 7950x cpu",             "amd ryzen 9 7950x cpu"),
    ("intel core i9 14900k unlokced",    "intel core i9 14900k unlocked"),
    ("msi mag 274qrf gamming monitor",   "msi mag 274qrf gaming monitor"),
    ("corsiar vengance 32gb ddr5",       "corsair vengeance 32gb ddr5"),
    ("samsng 980 pro 1tb sdd",           "samsung 980 pro 1tb ssd"),
    ("wd blackk 2tb hardrive",           "wd black 2tb hard drive"),
    ("asus rt-ax88u wifii rooter",       "asus rt-ax88u wifi router"),
    ("lg c3 65 inch oledd televison",    "lg c3 65 inch oled television"),
    ("ankr magsafe wirless charer",      "anker magsafe wireless charger"),
    ("logitech g pro superlite mose",    "logitech g pro superlight mouse"),
    ("razer huntsmen v3 prr tkll",       "razer huntsman v3 pro tkl"),
    ("senhiser hd 660s2 hedphones",      "sennheiser hd 660s2 headphones"),
    ("shre sm7b microfone",              "shure sm7b microphone"),
    ("elgto stream dec mk2",             "elgato stream deck mk2"),
    ("aple watche series 9",             "apple watch series 9"),
    ("samsng galxy buds pro 2",          "samsung galaxy buds pro 2"),
    ("steelseries arcis nva pro",        "steelseries arctis nova pro"),
    ("hyperx clud alpa s",               "hyperx cloud alpha s"),
    ("ankr soundcre liberty 4",          "anker soundcore liberty 4"),
    ("logitec g502 hero gamming mose",   "logitech g502 hero gaming mouse"),
    ("razerr basilisk v3 pro mose",      "razer basilisk v3 pro mouse"),
    ("corsiar k95 rgb platnum xt",       "corsair k95 rgb platinum xt"),
    ("razer blackwidoow v4 pro",         "razer blackwidow v4 pro"),
    ("steel seris apex pro tkl",         "steelseries apex pro tkl"),
]
for i, (q, e) in enumerate(L3_MEDIUM):
    rows.append(row(f"L3-MED-{i:03d}", q, e, "medium", "L3", "double_edit_or_transpose", ""))

L3_LONG = [
    ("aple macbok pro 16 inch m3 max 64gb 2tb space blak",
     "apple macbook pro 16 inch m3 max 64gb 2tb space black"),
    ("samsng galxy s24 ultra 512gb titanim black unloked",
     "samsung galaxy s24 ultra 512gb titanium black unlocked"),
    ("asus rog stix scar 18 g834 gamming laptp rtx 4090",
     "asus rog strix scar 18 g834 gaming laptop rtx 4090"),
    ("razerr blad 18 2024 intel core i9 rtx 4080 240hz qhdd",
     "razer blade 18 2024 intel core i9 rtx 4080 240hz qhd"),
    ("lg ultragar 27 inch 4k oledd gamming moniter 240hz",
     "lg ultragear 27 inch 4k oled gaming monitor 240hz"),
    ("logitec mx mater 3s for mac perfomance wirless mose",
     "logitech mx master 3s for mac performance wireless mouse"),
    ("sony wh 1000xm5 wirless noise cancling hedphones black",
     "sony wh 1000xm5 wireless noise canceling headphones black"),
    ("bose quitcomfort ultra wirless noise cancelling earbud",
     "bose quietcomfort ultra wireless noise cancelling earbuds"),
    ("corsiar hs80 max wirless gamming headsett rgb dolby atmos",
     "corsair hs80 max wireless gaming headset rgb dolby atmos"),
    ("nvidia gefoce rtx 4090 24gb gddr6x graphic crd founders",
     "nvidia geforce rtx 4090 24gb gddr6x graphics card founders"),
    ("amd ryzn 9 7950x3d 16 core 32 thread desktp procesor",
     "amd ryzen 9 7950x3d 16 core 32 thread desktop processor"),
    ("intel cor i9 14900k unloked desktop procesor 24 corres",
     "intel core i9 14900k unlocked desktop processor 24 cores"),
    ("samsng 990 pro 2tb pcie 4.0 nvme m 2 internl sdd",
     "samsung 990 pro 2tb pcie 4.0 nvme m 2 internal ssd"),
    ("westrn digtal black sn850x 1tb nvme internl gaming sdd",
     "western digital black sn850x 1tb nvme internal gaming ssd"),
    ("asus rog stix x670e creator wifii atx motherbord ddr5",
     "asus rog strix x670e creator wifi atx motherboard ddr5"),
    ("lg c3 65 inch oledd evo 4k smart tvv with self lit pixells",
     "lg c3 65 inch oled evo 4k smart tv with self lit pixels"),
    ("sony braviar xr 75 inch a95l qd oledd 4k hdr smart tvv",
     "sony bravia xr 75 inch a95l qd oled 4k hdr smart tv"),
    ("ankr 737 powr bank 24000mah 140w usb c portble charer",
     "anker 737 power bank 24000mah 140w usb c portable charger"),
    ("ugren nexode 100w usb c gan chargr 4 port fast charing",
     "ugreen nexode 100w usb c gan charger 4 port fast charging"),
    ("sateci usb c hub multport adaptr pro for macboook pro",
     "satechi usb c hub multiport adapter pro for macbook pro"),
    ("razerr basilik v3 pro wirless ergonmic gamming mose 30k dpi",
     "razer basilisk v3 pro wireless ergonomic gaming mouse 30k dpi"),
    ("logitec g pro x superlight 2 wirless gamming mose hero 2",
     "logitech g pro x superlight 2 wireless gaming mouse hero 2"),
    ("corsiar k70 rgb pro mecanikal gamming kybord cherry mx red",
     "corsair k70 rgb pro mechanical gaming keyboard cherry mx red"),
    ("senhiser hd 800 s referenc open back hedphones audiphile",
     "sennheiser hd 800 s reference open back headphones audiophile"),
    ("shre mv7 pls podcast dynamic microfone usb xlr",
     "shure mv7 plus podcast dynamic microphone usb xlr"),
    ("elgto cam link 4k externl vidoe capture crd hdmi",
     "elgato cam link 4k external video capture card hdmi"),
    ("rde procastor broadcast quailty dynamic microfone xlr",
     "rode procaster broadcast quality dynamic microphone xlr"),
    ("blu yeti x profesional condensr usb microfone for streming",
     "blue yeti x professional condenser usb microphone for streaming"),
    ("audio technia at2020 cardiod condensr studio xlr microfone",
     "audio technica at2020 cardioid condenser studio xlr microphone"),
    ("fifne ampligame a8 usb gamming microfone rgb cardiod",
     "fifine ampligame a8 usb gaming microphone rgb cardioid"),
    ("samsng odyssy neo g9 57 inch dul 4k 240hz curvd moniter",
     "samsung odyssey neo g9 57 inch dual 4k 240hz curved monitor"),
    ("asus rog swft pg27aqdm 26.5 inch 1440p oledd 240hz moniter",
     "asus rog swift pg27aqdm 26.5 inch 1440p oled 240hz monitor"),
    ("lg 27gp950 b ultragar 27 inch 4k nano ips 144hz gamming moniter",
     "lg 27gp950 b ultragear 27 inch 4k nano ips 144hz gaming monitor"),
    ("alienwar aw3423dwf 34 inch curvd qd oledd 165hz ultawide",
     "alienware aw3423dwf 34 inch curved qd oled 165hz ultrawide"),
    ("aple ipad pro 12.9 inch m4 wifii celluar 1tb space gry",
     "apple ipad pro 12.9 inch m4 wifi cellular 1tb space gray"),
    ("microsft surface pro 9 inel core i7 16gb 256gb plat",
     "microsoft surface pro 9 intel core i7 16gb 256gb platinum"),
    ("googel pixel 8 pro 256gb obsidian unloked smartfone",
     "google pixel 8 pro 256gb obsidian unlocked smartphone"),
    ("oneplus 12 5g 16gb 512gb fluid emerld unloked",
     "oneplus 12 5g 16gb 512gb fluid emerald unlocked"),
    ("xiaomi 14 ultra 16gb 1tb leica camera flagship",
     "xiaomi 14 ultra 16gb 1tb leica camera flagship"),
    ("nothing phone 2a plus 12gb 256gb glyph interfac",
     "nothing phone 2a plus 12gb 256gb glyph interface"),
]
for i, (q, e) in enumerate(L3_LONG):
    rows.append(row(f"L3-LNG-{i:03d}", q, e, "long", "L3", "double_edit_or_transpose", ""))


# ---------------------------------------------------------------------------
# L4 — HARD: dense multi-edit, brand corruption, phonetic, ED-2 deep. 105.
# ---------------------------------------------------------------------------

L4_SHORT = [
    ("smartfone",       "smartphone"),
    ("airepods",        "airpods"),
    ("hedphons",        "headphones"),
    ("microfone",       "microphone"),
    ("wirelss",         "wireless"),
    ("bluthooth",       "bluetooth"),
    ("noiscanceling",   "noise canceling"),
    ("powrbnk",         "power bank"),
    ("gpu rtx 4090ti",  "gpu rtx 4090 ti"),
    ("ddr5 32gigs",     "ddr5 32gb"),
    ("ssd 2terabyte",   "ssd 2tb"),
    ("monitor 144 hertz","monitor 144hz"),
    ("4k tellevizon",   "4k television"),
    ("ergmonomic mose", "ergonomic mouse"),
    ("mecan keybd",     "mechanical keyboard"),
    ("solidstat drv",   "solid state drive"),
    ("graficscard",     "graphics card"),
    ("smartwch",        "smart watch"),
    ("gameing pc",      "gaming pc"),
    ("hdmiicable",      "hdmi cable"),
    ("usbcable",        "usb cable"),
    ("wirlsearbud",     "wireless earbud"),
    ("blutoothspeker",  "bluetooth speaker"),
    ("noiscanceling hp","noise canceling headphone"),
    ("vrheadset",       "vr headset"),
]
for i, (q, e) in enumerate(L4_SHORT):
    rows.append(row(f"L4-SHO-{i:03d}", q, e, "short", "L4", "dense_phonetic", "ED-2+ or phonetic"))

# Brand-protection-with-typo (must correct typo, not corrupt brand)
L4_MEDIUM_BRAND = [
    ("logitch g502 mose",                 "logitech g502 mouse"),
    ("razzer basilisk mose",              "razer basilisk mouse"),
    ("bsose quietcomfrt headfones",       "bose quietcomfort headphones"),
    ("seinhiser hd650 hedphones",         "sennheiser hd650 headphones"),
    ("samsng galaxy s24 ultraa fone",     "samsung galaxy s24 ultra phone"),
    ("aplle iphone 15 pro maxx",          "apple iphone 15 pro max"),
    ("nvideea rtx 4090 ti card",          "nvidia rtx 4090 ti card"),
    ("amdd ryzen 9 7950x procesor",       "amd ryzen 9 7950x processor"),
    ("intell core i9 14900k cpuu",        "intel core i9 14900k cpu"),
    ("dle xps 15 laptp 16gb ram",         "dell xps 15 laptop 16gb ram"),
    ("hpp envy 17 inch laptp",            "hp envy 17 inch laptop"),
    ("lenova thinkpad x1 carbn gen 11",   "lenovo thinkpad x1 carbon gen 11"),
    ("asuss rog strix moniter 27in",      "asus rog strix monitor 27in"),
    ("acerr predator helios 16 laptp",    "acer predator helios 16 laptop"),
    ("msii mag 274qrf gamming moniter",   "msi mag 274qrf gaming monitor"),
    ("corsiar k70 rgb mecan keybd",       "corsair k70 rgb mechanical keyboard"),
    ("logitech mxx master 3 mose",        "logitech mx master 3 mouse"),
    ("razerr huntsman elite kybd",        "razer huntsman elite keyboard"),
    ("steelseris apex pro tkll kybd",     "steelseries apex pro tkl keyboard"),
    ("hyperxx cloud 2 hedset",            "hyperx cloud 2 headset"),
    ("anke r 737 powr bank 24kmah",       "anker 737 power bank 24000mah"),
    ("ugren nexod 100w gan chargr",       "ugreen nexode 100w gan charger"),
    ("sateci usbc hub for mcbook",        "satechi usb-c hub for macbook"),
    ("ekgato stream dec mk 2",            "elgato stream deck mk 2"),
    ("rde procastor microfone",           "rode procaster microphone"),
]
for i, (q, e) in enumerate(L4_MEDIUM_BRAND):
    rows.append(row(f"L4-MED-{i:03d}", q, e, "medium", "L4", "brand_corrupt_with_typo",
                    "brand has typo, must restore correctly"))

L4_LONG = [
    ("aple macbok prr 16 inch m3 mxx 64gb 2tb spac blak alumnium",
     "apple macbook pro 16 inch m3 max 64gb 2tb space black aluminum"),
    ("samsng galxy s24 ultraa 512gb titaniumm blk unloked dual sim",
     "samsung galaxy s24 ultra 512gb titanium black unlocked dual sim"),
    ("asus rog strx scar 18 g834 gamming laptp rtx 4090 24gb ddr5",
     "asus rog strix scar 18 g834 gaming laptop rtx 4090 24gb ddr5"),
    ("razerr blad 18 2024 intl core i9 rtx 4080 240hz qhdd 32gb",
     "razer blade 18 2024 intel core i9 rtx 4080 240hz qhd 32gb"),
    ("lg ulragar 27 inch 4k oledd gamming moniter 240hz hdmi 21",
     "lg ultragear 27 inch 4k oled gaming monitor 240hz hdmi 2.1"),
    ("logitec mx mater 3s mac perfmance wirless mose ergonmic",
     "logitech mx master 3s mac performance wireless mouse ergonomic"),
    ("sony wh1000xm5 wirless nois cancling hedphones blk leather",
     "sony wh-1000xm5 wireless noise canceling headphones black leather"),
    ("bsose quitcomfort ultraa wirless nois cancling earbud blk",
     "bose quietcomfort ultra wireless noise canceling earbuds black"),
    ("corsiar hs80 mxx wirless gamming hedset rgb dolby atmos",
     "corsair hs80 max wireless gaming headset rgb dolby atmos"),
    ("nvideea gefoce rtx 4090 24gb gdr6x graficscard fonders editon",
     "nvidia geforce rtx 4090 24gb gddr6x graphics card founders edition"),
    ("amdd ryzn 9 7950x3d 16 core 32 thrd dektop procesor",
     "amd ryzen 9 7950x3d 16 core 32 thread desktop processor"),
    ("intl cor i9 14900k unloked desktp procesor 24 corres 32 thread",
     "intel core i9 14900k unlocked desktop processor 24 cores 32 threads"),
    ("samsng 990 pro 2tb pcie 4 nvme m 2 internl sdd gamming",
     "samsung 990 pro 2tb pcie 4 nvme m 2 internal ssd gaming"),
    ("westrn digtl blackk sn850x 1tb nvme internl gamming sdd",
     "western digital black sn850x 1tb nvme internal gaming ssd"),
    ("asuss rog strx x670e creatr wifii atx motherbord ddr5",
     "asus rog strix x670e creator wifi atx motherboard ddr5"),
    ("lg c3 65 inch oledd evoo 4k smar tvv self lit pixells dolby",
     "lg c3 65 inch oled evo 4k smart tv self lit pixels dolby"),
    ("sony braviar xr 75 inch a95l qd oledd 4k hdr smar tvv 120hz",
     "sony bravia xr 75 inch a95l qd oled 4k hdr smart tv 120hz"),
    ("ankr 737 powr bnk 24kmah 140w usbc portbl chargr blk",
     "anker 737 power bank 24000mah 140w usb-c portable charger black"),
    ("ugren nexod 100w usbc gan chargr 4 port fast charing",
     "ugreen nexode 100w usb-c gan charger 4 port fast charging"),
    ("sateci usbc hub multport adaptr prr macbok pro 14 inch",
     "satechi usb-c hub multiport adapter pro macbook pro 14 inch"),
    ("razerr basilik v3 pro wirless ergnomic gamming mose 30000dpi",
     "razer basilisk v3 pro wireless ergonomic gaming mouse 30000dpi"),
    ("logitec g pro x superlite 2 wirless gamming mose hero 2 sensr",
     "logitech g pro x superlight 2 wireless gaming mouse hero 2 sensor"),
    ("corsiar k70 rgb prr mecan gamming kybord cherry mx red switc",
     "corsair k70 rgb pro mechanical gaming keyboard cherry mx red switch"),
    ("senhiser hd 800 s referenc open bak hedphones audiphile",
     "sennheiser hd 800 s reference open back headphones audiophile"),
    ("shre mv7 pls podcst dynmic microfone usb xlr broadcast",
     "shure mv7 plus podcast dynamic microphone usb xlr broadcast"),
    ("elgto cam linkk 4k externl vidoe capture crd hdmi",
     "elgato cam link 4k external video capture card hdmi"),
    ("rde procastor broadcst qualty dynmic microfone xlr",
     "rode procaster broadcast quality dynamic microphone xlr"),
    ("blu yeti x profesional condensr usb microfone for streming",
     "blue yeti x professional condenser usb microphone for streaming"),
    ("audi technia at2020 cardiod condensr studi xlr microfone",
     "audio technica at2020 cardioid condenser studio xlr microphone"),
    ("fifne ampligme a8 usb gamming microfone rgb cardiod",
     "fifine ampligame a8 usb gaming microphone rgb cardioid"),
    ("samsng odyssy neo g9 57 inch dul 4k 240hz curvd moniter rgb",
     "samsung odyssey neo g9 57 inch dual 4k 240hz curved monitor rgb"),
    ("asuss rog swft pg27aqdm 26.5 inch 1440p oledd 240hz moniter",
     "asus rog swift pg27aqdm 26.5 inch 1440p oled 240hz monitor"),
    ("lg 27gp950 b ultragar 27 inch 4k nano ips 144hz gamming moniter",
     "lg 27gp950 b ultragear 27 inch 4k nano ips 144hz gaming monitor"),
    ("alienwar aw3423dwf 34 inch curvd qd oledd 165hz ultawide gamming",
     "alienware aw3423dwf 34 inch curved qd oled 165hz ultrawide gaming"),
    ("aple ipd prr 12.9 inch m4 wifii celluar 1tb spac gry brand new",
     "apple ipad pro 12.9 inch m4 wifi cellular 1tb space gray brand new"),
    ("microsf surfac pro 9 inel core i7 16gb 256gb platinm",
     "microsoft surface pro 9 intel core i7 16gb 256gb platinum"),
    ("googel pixl 8 pro 256gb obsdian unloked smartfone",
     "google pixel 8 pro 256gb obsidian unlocked smartphone"),
    ("oneplus 12 5g 16gb 512gb fluid emerld unloked smartfone",
     "oneplus 12 5g 16gb 512gb fluid emerald unlocked smartphone"),
    ("xaomi 14 ultraa 16gb 1tb leica camra flagship smarphone",
     "xiaomi 14 ultra 16gb 1tb leica camera flagship smartphone"),
    ("nothng phone 2a plus 12gb 256gb glyph interfac smarphone",
     "nothing phone 2a plus 12gb 256gb glyph interface smartphone"),
    ("dson v15 detec absolute cordless vacum cleanr",
     "dyson v15 detect absolute cordless vacuum cleaner"),
    ("brevill barissta express espresso machne stainles steel",
     "breville barista express espresso machine stainless steel"),
    ("nintndo switch oledd model with white joy con consle",
     "nintendo switch oled model with white joy con console"),
]
for i, (q, e) in enumerate(L4_LONG):
    rows.append(row(f"L4-LNG-{i:03d}", q, e, "long", "L4", "dense_compound_typo",
                    "multi-token dense errors"))


# ---------------------------------------------------------------------------
# L5 — VERY HARD / ADVERSARIAL: very dense, mixed-error, ED-3+. 100.
# ---------------------------------------------------------------------------

# Very dense single-token corruptions
L5_SHORT = [
    ("hedphns",         "headphones"),
    ("micrfn",          "microphone"),
    ("wirels",          "wireless"),
    ("bltoth",          "bluetooth"),
    ("nseclngn",        "noise canceling"),
    ("pwrbnk",          "power bank"),
    ("ergnmcmus",       "ergonomic mouse"),
    ("mecnclkybd",      "mechanical keyboard"),
    ("ssdtb",           "ssd 1tb"),
    ("smrtwch",         "smart watch"),
    ("gmnpc",           "gaming pc"),
    ("hdmcbl",          "hdmi cable"),
    ("usbcbl",          "usb cable"),
    ("blsspkr",         "bluetooth speaker"),
    ("vrhdst",          "vr headset"),
    ("4ktv",            "4k tv"),
    ("monitr 144hz",    "monitor 144hz"),
    ("kybd mecan",      "mechanical keyboard"),
    ("rtx4090tii",      "rtx 4090 ti"),
    ("ddr532gigs",      "ddr5 32gb"),
]
for i, (q, e) in enumerate(L5_SHORT):
    rows.append(row(f"L5-SHO-{i:03d}", q, e, "short", "L5", "extreme_compression",
                    "vowels dropped, joined tokens"))

# Brand+typo extreme
L5_MEDIUM = [
    ("aplee macbok prr m3",                "apple macbook pro m3"),
    ("samsng galxy s24 ultraa fne",        "samsung galaxy s24 ultra phone"),
    ("logitch g502 mose gamming",          "logitech g502 mouse gaming"),
    ("rzr basilisk mose wirless",          "razer basilisk mouse wireless"),
    ("bsose quietcomft hedfones",          "bose quietcomfort headphones"),
    ("senhiser hd 660s2 hedfns",           "sennheiser hd 660s2 headphones"),
    ("nvideea rtx 4090 ti grafcrd",        "nvidia rtx 4090 ti graphics card"),
    ("amdd ryzn 9 7950x dektpcpu",         "amd ryzen 9 7950x desktop cpu"),
    ("intell cor i9 14900k unloked",       "intel core i9 14900k unlocked"),
    ("dle xps 15 oledd laptp 16gb",        "dell xps 15 oled laptop 16gb"),
    ("hpp envy 17in laptp 32gb",           "hp envy 17in laptop 32gb"),
    ("lenova thinkpad x1 carbn gen11",     "lenovo thinkpad x1 carbon gen11"),
    ("asuss rog strx moniter 27in 4k",     "asus rog strix monitor 27in 4k"),
    ("acerr predtor helios 16 laptp",      "acer predator helios 16 laptop"),
    ("msii mag 274qrf gamming moniter",    "msi mag 274qrf gaming monitor"),
    ("corsiar k70 rgb prr mecan kybd",     "corsair k70 rgb pro mechanical keyboard"),
    ("logitch mxx mater 3 mose wirless",   "logitech mx master 3 mouse wireless"),
    ("rzrr huntsmn elite kybd rgb",        "razer huntsman elite keyboard rgb"),
    ("steelseris apex prr tkl kybd",       "steelseries apex pro tkl keyboard"),
    ("hyperxxcloud 2 hedset gamming",      "hyperx cloud 2 headset gaming"),
    ("ankr 737 powr bnk 24kmah pd",        "anker 737 power bank 24000mah pd"),
    ("ugren nexod 100w gan chargr 4port",  "ugreen nexode 100w gan charger 4port"),
    ("sateci usbchub for mcbook prr",      "satechi usb-c hub for macbook pro"),
    ("ekgato strem dec mk 2 prr",          "elgato stream deck mk 2 pro"),
    ("rde procstr microfone usbxlr",       "rode procaster microphone usb xlr"),
    ("googel pixl 8 prr 256gb",            "google pixel 8 pro 256gb"),
    ("oneplus12 5g 16gb 512gb",            "oneplus 12 5g 16gb 512gb"),
    ("xaomi 14 ultraa leica cam",          "xiaomi 14 ultra leica camera"),
    ("nothng phn 2a pls glyph",            "nothing phone 2a plus glyph"),
    ("dson v15 detec cordlss vacum",       "dyson v15 detect cordless vacuum"),
]
for i, (q, e) in enumerate(L5_MEDIUM):
    rows.append(row(f"L5-MED-{i:03d}", q, e, "medium", "L5", "extreme_brand_typo",
                    "brand+product extreme typos"))

# Adversarial long compound queries with mixed brand+spec+model errors
L5_LONG = [
    ("aplemcbk prr 16in m3 mxx 64gb 2tb spc blk alminm 1yr",
     "apple macbook pro 16 inch m3 max 64gb 2tb space black aluminum 1 year"),
    ("samsngglxy s24ultraa 512gb titnum blk unlckd dualsim",
     "samsung galaxy s24 ultra 512gb titanium black unlocked dual sim"),
    ("asus rog strxscar18 g834 gamng laptp rtx4090 24gb ddr5 ssd2tb",
     "asus rog strix scar 18 g834 gaming laptop rtx 4090 24gb ddr5 ssd 2tb"),
    ("rzrrblade18 2024 intl i9 rtx4080 240hz qhdd 32gbram",
     "razer blade 18 2024 intel i9 rtx 4080 240hz qhd 32gb ram"),
    ("lg ulragear 27in 4k oledd gamng moniter 240hz hdmi21 disply",
     "lg ultragear 27 inch 4k oled gaming monitor 240hz hdmi 2.1 displayport"),
    ("logitch mxmaster3s for mac perfmance wirless mose ergnmic",
     "logitech mx master 3s for mac performance wireless mouse ergonomic"),
    ("sonywh1000xm5 wirless nois cancling hedphns blk leather plush",
     "sony wh-1000xm5 wireless noise canceling headphones black leather plush"),
    ("bsose quitcomft ultraa wirless nois cancling earbud blk caser",
     "bose quietcomfort ultra wireless noise canceling earbuds black case"),
    ("corsiarhs80mxx wirless gamng hedset rgb dolbyatmos 50hr btr",
     "corsair hs80 max wireless gaming headset rgb dolby atmos 50hr battery"),
    ("nvideeagefoce rtx4090 24gb gdr6x graficscrd fondrs editon limtd",
     "nvidia geforce rtx 4090 24gb gddr6x graphics card founders edition limited"),
    ("amdd ryzn9 7950x3d 16cor 32thrd dektpcpu am5 socket",
     "amd ryzen 9 7950x3d 16 core 32 thread desktop cpu am5 socket"),
    ("intl cor i9 14900k unloked dektpcpu 24cor 32thrd lga1700",
     "intel core i9 14900k unlocked desktop cpu 24 core 32 thread lga1700"),
    ("samsng990pro 2tb pcie4 nvme m2 internl sdd gamng pst5 compatbl",
     "samsung 990 pro 2tb pcie 4 nvme m.2 internal ssd gaming ps5 compatible"),
    ("westrndigtl blk sn850x 1tb nvme internl gamng sdd heatsink",
     "western digital black sn850x 1tb nvme internal gaming ssd heatsink"),
    ("asussrog strx x670e creatr wifii atx motherbord ddr5 ddr5",
     "asus rog strix x670e creator wifi atx motherboard ddr5"),
    ("lgc3 65inch oledd evoo 4k smar tvv selfltpixl dolby atmos",
     "lg c3 65 inch oled evo 4k smart tv self lit pixels dolby atmos"),
    ("sony braviar xr 75in a95l qd oledd 4khdr smar tvv 120hz",
     "sony bravia xr 75 inch a95l qd oled 4k hdr smart tv 120hz"),
    ("ankr 737 powrbnk 24kmah 140w usbc portbl chargr blk slim",
     "anker 737 power bank 24000mah 140w usb-c portable charger black slim"),
    ("ugren nexod 100w usbc gan chargr 4port fast charing macsuprt",
     "ugreen nexode 100w usb-c gan charger 4 port fast charging mac support"),
    ("sateci usbchub multport adaptr prr macbok prr 14 inch",
     "satechi usb-c hub multiport adapter pro macbook pro 14 inch"),
    ("rzrr basilik v3 prr wirless ergnomic gamng mose 30kdpi sensr",
     "razer basilisk v3 pro wireless ergonomic gaming mouse 30000dpi sensor"),
    ("logitch gprx superlite2 wirless gamng mose hero2 sensr 60hr btr",
     "logitech g pro x superlight 2 wireless gaming mouse hero 2 sensor 60hr battery"),
    ("corsiar k70 rgb prr mecan gamng kybord cherrymx red switc opto",
     "corsair k70 rgb pro mechanical gaming keyboard cherry mx red switch opto"),
    ("senhiser hd800s referenc open bak hedphones audiphile 300ohm",
     "sennheiser hd 800 s reference open back headphones audiophile 300ohm"),
    ("shre mv7pls podcst dynmic microfone usbxlr broadcast hybrd",
     "shure mv7 plus podcast dynamic microphone usb xlr broadcast hybrid"),
    ("elgto camlinkk 4k externl vidoe capture crd hdmi 1080p60",
     "elgato cam link 4k external video capture card hdmi 1080p 60"),
    ("rde procstr broadcst qualty dynmic microfone xlr 50ohm",
     "rode procaster broadcast quality dynamic microphone xlr 50 ohm"),
    ("blu yetiX profesional condensr usb microfone streming 4patrn",
     "blue yeti x professional condenser usb microphone streaming 4 pattern"),
    ("audiotechnia at2020 cardiod condensr studi xlr microfone podcst",
     "audio technica at2020 cardioid condenser studio xlr microphone podcast"),
    ("fifne ampligme a8 usbgamng microfone rgb cardiod scissrarm",
     "fifine ampligame a8 usb gaming microphone rgb cardioid scissor arm"),
    ("samsng odyssy neo g9 57in dul4k 240hz curvd moniter rgb 1000nit",
     "samsung odyssey neo g9 57 inch dual 4k 240hz curved monitor rgb 1000 nit"),
    ("asuss rog swft pg27aqdm 26.5in 1440p oledd 240hz moniter glsy",
     "asus rog swift pg27aqdm 26.5 inch 1440p oled 240hz monitor glossy"),
    ("lg27gp950b ultragar 27in 4k nanoips 144hz gamng moniter usbc",
     "lg 27gp950b ultragear 27 inch 4k nano ips 144hz gaming monitor usb-c"),
    ("alienwar aw3423dwf 34in curvd qd oledd 165hz ultawide gamng dp21",
     "alienware aw3423dwf 34 inch curved qd oled 165hz ultrawide gaming dp 2.1"),
    ("apleipd prr 12.9in m4 wifii celluar 1tb spc gry brndnew sealed",
     "apple ipad pro 12.9 inch m4 wifi cellular 1tb space gray brand new sealed"),
    ("microsf surfac prr 9 inel cor i7 16gb 256gb platinm wifi",
     "microsoft surface pro 9 intel core i7 16gb 256gb platinum wifi"),
    ("googel pixl8 prr 256gb obsdian unloked smartfone factry",
     "google pixel 8 pro 256gb obsidian unlocked smartphone factory"),
    ("oneplus12 5g 16gb 512gb fluid emerld unloked smartfone naa",
     "oneplus 12 5g 16gb 512gb fluid emerald unlocked smartphone na"),
    ("xaomi14 ultraa 16gb 1tb leica camra flagship smarphone glblrom",
     "xiaomi 14 ultra 16gb 1tb leica camera flagship smartphone global rom"),
    ("nothng phn 2aplus 12gb 256gb glyph interfac smarphone andr14",
     "nothing phone 2a plus 12gb 256gb glyph interface smartphone android 14"),
    ("dson v15 detec absolute cordlss vacum cleanr lasr dust dtct",
     "dyson v15 detect absolute cordless vacuum cleaner laser dust detect"),
    ("brevill barissta exprss espreso machne stainles steel 15bar",
     "breville barista express espresso machine stainless steel 15 bar"),
    ("nintndo swich oledd modl whte joy cn consle 64gb",
     "nintendo switch oled model white joy con console 64gb"),
    ("playstation5 sla edtn dsk slot blry consle 1tb",
     "playstation 5 slim edition disk slot blu ray console 1tb"),
    ("xboxsereies x 1tb consle wirless controler 4k 120hz",
     "xbox series x 1tb console wireless controller 4k 120hz"),
    ("dlinkk dir x5460 wifi6e routr ax5400 gigabit",
     "dlink dir x5460 wifi 6e router ax5400 gigabit"),
    ("netgear orbi rbk863s wifi6e mesh routr tri band",
     "netgear orbi rbk863s wifi 6e mesh router tri band"),
    ("ubiquiti unfi dream machne pro routr enterprise",
     "ubiquiti unifi dream machine pro router enterprise"),
    ("synology ds923 plus 4 bay nas storg ryzn embeded",
     "synology ds923 plus 4 bay nas storage ryzen embedded"),
    ("qnap ts464 4 bay desktp nas intl celron quadcr",
     "qnap ts464 4 bay desktop nas intel celeron quad core"),
]
for i, (q, e) in enumerate(L5_LONG):
    rows.append(row(f"L5-LNG-{i:03d}", q, e, "long", "L5", "adversarial_compound",
                    "extreme dense compound queries"))


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def main():
    print(f"Total queries: {len(rows)}")

    by_difficulty = {}
    by_length = {}
    for r in rows:
        by_difficulty[r["difficulty"]] = by_difficulty.get(r["difficulty"], 0) + 1
        by_length[r["length"]] = by_length.get(r["length"], 0) + 1

    print("By difficulty:", by_difficulty)
    print("By length:", by_length)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
