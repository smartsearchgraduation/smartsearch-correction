#!/usr/bin/env python3
"""
Generate Long Sentence Training Data for Typo Correction Models
================================================================

Bu script, mevcut typo eğitim verisine uzun cümle örnekleri ekler.
Modellerin sadece kısa kelimeler değil, tam cümleleri düzeltmeyi öğrenmesi için gerekli.

Kullanım:
    python generate_sentence_data.py
    python generate_sentence_data.py --count 2000  # 2000 örnek üret

Çıktı:
    data/train_llm_sentences.jsonl  # Uzun cümleli eğitim verisi
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple

# E-commerce search sentence templates
SENTENCE_TEMPLATES = [
    # Product search patterns
    "I want to buy {product}",
    "I need a {product}",
    "Looking for {product}",
    "Search for {product}",
    "Find me {product}",
    "Show me {product}",
    "I'm looking for {product}",
    "Can you find {product}",
    "Where can I buy {product}",
    "Best {product} deals",
    "Cheap {product} online",
    "{product} for sale",
    "{product} best price",
    "{product} reviews",
    "Buy {product} now",
    
    # Multi-product patterns
    "{product} and {product2}",
    "{product} with {product2}",
    "{product} or {product2}",
    "{product} vs {product2}",
    
    # Specification patterns
    "{product} {spec}",
    "{spec} {product}",
    "{product} with {spec}",
    "Best {product} {spec}",
    
    # Question patterns
    "What is the best {product}",
    "Which {product} should I buy",
    "Is {product} good",
    "How much is {product}",
]

# Product categories with typo mappings
PRODUCTS = {
    # Phones
    ("iphnoe", "iphone"): ["15 pro", "15 pro max", "14", "14 pro", "13", "se"],
    ("samsng", "samsung"): ["galaxy s24", "galaxy s24 ultra", "galaxy a54", "galaxy z fold"],
    ("gogle", "google"): ["pixel 8", "pixel 8 pro", "pixel 7a"],
    ("onplus", "oneplus"): ["12", "12 pro", "nord"],
    ("xaomi", "xiaomi"): ["14", "14 ultra", "redmi note 13"],
    
    # Laptops
    ("macbok", "macbook"): ["pro m3", "pro m3 max", "air m2", "air m3"],
    ("thikpad", "thinkpad"): ["x1 carbon", "t14", "x13"],
    ("dellxps", "dell xps"): ["15", "13", "17"],
    ("asusrog", "asus rog"): ["strix", "zephyrus g14", "flow"],
    ("hpspectr", "hp spectre"): ["x360", "14"],
    ("surfac", "surface"): ["pro 9", "laptop 5", "go 3"],
    
    # Audio
    ("airpds", "airpods"): ["pro 2", "max", "3rd gen"],
    ("sonyxm", "sony wh-1000xm"): ["5", "4"],
    ("boes", "bose"): ["quietcomfort ultra", "700", "soundlink"],
    ("beatsolo", "beats solo"): ["4", "pro"],
    ("jabr", "jabra"): ["elite 85t", "elite 7 pro"],
    
    # Accessories
    ("logitec", "logitech"): ["mx master 3", "g pro x", "mx keys"],
    ("razr", "razer"): ["blackwidow", "deathadder", "kraken"],
    ("corsiar", "corsair"): ["k100", "dark core", "virtuoso"],
    ("steelsries", "steelseries"): ["arctis nova", "apex pro"],
    
    # Graphics Cards
    ("nvdia", "nvidia"): ["rtx 4090", "rtx 4080", "rtx 4070 ti"],
    ("gefrce", "geforce"): ["rtx 4060", "rtx 3080"],
    ("radoen", "radeon"): ["rx 7900 xtx", "rx 7800 xt"],
    
    # Storage
    ("samsngssd", "samsung ssd"): ["990 pro", "980 pro", "870 evo"],
    ("segate", "seagate"): ["barracuda", "firecuda"],
    ("wstrn", "western digital"): ["black sn850x", "blue"],
    
    # Cameras
    ("canno", "canon"): ["eos r5", "eos r6", "m50"],
    ("sonyalph", "sony alpha"): ["a7 iv", "a7c", "a6400"],
    ("gopor", "gopro"): ["hero 12", "hero 11"],
    ("djimavic", "dji mavic"): ["3 pro", "air 2s"],
    
    # Gaming
    ("playstaton", "playstation"): ["5", "5 slim", "vr2"],
    ("xobx", "xbox"): ["series x", "series s"],
    ("nintedo", "nintendo"): ["switch oled", "switch lite"],
}

# Specifications that can be typo'd
SPECS = {
    # Size/Storage
    ("128gn", "128gb"): True,
    ("256gv", "256gb"): True,
    ("512gn", "512gb"): True,
    ("1tb", "1tb"): True,
    ("16gm", "16gb"): True,
    ("32gv", "32gb"): True,
    
    # Colors
    ("blck", "black"): True,
    ("whte", "white"): True,
    ("slver", "silver"): True,
    ("spcgray", "space gray"): True,
    ("midnght", "midnight"): True,
    
    # Features
    ("wirelss", "wireless"): True,
    ("bluetoth", "bluetooth"): True,
    ("gamng", "gaming"): True,
    ("mechancal", "mechanical"): True,
    ("waterpoof", "waterproof"): True,
}

# Secondary products for multi-product queries
SECONDARY_PRODUCTS = [
    ("case", "case"),
    ("charger", "charger"),
    ("cabl", "cable"),
    ("adaptr", "adapter"),
    ("screenprot", "screen protector"),
    ("stand", "stand"),
    ("mount", "mount"),
    ("bag", "bag"),
    ("cover", "cover"),
]


def generate_typo_sentence() -> Tuple[str, str]:
    """Generate a single typo sentence pair."""
    template = random.choice(SENTENCE_TEMPLATES)
    
    # Select random product
    product_key = random.choice(list(PRODUCTS.keys()))
    typo_brand, correct_brand = product_key
    variants = PRODUCTS[product_key]
    variant = random.choice(variants)
    
    typo_product = f"{typo_brand} {variant}"
    correct_product = f"{correct_brand} {variant}"
    
    # Check if template needs second product
    if "{product2}" in template:
        sec_typo, sec_correct = random.choice(SECONDARY_PRODUCTS)
        typo_sentence = template.format(product=typo_product, product2=sec_typo)
        correct_sentence = template.format(product=correct_product, product2=sec_correct)
    elif "{spec}" in template:
        spec_key = random.choice(list(SPECS.keys()))
        typo_spec, correct_spec = spec_key
        typo_sentence = template.format(product=typo_product, spec=typo_spec)
        correct_sentence = template.format(product=correct_product, spec=correct_spec)
    else:
        typo_sentence = template.format(product=typo_product)
        correct_sentence = template.format(product=correct_product)
    
    return typo_sentence, correct_sentence


def generate_multi_typo_sentence() -> Tuple[str, str]:
    """Generate sentence with multiple typos."""
    # Random template with multiple products
    templates = [
        "I want to buy {p1} and {p2}",
        "{p1} {spec} with {p2}",
        "Looking for {p1} {spec} and {p2}",
        "Best deals on {p1} and {p2} {spec}",
        "Show me {p1} {p2} combo",
    ]
    
    template = random.choice(templates)
    
    # First product
    pk1 = random.choice(list(PRODUCTS.keys()))
    typo1, correct1 = pk1
    var1 = random.choice(PRODUCTS[pk1])
    
    # Second product (different)
    pk2 = random.choice([k for k in PRODUCTS.keys() if k != pk1])
    typo2, correct2 = pk2
    var2 = random.choice(PRODUCTS[pk2])
    
    # Spec
    spec_key = random.choice(list(SPECS.keys()))
    typo_spec, correct_spec = spec_key
    
    typo_sentence = template.format(
        p1=f"{typo1} {var1}",
        p2=f"{typo2} {var2}",
        spec=typo_spec
    )
    correct_sentence = template.format(
        p1=f"{correct1} {var1}",
        p2=f"{correct2} {var2}",
        spec=correct_spec
    )
    
    return typo_sentence, correct_sentence


def format_for_llm(typo: str, correct: str) -> dict:
    """Format as LLM training example."""
    return {
        "messages": [
            {"role": "user", "content": typo},
            {"role": "assistant", "content": correct}
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Generate sentence training data")
    parser.add_argument("--count", type=int, default=3000, help="Number of examples")
    parser.add_argument("--output", type=str, default="data/train_sentences.jsonl")
    args = parser.parse_args()
    
    print("=" * 60)
    print("📝 Generating Sentence Training Data")
    print("=" * 60)
    
    examples = []
    
    # Generate single-typo sentences (60%)
    single_count = int(args.count * 0.6)
    print(f"\n⏳ Generating {single_count} single-typo sentences...")
    for _ in range(single_count):
        typo, correct = generate_typo_sentence()
        examples.append(format_for_llm(typo, correct))
    
    # Generate multi-typo sentences (40%)
    multi_count = args.count - single_count
    print(f"⏳ Generating {multi_count} multi-typo sentences...")
    for _ in range(multi_count):
        typo, correct = generate_multi_typo_sentence()
        examples.append(format_for_llm(typo, correct))
    
    # Shuffle
    random.shuffle(examples)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Generated {len(examples)} examples")
    print(f"💾 Saved to: {output_path}")
    
    # Show samples
    print("\n📋 Sample examples:")
    print("-" * 60)
    for i, ex in enumerate(examples[:5]):
        typo = ex["messages"][0]["content"]
        correct = ex["messages"][1]["content"]
        print(f"{i+1}. {typo}")
        print(f"   → {correct}")
        print()


if __name__ == "__main__":
    main()
