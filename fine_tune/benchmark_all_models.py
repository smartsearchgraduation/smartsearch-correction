#!/usr/bin/env python3
"""
Comprehensive Benchmark Test for All Fine-tuned Typo Correction Models
======================================================================

Tests 60 queries with varying difficulty levels:
- Easy (10): 1-2 typos, common brands
- Medium (20): 2-3 typos, mixed brands/products  
- Hard (15): 3-4 typos, complex queries
- Extreme (15): 4+ typos, severely corrupted

Each model is tested separately to avoid memory issues.
"""

import time
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import torch

# Test dataset with difficulty levels
BENCHMARK_QUERIES = {
    "easy": [
        # 1-2 typos, common brands - should be easy to fix
        ("iphnoe 15 pro max", "iphone 15 pro max"),
        ("samsng galaxy s24", "samsung galaxy s24"),
        ("macbok air m2", "macbook air m2"),
        ("airpods pro 2", "airpods pro 2"),  # No typo - should preserve
        ("nvdia rtx 4090", "nvidia rtx 4090"),
        ("logitec mouse wireless", "logitech mouse wireless"),
        ("sonny headphones", "sony headphones"),
        ("dell monitör 27 inch", "dell monitor 27 inch"),
        ("aple watch ultra", "apple watch ultra"),
        ("razr keyboard gaming", "razer keyboard gaming"),
    ],
    "medium": [
        # 2-3 typos, mixed complexity
        ("samsng galxy s24 ultra", "samsung galaxy s24 ultra"),
        ("macbok pro m3 max", "macbook pro m3 max"),
        ("iphnoe 15 pro kılıf", "iphone 15 pro kılıf"),
        ("nvdia gefrce rtx 4080", "nvidia geforce rtx 4080"),
        ("logitrch mx master 3", "logitech mx master 3"),
        ("airpds pro 2 case", "airpods pro 2 case"),
        ("samsng buds 2 pro", "samsung buds 2 pro"),
        ("aple macbok charger", "apple macbook charger"),
        ("asus rog strix laptpo", "asus rog strix laptop"),
        ("lenvo thinkpad x1 carbon", "lenovo thinkpad x1 carbon"),
        ("hp spectr x360 laptop", "hp spectre x360 laptop"),
        ("msi gamng monitor 144hz", "msi gaming monitor 144hz"),
        ("corsiar vengeance ram", "corsair vengeance ram"),
        ("kingston fury ddr5 32gb", "kingston fury ddr5 32gb"),  # No typo
        ("segate barracuda 2tb", "seagate barracuda 2tb"),
        ("wstrn digital ssd 1tb", "western digital ssd 1tb"),
        ("creativ sound blaster", "creative sound blaster"),
        ("steelsries arctis nova", "steelseries arctis nova"),
        ("hpyer x cloud ii", "hyperx cloud ii"),
        ("benq zowie gaming mouse", "benq zowie gaming mouse"),  # No typo
    ],
    "hard": [
        # 3-4 typos, complex multi-word queries
        ("samsng galxy bds 2 pro", "samsung galaxy buds 2 pro"),
        ("aple mcbook pro m3 mxa", "apple macbook pro m3 max"),
        ("nvdia gefroce rtx 4070 ti", "nvidia geforce rtx 4070 ti"),
        ("logitch mx keybord wireless", "logitech mx keyboard wireless"),
        ("razre blackwidow v4 pro", "razer blackwidow v4 pro"),
        ("crosair k100 rgb keybrd", "corsair k100 rgb keyboard"),
        ("asus zenbok pro duo oled", "asus zenbook pro duo oled"),
        ("micorsoft surface pro 9", "microsoft surface pro 9"),
        ("aple ipda pro 12.9 m2", "apple ipad pro 12.9 m2"),
        ("samsng odyysey g9 49 inch", "samsung odyssey g9 49 inch"),
        ("dji mavci 3 pro drone", "dji mavic 3 pro drone"),
        ("gopor hero 12 black", "gopro hero 12 black"),
        ("canno eos r5 mirrorless", "canon eos r5 mirrorless"),
        ("sono beam soundbar gen2", "sonos beam soundbar gen2"),
        ("boes quietcomfort ultra", "bose quietcomfort ultra"),
    ],
    "extreme": [
        # 4+ typos, severely corrupted - very challenging
        ("smasng glxy s24 ultr kılf", "samsung galaxy s24 ultra kılıf"),
        ("aple mcbok air m2 chrger", "apple macbook air m2 charger"),
        ("nvda gfrce rtx 4090 grphc", "nvidia geforce rtx 4090 graphic"),
        ("lgtch mx mstr 3s mouse", "logitech mx master 3s mouse"),
        ("razr blckwidw v4 pro kybd", "razer blackwidow v4 pro keyboard"),
        ("crsar k100 rgb mechnical", "corsair k100 rgb mechanical"),
        ("iphn 15 pr mx spcgray", "iphone 15 pro max spacegray"),
        ("smsng glaxy buds fe case", "samsung galaxy buds fe case"),
        ("apl wtch ultr 2 titanum", "apple watch ultra 2 titanium"),
        ("nvda shld tv pro 2019", "nvidia shield tv pro 2019"),
        ("lnvo thkpad x1 crbn gen11", "lenovo thinkpad x1 carbon gen11"),
        ("dl xps 15 9530 oltd oled", "dell xps 15 9530 oled"),
        ("micrsft srfc lptp stdo 2", "microsoft surface laptop studio 2"),
        ("assu rog zphyrs g14 2024", "asus rog zephyrus g14 2024"),
        ("sny wh1000xm5 headphnes", "sony wh1000xm5 headphones"),
    ],
}


def load_model_tester(model_name: str):
    """Load a specific model tester."""
    from test_all_finetuned_models import (
        ByT5Tester, QwenTester, LlamaTester, Phi3Tester,
        find_model_path, OUTPUT_DIR
    )
    
    if model_name == "byt5":
        path = find_model_path("byt5")
        if path:
            return ByT5Tester(str(path))
    elif model_name == "qwen-0.5b":
        qwen_dir = OUTPUT_DIR / "qwen-typo"
        if qwen_dir.exists():
            for p in sorted(qwen_dir.glob("*0.5b*-final"), reverse=True):
                return QwenTester(str(p), "0.5B")
    elif model_name == "qwen-1.5b":
        qwen_dir = OUTPUT_DIR / "qwen-typo"
        if qwen_dir.exists():
            for p in sorted(qwen_dir.glob("*1.5b*-final"), reverse=True):
                return QwenTester(str(p), "1.5B")
    elif model_name == "llama":
        path = find_model_path("llama")
        if path:
            return LlamaTester(str(path))
    elif model_name == "phi3":
        path = find_model_path("phi3")
        if path:
            return Phi3Tester(str(path))
    
    return None


def calculate_accuracy(predicted: str, expected: str) -> Tuple[bool, float]:
    """Calculate if prediction matches expected and similarity score."""
    pred_lower = predicted.lower().strip()
    exp_lower = expected.lower().strip()
    
    exact_match = pred_lower == exp_lower
    
    # Calculate word-level similarity
    pred_words = set(pred_lower.split())
    exp_words = set(exp_lower.split())
    
    if len(exp_words) == 0:
        return exact_match, 1.0 if exact_match else 0.0
    
    intersection = pred_words & exp_words
    similarity = len(intersection) / len(exp_words)
    
    return exact_match, similarity


def test_single_model(model_name: str) -> Dict:
    """Test a single model on all benchmark queries."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {model_name.upper()}")
    print(f"{'='*60}")
    
    tester = load_model_tester(model_name)
    if tester is None:
        print(f"❌ Model not found: {model_name}")
        return None
    
    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "by_difficulty": {},
        "total": {
            "correct": 0,
            "total": 0,
            "accuracy": 0.0,
            "avg_latency_ms": 0.0,
            "avg_similarity": 0.0,
        },
        "details": []
    }
    
    all_latencies = []
    all_similarities = []
    
    for difficulty, queries in BENCHMARK_QUERIES.items():
        print(f"\n📊 {difficulty.upper()} ({len(queries)} queries)")
        print("-" * 50)
        
        difficulty_results = {
            "correct": 0,
            "total": len(queries),
            "accuracy": 0.0,
            "avg_latency_ms": 0.0,
        }
        
        latencies = []
        
        for typo_query, expected in queries:
            result = tester.test(typo_query)
            
            if result.get("error"):
                print(f"  ❌ {typo_query[:30]:<30} → ERROR")
                results["details"].append({
                    "difficulty": difficulty,
                    "input": typo_query,
                    "expected": expected,
                    "output": "ERROR",
                    "correct": False,
                    "similarity": 0.0,
                    "latency_ms": 0,
                })
                continue
            
            predicted = result["output"]
            latency = result["latency_ms"]
            exact_match, similarity = calculate_accuracy(predicted, expected)
            
            latencies.append(latency)
            all_latencies.append(latency)
            all_similarities.append(similarity)
            
            if exact_match:
                difficulty_results["correct"] += 1
                results["total"]["correct"] += 1
                status = "✅"
            else:
                status = "❌" if similarity < 0.5 else "⚠️"
            
            results["total"]["total"] += 1
            
            # Print compact result
            pred_short = predicted[:25] + "..." if len(predicted) > 25 else predicted
            print(f"  {status} {typo_query[:25]:<25} → {pred_short:<28} ({latency:.0f}ms)")
            
            results["details"].append({
                "difficulty": difficulty,
                "input": typo_query,
                "expected": expected,
                "output": predicted,
                "correct": exact_match,
                "similarity": similarity,
                "latency_ms": latency,
            })
        
        if latencies:
            difficulty_results["avg_latency_ms"] = sum(latencies) / len(latencies)
        difficulty_results["accuracy"] = difficulty_results["correct"] / difficulty_results["total"] * 100
        
        results["by_difficulty"][difficulty] = difficulty_results
        print(f"  → Accuracy: {difficulty_results['correct']}/{difficulty_results['total']} ({difficulty_results['accuracy']:.1f}%)")
    
    # Calculate totals
    if all_latencies:
        results["total"]["avg_latency_ms"] = sum(all_latencies) / len(all_latencies)
    if all_similarities:
        results["total"]["avg_similarity"] = sum(all_similarities) / len(all_similarities)
    if results["total"]["total"] > 0:
        results["total"]["accuracy"] = results["total"]["correct"] / results["total"]["total"] * 100
    
    return results


def print_summary(all_results: List[Dict]):
    """Print summary comparison table."""
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Header
    print(f"\n{'Model':<15} {'Easy':<10} {'Medium':<10} {'Hard':<10} {'Extreme':<10} {'TOTAL':<10} {'Latency':<10}")
    print("-" * 80)
    
    for r in all_results:
        if r is None:
            continue
        
        easy = r["by_difficulty"].get("easy", {}).get("accuracy", 0)
        medium = r["by_difficulty"].get("medium", {}).get("accuracy", 0)
        hard = r["by_difficulty"].get("hard", {}).get("accuracy", 0)
        extreme = r["by_difficulty"].get("extreme", {}).get("accuracy", 0)
        total = r["total"]["accuracy"]
        latency = r["total"]["avg_latency_ms"]
        
        print(f"{r['model']:<15} {easy:>6.1f}%   {medium:>6.1f}%   {hard:>6.1f}%   {extreme:>6.1f}%   {total:>6.1f}%   {latency:>6.0f}ms")
    
    print("-" * 80)
    
    # Find best model
    if all_results:
        valid = [r for r in all_results if r is not None]
        if valid:
            best = max(valid, key=lambda x: x["total"]["accuracy"])
            fastest = min(valid, key=lambda x: x["total"]["avg_latency_ms"])
            print(f"\n🏆 Best Accuracy: {best['model']} ({best['total']['accuracy']:.1f}%)")
            print(f"⚡ Fastest: {fastest['model']} ({fastest['total']['avg_latency_ms']:.0f}ms)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark all fine-tuned models")
    parser.add_argument("--models", nargs="+", 
                        default=["byt5", "qwen-0.5b", "qwen-1.5b", "llama", "phi3"],
                        help="Models to test")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔬 Comprehensive Model Benchmark")
    print("=" * 60)
    print(f"\nTotal queries: {sum(len(q) for q in BENCHMARK_QUERIES.values())}")
    print(f"  - Easy: {len(BENCHMARK_QUERIES['easy'])}")
    print(f"  - Medium: {len(BENCHMARK_QUERIES['medium'])}")
    print(f"  - Hard: {len(BENCHMARK_QUERIES['hard'])}")
    print(f"  - Extreme: {len(BENCHMARK_QUERIES['extreme'])}")
    print(f"\nModels to test: {', '.join(args.models)}")
    
    all_results = []
    
    for model_name in args.models:
        try:
            result = test_single_model(model_name)
            all_results.append(result)
            
            # Clear memory between models
            import gc
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            all_results.append(None)
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = Path(__file__).parent / f"benchmark_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "queries": BENCHMARK_QUERIES,
                "results": [r for r in all_results if r is not None]
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
