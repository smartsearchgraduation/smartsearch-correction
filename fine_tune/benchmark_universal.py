"""
Universal benchmark script for testing typo correction models across e-commerce categories.

Tests ByT5 and T5-large models across 120+ queries spanning electronics, fashion,
beauty, home & kitchen, sports, toys, automotive, and grocery categories.

Includes special handling for brand names, abbreviations, units, and currencies.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    print("Warning: torch/transformers not installed. Install with: pip install torch transformers")
    torch = None


@dataclass
class BenchmarkResult:
    """Single query benchmark result."""
    query: str
    expected: str
    model_output: str
    category: str
    difficulty: str
    exact_match: bool
    word_similarity: float
    latency_ms: float
    is_false_positive: bool
    notes: str = ""


class T5LargeTester:
    """T5-large model inference wrapper for correction tasks."""

    def __init__(self, model_path: str = None, device: str = "cuda" if torch and torch.cuda.is_available() else "cpu"):
        """Initialize T5-large model and tokenizer.

        Args:
            model_path: Path to fine-tuned model checkpoint, or None for default.
                        Searches: outputs/t5-large-typo/best → outputs/t5-large-typo → google-t5/t5-large
            device: torch device (cuda, cpu, etc.)
        """
        self.device = device

        # Resolve model path
        if model_path:
            self.model_name = model_path
        else:
            # Search for fine-tuned checkpoint
            script_dir = Path(__file__).resolve().parent
            candidates = [
                script_dir / "outputs" / "t5-large-typo" / "best",
                script_dir / "outputs" / "t5-large-typo",
                script_dir.parent / "t5-large-typo" / "best",
                script_dir.parent / "t5-large-typo",
            ]
            self.model_name = "google-t5/t5-large"  # fallback
            for c in candidates:
                if c.exists() and (c / "config.json").exists():
                    self.model_name = str(c)
                    break

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
            self.model.eval()
            print(f"Loaded T5-large from {self.model_name} on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.tokenizer = None
            self.model = None

    def correct(self, query: str, max_length: int = 128) -> str:
        """Run typo correction on query.

        Args:
            query: Input query with potential typos
            max_length: Max output length

        Returns:
            Corrected query
        """
        if self.model is None:
            return query

        # Use same prompt format as training data
        prompt = f"correct: {query}"

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=1,
                    do_sample=False
                )
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.strip()
        except Exception as e:
            print(f"Error during inference: {e}")
            return query


class ByteTester:
    """ByT5 model inference wrapper for correction tasks."""

    def __init__(self, device: str = "cuda" if torch and torch.cuda.is_available() else "cpu"):
        """Initialize ByT5 model and tokenizer.

        Args:
            device: torch device (cuda, cpu, etc.)
        """
        self.device = device
        self.model_name = "google/byt5-small"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
            self.model.eval()
            print(f"Loaded {self.model_name} on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.tokenizer = None
            self.model = None

    def correct(self, query: str, max_length: int = 128) -> str:
        """Run typo correction on query.

        Args:
            query: Input query with potential typos
            max_length: Max output length

        Returns:
            Corrected query
        """
        if self.model is None:
            return query

        prompt = f"correct: {query}"

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=1,
                    do_sample=False
                )
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.strip()
        except Exception as e:
            print(f"Error during inference: {e}")
            return query


class BenchmarkDataset:
    """Comprehensive benchmark dataset for e-commerce typo correction."""

    def __init__(self):
        """Initialize benchmark dataset with 120+ test queries."""
        self.queries = self._build_dataset()

    def _build_dataset(self) -> List[Dict[str, str]]:
        """Build comprehensive test dataset.

        Returns:
            List of query dictionaries with keys: query, expected, category, difficulty
        """
        dataset = []

        # Electronics (20 queries)
        dataset.extend([
            # Easy
            {"query": "iphone 15 pro", "expected": "iphone 15 pro", "category": "electronics", "difficulty": "easy"},
            {"query": "wirelss headphones", "expected": "wireless headphones", "category": "electronics", "difficulty": "easy"},
            {"query": "4k monitor", "expected": "4k monitor", "category": "electronics", "difficulty": "easy"},
            # Medium
            {"query": "smarpthone charger", "expected": "smartphone charger", "category": "electronics", "difficulty": "medium"},
            {"query": "ssd 1tb nvme", "expected": "ssd 1tb nvme", "category": "electronics", "difficulty": "medium"},
            {"query": "bluethoth speaker", "expected": "bluetooth speaker", "category": "electronics", "difficulty": "medium"},
            {"query": "usb-c cable 2m", "expected": "usb-c cable 2m", "category": "electronics", "difficulty": "medium"},
            # Hard
            {"query": "grapics card rtx 4090", "expected": "graphics card rtx 4090", "category": "electronics", "difficulty": "hard"},
            {"query": "mechnaical keyboard rgb", "expected": "mechanical keyboard rgb", "category": "electronics", "difficulty": "hard"},
            {"query": "wireles ergo mouse", "expected": "wireless ergo mouse", "category": "electronics", "difficulty": "hard"},
            # No change cases (should NOT be modified)
            {"query": "ASUS ROG laptop", "expected": "ASUS ROG laptop", "category": "electronics", "difficulty": "easy"},
            {"query": "32 gb ram ddr4", "expected": "32 gb ram ddr4", "category": "electronics", "difficulty": "easy"},
            {"query": "5 pcs usb adapter", "expected": "5 pcs usb adapter", "category": "electronics", "difficulty": "easy"},
            # Additional queries
            {"query": "lapto bag 15 inch", "expected": "laptop bag 15 inch", "category": "electronics", "difficulty": "medium"},
            {"query": "photograpy lighting kit", "expected": "photography lighting kit", "category": "electronics", "difficulty": "hard"},
            {"query": "webing camera 1080p", "expected": "webcam 1080p", "category": "electronics", "difficulty": "hard"},
            {"query": "gamin mouse pad", "expected": "gaming mouse pad", "category": "electronics", "difficulty": "easy"},
            {"query": "$199.99 smarpwatch", "expected": "$199.99 smartwatch", "category": "electronics", "difficulty": "medium"},
            {"query": "POCO smartphone", "expected": "POCO smartphone", "category": "electronics", "difficulty": "easy"},
            {"query": "wireles charger pad", "expected": "wireless charger pad", "category": "electronics", "difficulty": "easy"},
        ])

        # Fashion (15 queries)
        dataset.extend([
            # Easy
            {"query": "cotton t-shirt", "expected": "cotton t-shirt", "category": "fashion", "difficulty": "easy"},
            {"query": "denim jeans black", "expected": "denim jeans black", "category": "fashion", "difficulty": "easy"},
            # Medium
            {"query": "cashmere sweeter", "expected": "cashmere sweater", "category": "fashion", "difficulty": "medium"},
            {"query": "athlesic wear", "expected": "athletic wear", "category": "fashion", "difficulty": "medium"},
            {"query": "lether jacket", "expected": "leather jacket", "category": "fashion", "difficulty": "medium"},
            # Hard
            {"query": "syntetc fabric dress", "expected": "synthetic fabric dress", "category": "fashion", "difficulty": "hard"},
            {"query": "polyeseter blend shirt", "expected": "polyester blend shirt", "category": "fashion", "difficulty": "hard"},
            # No change
            {"query": "SHEIN dress xxl", "expected": "SHEIN dress xxl", "category": "fashion", "difficulty": "easy"},
            {"query": "2 pcs sock pack", "expected": "2 pcs sock pack", "category": "fashion", "difficulty": "easy"},
            {"query": "$29.99 pants", "expected": "$29.99 pants", "category": "fashion", "difficulty": "easy"},
            # Additional
            {"query": "wool socks warm", "expected": "wool socks warm", "category": "fashion", "difficulty": "easy"},
            {"query": "formal neckty", "expected": "formal necktie", "category": "fashion", "difficulty": "medium"},
            {"query": "silk scurf", "expected": "silk scarf", "category": "fashion", "difficulty": "medium"},
            {"query": "embro jacket", "expected": "embroider jacket", "category": "fashion", "difficulty": "hard"},
            {"query": "velcro shoes", "expected": "velcro shoes", "category": "fashion", "difficulty": "easy"},
        ])

        # Beauty (15 queries)
        dataset.extend([
            # Easy
            {"query": "face cream", "expected": "face cream", "category": "beauty", "difficulty": "easy"},
            {"query": "mascara black", "expected": "mascara black", "category": "beauty", "difficulty": "easy"},
            # Medium
            {"query": "moistrizer lotion", "expected": "moisturizer lotion", "category": "beauty", "difficulty": "medium"},
            {"query": "fundation makeup", "expected": "foundation makeup", "category": "beauty", "difficulty": "medium"},
            {"query": "lipstic red", "expected": "lipstick red", "category": "beauty", "difficulty": "medium"},
            # Hard
            {"query": "serrum vitamin c", "expected": "serum vitamin c", "category": "beauty", "difficulty": "hard"},
            {"query": "conceler brush set", "expected": "concealer brush set", "category": "beauty", "difficulty": "hard"},
            # No change
            {"query": "50 ml cream", "expected": "50 ml cream", "category": "beauty", "difficulty": "easy"},
            {"query": "spf 30 sunscreen", "expected": "spf 30 sunscreen", "category": "beauty", "difficulty": "easy"},
            {"query": "$25.99 perfume", "expected": "$25.99 perfume", "category": "beauty", "difficulty": "easy"},
            # Additional
            {"query": "eye shodow palette", "expected": "eye shadow palette", "category": "beauty", "difficulty": "medium"},
            {"query": "bb cream lighweight", "expected": "bb cream lightweight", "category": "beauty", "difficulty": "medium"},
            {"query": "brow pencil duo", "expected": "brow pencil duo", "category": "beauty", "difficulty": "easy"},
            {"query": "nial polish remover", "expected": "nail polish remover", "category": "beauty", "difficulty": "medium"},
            {"query": "clay mask detox", "expected": "clay mask detox", "category": "beauty", "difficulty": "easy"},
        ])

        # Home & Kitchen (15 queries)
        dataset.extend([
            # Easy
            {"query": "coffee maker", "expected": "coffee maker", "category": "home_kitchen", "difficulty": "easy"},
            {"query": "cooking pot set", "expected": "cooking pot set", "category": "home_kitchen", "difficulty": "easy"},
            # Medium
            {"query": "blender smooothie", "expected": "blender smoothie", "category": "home_kitchen", "difficulty": "medium"},
            {"query": "mircrowave oven", "expected": "microwave oven", "category": "home_kitchen", "difficulty": "medium"},
            {"query": "dishwasher tablete", "expected": "dishwasher tablet", "category": "home_kitchen", "difficulty": "medium"},
            # Hard
            {"query": "refridgerator organzer", "expected": "refrigerator organizer", "category": "home_kitchen", "difficulty": "hard"},
            {"query": "vaccuum cleaner", "expected": "vacuum cleaner", "category": "home_kitchen", "difficulty": "hard"},
            # No change
            {"query": "IKEA dining table", "expected": "IKEA dining table", "category": "home_kitchen", "difficulty": "easy"},
            {"query": "5 liter water jug", "expected": "5 liter water jug", "category": "home_kitchen", "difficulty": "easy"},
            {"query": "qty 3 plates set", "expected": "qty 3 plates set", "category": "home_kitchen", "difficulty": "easy"},
            # Additional
            {"query": "steel cutlery", "expected": "steel cutlery", "category": "home_kitchen", "difficulty": "easy"},
            {"query": "non-stik cookware", "expected": "non-stick cookware", "category": "home_kitchen", "difficulty": "medium"},
            {"query": "ruber mat floor", "expected": "rubber mat floor", "category": "home_kitchen", "difficulty": "medium"},
            {"query": "cutting baord wood", "expected": "cutting board wood", "category": "home_kitchen", "difficulty": "medium"},
            {"query": "towel hanger rack", "expected": "towel hanger rack", "category": "home_kitchen", "difficulty": "easy"},
        ])

        # Sports (10 queries)
        dataset.extend([
            # Easy
            {"query": "yoga mat", "expected": "yoga mat", "category": "sports", "difficulty": "easy"},
            {"query": "dumbell set", "expected": "dumbbell set", "category": "sports", "difficulty": "easy"},
            # Medium
            {"query": "running shose", "expected": "running shoes", "category": "sports", "difficulty": "medium"},
            {"query": "basketbal hoop", "expected": "basketball hoop", "category": "sports", "difficulty": "medium"},
            # Hard
            {"query": "treadmil excersise", "expected": "treadmill exercise", "category": "sports", "difficulty": "hard"},
            {"query": "bicikle helmet", "expected": "bicycle helmet", "category": "sports", "difficulty": "hard"},
            # No change
            {"query": "5 kg kettlebell", "expected": "5 kg kettlebell", "category": "sports", "difficulty": "easy"},
            {"query": "$49.99 baseball glove", "expected": "$49.99 baseball glove", "category": "sports", "difficulty": "easy"},
            {"query": "nike running", "expected": "nike running", "category": "sports", "difficulty": "easy"},
            {"query": "swimming pool floatie", "expected": "swimming pool floatie", "category": "sports", "difficulty": "easy"},
        ])

        # Toys (10 queries)
        dataset.extend([
            # Easy
            {"query": "board game", "expected": "board game", "category": "toys", "difficulty": "easy"},
            {"query": "puzzle set", "expected": "puzzle set", "category": "toys", "difficulty": "easy"},
            # Medium
            {"query": "actin figures", "expected": "action figures", "category": "toys", "difficulty": "medium"},
            {"query": "building bricks", "expected": "building bricks", "category": "toys", "difficulty": "easy"},
            # Hard
            {"query": "constructio vehicle", "expected": "construction vehicle", "category": "toys", "difficulty": "hard"},
            {"query": "robotic toy learn", "expected": "robotic toy learn", "category": "toys", "difficulty": "easy"},
            # No change
            {"query": "LEGO minifigure", "expected": "LEGO minifigure", "category": "toys", "difficulty": "easy"},
            {"query": "qty 20 marbles", "expected": "qty 20 marbles", "category": "toys", "difficulty": "easy"},
            {"query": "$15.99 toy car", "expected": "$15.99 toy car", "category": "toys", "difficulty": "easy"},
            {"query": "stuffed animal plush", "expected": "stuffed animal plush", "category": "toys", "difficulty": "easy"},
        ])

        # Automotive (10 queries)
        dataset.extend([
            # Easy
            {"query": "car seat cover", "expected": "car seat cover", "category": "automotive", "difficulty": "easy"},
            {"query": "oil filter", "expected": "oil filter", "category": "automotive", "difficulty": "easy"},
            # Medium
            {"query": "brake pade", "expected": "brake pad", "category": "automotive", "difficulty": "medium"},
            {"query": "windshied wipper", "expected": "windshield wiper", "category": "automotive", "difficulty": "medium"},
            # Hard
            {"query": "exaust pipe sistem", "expected": "exhaust pipe system", "category": "automotive", "difficulty": "hard"},
            {"query": "transmision fluid", "expected": "transmission fluid", "category": "automotive", "difficulty": "hard"},
            # No change
            {"query": "5 liter motor oil", "expected": "5 liter motor oil", "category": "automotive", "difficulty": "easy"},
            {"query": "$89.99 battery", "expected": "$89.99 battery", "category": "automotive", "difficulty": "easy"},
            {"query": "qty 4 tire plugs", "expected": "qty 4 tire plugs", "category": "automotive", "difficulty": "easy"},
            {"query": "BOSCH spark plugs", "expected": "BOSCH spark plugs", "category": "automotive", "difficulty": "easy"},
        ])

        # Grocery (10 queries)
        dataset.extend([
            # Easy
            {"query": "olive oil", "expected": "olive oil", "category": "grocery", "difficulty": "easy"},
            {"query": "wheat bread", "expected": "wheat bread", "category": "grocery", "difficulty": "easy"},
            # Medium
            {"query": "almond buter", "expected": "almond butter", "category": "grocery", "difficulty": "medium"},
            {"query": "chocklate bar", "expected": "chocolate bar", "category": "grocery", "difficulty": "medium"},
            # Hard
            {"query": "spinach salad organc", "expected": "spinach salad organic", "category": "grocery", "difficulty": "hard"},
            {"query": "yogurt probotic", "expected": "yogurt probiotic", "category": "grocery", "difficulty": "hard"},
            # No change
            {"query": "5 kg rice bag", "expected": "5 kg rice bag", "category": "grocery", "difficulty": "easy"},
            {"query": "$12.99 coffee beans", "expected": "$12.99 coffee beans", "category": "grocery", "difficulty": "easy"},
            {"query": "qty 6 eggs carton", "expected": "qty 6 eggs carton", "category": "grocery", "difficulty": "easy"},
            {"query": "NESTLÉ chocolate", "expected": "NESTLÉ chocolate", "category": "grocery", "difficulty": "easy"},
        ])

        # Special cases: Abbreviations
        dataset.extend([
            {"query": "2 pcs hdmi cable", "expected": "2 pcs hdmi cable", "category": "special_abbreviations", "difficulty": "easy"},
            {"query": "qty 10 usb dongle", "expected": "qty 10 usb dongle", "category": "special_abbreviations", "difficulty": "easy"},
            {"query": "doz eggs carton", "expected": "doz eggs carton", "category": "special_abbreviations", "difficulty": "easy"},
            {"query": "gpu ram upgrade", "expected": "gpu ram upgrade", "category": "special_abbreviations", "difficulty": "easy"},
            {"query": "cpu cooler rgb", "expected": "cpu cooler rgb", "category": "special_abbreviations", "difficulty": "easy"},
        ])

        # Special cases: Units
        dataset.extend([
            {"query": "5 kg protein powder", "expected": "5 kg protein powder", "category": "special_units", "difficulty": "easy"},
            {"query": "32 gb flash drive", "expected": "32 gb flash drive", "category": "special_units", "difficulty": "easy"},
            {"query": "1000 ml juice bottle", "expected": "1000 ml juice bottle", "category": "special_units", "difficulty": "easy"},
            {"query": "500 mg vitamin c", "expected": "500 mg vitamin c", "category": "special_units", "difficulty": "easy"},
            {"query": "50 oz water pitcher", "expected": "50 oz water pitcher", "category": "special_units", "difficulty": "easy"},
        ])

        # Special cases: Currencies
        dataset.extend([
            {"query": "$99.99 headphones", "expected": "$99.99 headphones", "category": "special_currencies", "difficulty": "easy"},
            {"query": "€50 shoes", "expected": "€50 shoes", "category": "special_currencies", "difficulty": "easy"},
            {"query": "₺200 bag", "expected": "₺200 bag", "category": "special_currencies", "difficulty": "easy"},
            {"query": "usd 150 monitor", "expected": "usd 150 monitor", "category": "special_currencies", "difficulty": "easy"},
            {"query": "eur 75 jacket", "expected": "eur 75 jacket", "category": "special_currencies", "difficulty": "easy"},
        ])

        # Special cases: Brand names that look like typos
        dataset.extend([
            {"query": "ASUS laptop gaming", "expected": "ASUS laptop gaming", "category": "special_brands", "difficulty": "easy"},
            {"query": "OPPO smartphone 5g", "expected": "OPPO smartphone 5g", "category": "special_brands", "difficulty": "easy"},
            {"query": "POCO budget phone", "expected": "POCO budget phone", "category": "special_brands", "difficulty": "easy"},
            {"query": "IKEA furniture", "expected": "IKEA furniture", "category": "special_brands", "difficulty": "easy"},
            {"query": "SHEIN dress", "expected": "SHEIN dress", "category": "special_brands", "difficulty": "easy"},
        ])

        return dataset

    def get_by_category(self, category: str) -> List[Dict[str, str]]:
        """Get all queries for a specific category."""
        return [q for q in self.queries if q["category"] == category]

    def get_all_categories(self) -> List[str]:
        """Get unique category names."""
        return sorted(set(q["category"] for q in self.queries))


class WordSimilarity:
    """Word-level similarity metrics."""

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return WordSimilarity.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def similarity_score(s1: str, s2: str) -> float:
        """Calculate similarity score between 0 and 1.

        Uses normalized Levenshtein distance.
        """
        if s1 == s2:
            return 1.0

        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0

        distance = WordSimilarity.levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)


class BenchmarkRunner:
    """Main benchmark runner for correction models."""

    def __init__(self, models: List[str], verbose: bool = False, model_path: str = None):
        """Initialize benchmark runner.

        Args:
            models: List of model names to test (byt5, t5_large)
            verbose: Print detailed output
            model_path: Optional path to fine-tuned T5-large checkpoint
        """
        self.models_to_test = models
        self.verbose = verbose
        self.model_path = model_path
        self.results: Dict[str, List[BenchmarkResult]] = {model: [] for model in models}
        self.dataset = BenchmarkDataset()

    def run(self):
        """Run benchmark across all models and queries."""
        print(f"\nBenchmark Configuration:")
        print(f"- Models: {', '.join(self.models_to_test)}")
        print(f"- Total queries: {len(self.dataset.queries)}")
        print(f"- Categories: {', '.join(self.dataset.get_all_categories())}")
        print("-" * 80)

        for model_name in self.models_to_test:
            print(f"\nTesting model: {model_name}")
            self._test_model(model_name)

    def _test_model(self, model_name: str):
        """Test a single model across all queries.

        Args:
            model_name: Name of model to test
        """
        if model_name == "byt5":
            tester = ByteTester()
        elif model_name == "t5_large":
            tester = T5LargeTester(model_path=self.model_path)
        else:
            print(f"Unknown model: {model_name}")
            return

        if tester.model is None:
            print(f"Skipping {model_name} - model not loaded")
            return

        for i, test_case in enumerate(self.dataset.queries):
            query = test_case["query"]
            expected = test_case["expected"]
            category = test_case["category"]
            difficulty = test_case["difficulty"]

            start_time = time.time()
            model_output = tester.correct(query)
            latency_ms = (time.time() - start_time) * 1000

            exact_match = model_output.lower() == expected.lower()
            is_false_positive = (query.lower() == expected.lower()) and not exact_match
            word_sim = WordSimilarity.similarity_score(model_output, expected)

            result = BenchmarkResult(
                query=query,
                expected=expected,
                model_output=model_output,
                category=category,
                difficulty=difficulty,
                exact_match=exact_match,
                word_similarity=word_sim,
                latency_ms=latency_ms,
                is_false_positive=is_false_positive
            )

            self.results[model_name].append(result)

            if self.verbose:
                status = "✓" if exact_match else "✗"
                print(f"  {status} [{category}] {query} -> {model_output}")

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(self.dataset.queries)} queries")

    def compute_metrics(self) -> Dict[str, Dict]:
        """Compute detailed metrics per model and category.

        Returns:
            Nested dict: metrics[model_name][category] = metric_dict
        """
        metrics = {}

        for model_name in self.models_to_test:
            model_results = self.results[model_name]
            metrics[model_name] = {}

            # Group by category
            by_category = defaultdict(list)
            for result in model_results:
                by_category[result.category].append(result)

            # Compute per-category metrics
            for category, results in by_category.items():
                metrics[model_name][category] = self._compute_category_metrics(results)

            # Overall metrics
            metrics[model_name]["overall"] = self._compute_category_metrics(model_results)

        return metrics

    def _compute_category_metrics(self, results: List[BenchmarkResult]) -> Dict:
        """Compute metrics for a set of results.

        Args:
            results: List of benchmark results

        Returns:
            Dictionary with computed metrics
        """
        if not results:
            return {}

        exact_matches = sum(1 for r in results if r.exact_match)
        false_positives = sum(1 for r in results if r.is_false_positive)
        latencies = [r.latency_ms for r in results]
        similarities = [r.word_similarity for r in results]

        # Compute precision/recall
        tp = sum(1 for r in results if r.exact_match and not r.is_false_positive)
        fp = false_positives
        total_needed_correction = sum(1 for r in results if r.query != r.expected)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_needed_correction if total_needed_correction > 0 else 0

        return {
            "total_queries": len(results),
            "exact_match_accuracy": exact_matches / len(results) if results else 0,
            "false_positive_rate": false_positives / len(results) if results else 0,
            "avg_word_similarity": statistics.mean(similarities) if similarities else 0,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "median_latency_ms": statistics.median(latencies) if latencies else 0,
            "precision": precision,
            "recall": recall,
            "f1_score": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        }

    def print_report(self, metrics: Dict[str, Dict]):
        """Print formatted benchmark report.

        Args:
            metrics: Computed metrics dictionary
        """
        print("\n" + "=" * 100)
        print("BENCHMARK REPORT")
        print("=" * 100)

        for model_name in self.models_to_test:
            print(f"\n{'=' * 100}")
            print(f"Model: {model_name.upper()}")
            print(f"{'=' * 100}")

            model_metrics = metrics.get(model_name, {})

            # Print overall metrics
            if "overall" in model_metrics:
                overall = model_metrics["overall"]
                print(f"\nOVERALL METRICS:")
                print(f"  Total Queries: {overall.get('total_queries', 0)}")
                print(f"  Exact Match Accuracy: {overall.get('exact_match_accuracy', 0):.2%}")
                print(f"  False Positive Rate: {overall.get('false_positive_rate', 0):.2%}")
                print(f"  Average Word Similarity: {overall.get('avg_word_similarity', 0):.4f}")
                print(f"  Average Latency: {overall.get('avg_latency_ms', 0):.2f}ms")
                print(f"  Precision: {overall.get('precision', 0):.4f}")
                print(f"  Recall: {overall.get('recall', 0):.4f}")
                print(f"  F1 Score: {overall.get('f1_score', 0):.4f}")

            # Print per-category metrics
            print(f"\nPER-CATEGORY METRICS:")
            print(f"{'Category':<25} {'Accuracy':<12} {'FP Rate':<12} {'Latency':<12} {'F1 Score':<10}")
            print("-" * 70)

            for category in sorted(model_metrics.keys()):
                if category == "overall":
                    continue

                cat_metrics = model_metrics[category]
                accuracy = cat_metrics.get('exact_match_accuracy', 0)
                fp_rate = cat_metrics.get('false_positive_rate', 0)
                latency = cat_metrics.get('avg_latency_ms', 0)
                f1 = cat_metrics.get('f1_score', 0)

                print(f"{category:<25} {accuracy:<12.2%} {fp_rate:<12.2%} {latency:<12.2f}ms {f1:<10.4f}")

    def save_results(self, output_path: Path):
        """Save detailed results to JSON.

        Args:
            output_path: Path to save results
        """
        output_data = {}

        for model_name, results in self.results.items():
            output_data[model_name] = [asdict(r) for r in results]

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Universal benchmark for typo correction models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["byt5", "t5_large"],
        default=["t5_large"],
        help="Models to test"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save detailed results to JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file path for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-query output"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned T5-large checkpoint (overrides auto-search)"
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(models=args.models, verbose=args.verbose, model_path=args.model_path)
    runner.run()

    metrics = runner.compute_metrics()
    runner.print_report(metrics)

    if args.save:
        runner.save_results(Path(args.output))


if __name__ == "__main__":
    main()
