#!/usr/bin/env python3
"""
Real Spelling Correction Dataset Downloader and Processor

Downloads multiple real-world spelling correction datasets and processes them into a unified format
for ByT5-Large finetuning. Replaces synthetic training data with real-world misspellings.

Datasets included:
- Wikipedia Common Misspellings (~4500 pairs)
- Birkbeck Corpus (36K pairs)
- Peter Norvig's Spell Testsets (~1K pairs)
- GitHub Typo Corpus (optional, large)
- Amazon ESCI Queries (identity pairs)
- Aspell vocabulary for validation

Output: real_data/combined_real_pairs.jsonl with deduplicated, filtered pairs
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Handles downloading and processing spelling correction datasets."""

    def __init__(self, output_dir: str = "real_data", max_retries: int = 3):
        """
        Initialize the downloader.

        Args:
            output_dir: Directory to store downloaded datasets
            max_retries: Number of retries for network requests
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_retries = max_retries
        self.pairs: Dict[Tuple[str, str], str] = {}  # (misspelled, correct) -> source
        self.stats = defaultdict(int)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _fetch_url(self, url: str, timeout: int = 15) -> str:
        """
        Fetch URL content with retries.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            Response content as string

        Raises:
            requests.RequestException: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Fetching {url} (attempt {attempt + 1}/{self.max_retries})")
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")

    def _add_pair(self, misspelled: str, correct: str, source: str) -> bool:
        """
        Add a correction pair with validation.

        Args:
            misspelled: Misspelled word
            correct: Correct word
            source: Source dataset name

        Returns:
            True if pair was added, False if rejected
        """
        # Normalize and validate
        misspelled = misspelled.strip().lower()
        correct = correct.strip().lower()

        # Skip invalid pairs
        if not misspelled or not correct:
            return False
        if len(misspelled) > 50 or len(correct) > 50:
            return False
        # Allow alphanumeric, spaces, hyphens, dots, and common e-commerce chars
        # (old isalpha() filter was rejecting everything with spaces/numbers)
        allowed_pattern = re.compile(r'^[a-z0-9\s\-\.\/\'\&\+]+$')
        if not allowed_pattern.match(misspelled) or not allowed_pattern.match(correct):
            return False

        # Check edit distance (max 5)
        if self._edit_distance(misspelled, correct) > 5:
            return False

        # Add pair (keep first source if duplicate)
        pair_key = (misspelled, correct)
        if pair_key not in self.pairs:
            self.pairs[pair_key] = source
            self.stats[source] += 1
            return True
        return False

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return DatasetDownloader._edit_distance(s2, s1)

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

    def download_wikipedia_misspellings(self) -> int:
        """
        Download Wikipedia common misspellings list.
        Uses the MediaWiki API to get raw wikitext, then parses "misspelling->correct" format.
        Also parses individual letter pages (A-Z) as fallback.
        """
        logger.info("Downloading Wikipedia common misspellings...")
        added = 0

        try:
            # Method 1: Use MediaWiki API to get raw wikitext of /For_machines page
            api_url = (
                "https://en.wikipedia.org/w/index.php?title="
                "Wikipedia:Lists_of_common_misspellings/For_machines&action=raw"
            )
            try:
                content = self._fetch_url(api_url)
                for line in content.split('\n'):
                    line = line.strip()
                    if not line or '->' not in line:
                        continue
                    try:
                        parts = line.split('->')
                        if len(parts) == 2:
                            misspelled = parts[0].strip()
                            # correct side can have multiple options separated by comma
                            correct_options = parts[1].strip().split(',')
                            correct = correct_options[0].strip()
                            if self._add_pair(misspelled, correct, "wikipedia"):
                                added += 1
                    except Exception as e:
                        logger.debug(f"Error parsing line '{line}': {e}")
                logger.info(f"Wikipedia /For_machines: Added {added} pairs")
            except Exception as e:
                logger.warning(f"Failed to fetch /For_machines: {e}")

            # Method 2: Also fetch individual letter pages for more coverage
            if added < 100:
                logger.info("Fetching individual Wikipedia misspelling pages (A-Z)...")
                for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0":
                    page_name = letter if letter != "0" else "0-9"
                    api_url = (
                        f"https://en.wikipedia.org/w/index.php?title="
                        f"Wikipedia:Lists_of_common_misspellings/{page_name}&action=raw"
                    )
                    try:
                        content = self._fetch_url(api_url, timeout=10)
                        # Format on these pages: {{misspelling|correct}}
                        # or lines like: * misspelling (correct)
                        # Actually the common format is plain wikitext lines
                        for line in content.split('\n'):
                            line = line.strip()
                            # Pattern 1: "misspelling (correct)" or "misspelling (correct1, correct2)"
                            match = re.match(r'^\*?\s*(\w+)\s*\(([^)]+)\)', line)
                            if match:
                                misspelled = match.group(1).strip()
                                correct = match.group(2).split(',')[0].strip()
                                if self._add_pair(misspelled, correct, "wikipedia"):
                                    added += 1
                    except Exception:
                        continue

            logger.info(f"Wikipedia: Added {added} pairs total")
            return added

        except Exception as e:
            logger.error(f"Failed to download Wikipedia misspellings: {e}")
            return 0

    def download_birkbeck_corpus(self) -> int:
        """
        Download Birkbeck Corpus (Roger Mitton).
        Format: correct word starts with $, followed by misspellings one per line.
        """
        logger.info("Downloading Birkbeck Corpus...")
        added = 0

        try:
            url = "https://www.dcs.bbk.ac.uk/~roger/missp.dat"
            content = self._fetch_url(url)

            current_correct = None
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue

                if line.startswith('$'):
                    current_correct = line[1:].strip()
                else:
                    if current_correct:
                        if self._add_pair(line, current_correct, "birkbeck"):
                            added += 1

            logger.info(f"Birkbeck: Added {added} pairs")
            return added

        except Exception as e:
            logger.error(f"Failed to download Birkbeck Corpus: {e}")
            return 0

    def download_norvig_testsets(self) -> int:
        """
        Download Peter Norvig's spell testsets.

        Format varies between files:
        - spell-testset1.txt: "correct: wrong1 wrong2" (space-separated)
        - spell-testset2.txt: "correct: wrong1, wrong2" (comma-separated or space)

        Also downloads Norvig's additional Birkbeck-derived data.
        """
        logger.info("Downloading Peter Norvig's spell testsets...")
        added = 0

        testset_urls = [
            "https://norvig.com/spell-testset1.txt",
            "https://norvig.com/spell-testset2.txt",
        ]

        for url in testset_urls:
            try:
                logger.debug(f"Fetching {url}")
                content = self._fetch_url(url)

                for line in content.split('\n'):
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    try:
                        # Split on first colon only
                        colon_idx = line.index(':')
                        correct = line[:colon_idx].strip()
                        misspellings_raw = line[colon_idx + 1:].strip()

                        # Handle both comma-separated and space-separated
                        if ',' in misspellings_raw:
                            misspellings = [m.strip() for m in misspellings_raw.split(',')]
                        else:
                            misspellings = misspellings_raw.split()

                        for misspelled in misspellings:
                            misspelled = misspelled.strip()
                            if misspelled and self._add_pair(misspelled, correct, "norvig"):
                                added += 1
                    except Exception as e:
                        logger.debug(f"Error parsing line '{line}': {e}")

                logger.info(f"Norvig {url}: running total = {added}")

            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")

        # Also try Norvig's Wikipedia-derived list
        wiki_url = "https://norvig.com/ngrams/spell-errors.txt"
        try:
            content = self._fetch_url(wiki_url)
            for line in content.split('\n'):
                line = line.strip()
                if not line or ':' not in line:
                    continue
                try:
                    colon_idx = line.index(':')
                    correct = line[:colon_idx].strip()
                    misspellings_raw = line[colon_idx + 1:].strip()
                    if ',' in misspellings_raw:
                        misspellings = [m.strip() for m in misspellings_raw.split(',')]
                    else:
                        misspellings = misspellings_raw.split()
                    for misspelled in misspellings:
                        misspelled = misspelled.strip()
                        if misspelled and self._add_pair(misspelled, correct, "norvig"):
                            added += 1
                except Exception:
                    continue
            logger.info(f"Norvig spell-errors.txt: running total = {added}")
        except Exception as e:
            logger.debug(f"Norvig spell-errors.txt not available: {e}")

        logger.info(f"Norvig: Added {added} pairs total")
        return added

    def download_github_typo_corpus(self) -> int:
        """
        Download GitHub Typo Corpus (if not too large).
        Note: This is optional and may be skipped if too large.
        """
        logger.info("Attempting GitHub Typo Corpus download...")
        added = 0

        try:
            # Check if repo data is available
            url = "https://raw.githubusercontent.com/mhagiwara/github-typo-corpus/master/splits/data.csv"
            logger.warning("GitHub Typo Corpus is typically large. If you need it, please download manually from:")
            logger.warning("https://github.com/mhagiwara/github-typo-corpus")
            logger.warning("Skipping automatic download to avoid bandwidth issues.")
            return 0

        except Exception as e:
            logger.warning(f"GitHub Typo Corpus unavailable: {e}")
            return 0

    def download_esci_identity_pairs(self) -> int:
        """
        Download Amazon ESCI queries from HuggingFace.
        Use unique English queries as identity pairs (correct->correct).

        The ESCI dataset has ~2.6M rows but only ~130K unique queries.
        We extract just the query column and deduplicate to avoid
        iterating over millions of duplicate rows.
        """
        logger.info("Downloading Amazon ESCI queries for identity pairs...")
        added = 0

        try:
            try:
                from datasets import load_dataset
                logger.info("Loading ESCI dataset from HuggingFace (query column only)...")

                # Load only the 'query' column to save memory and time
                dataset = load_dataset(
                    "tasksource/esci",
                    split="train",
                    columns=["query"],          # only fetch query column
                )

                # Extract unique queries efficiently using pandas
                logger.info("Extracting unique queries...")
                try:
                    import pandas as pd
                    df = dataset.to_pandas()
                    unique_queries = df["query"].dropna().str.strip().str.lower().unique()
                    logger.info(f"Found {len(unique_queries)} unique queries from {len(df)} total rows")
                except Exception:
                    # Fallback: iterate manually but with early dedup
                    logger.info("Pandas fallback: iterating with dedup...")
                    seen = set()
                    unique_queries = []
                    for sample in dataset:
                        q = sample.get("query", "").strip().lower()
                        if q and q not in seen:
                            seen.add(q)
                            unique_queries.append(q)

                # Add as identity pairs
                for query in tqdm(unique_queries, desc="Adding ESCI identity pairs"):
                    if isinstance(query, str) and 2 < len(query) < 50:
                        if self._add_pair(query, query, "esci_identity"):
                            added += 1

                logger.info(f"ESCI: Added {added} identity pairs from {len(unique_queries)} unique queries")

            except ImportError:
                logger.warning("HuggingFace datasets library not installed.")
                logger.warning("Install with: pip install datasets")
                logger.warning("Skipping ESCI dataset.")

            return added

        except Exception as e:
            logger.warning(f"Failed to download ESCI dataset: {e}")
            return 0

    def download_aspell_vocabulary(self) -> Set[str]:
        """
        Get Aspell vocabulary for validation.
        Attempts to use system aspell, falls back to online list.
        """
        logger.info("Loading Aspell vocabulary for validation...")
        vocabulary = set()

        try:
            import subprocess
            result = subprocess.run(
                ["aspell", "dump", "master"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                vocabulary = set(word.strip().lower() for word in result.stdout.split('\n') if word.strip())
                logger.info(f"Loaded {len(vocabulary)} words from system Aspell")
                return vocabulary
        except Exception as e:
            logger.debug(f"Could not load system Aspell: {e}")

        # Fallback: download common words list
        try:
            logger.info("Downloading fallback word list...")
            url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
            content = self._fetch_url(url, timeout=30)
            vocabulary = set(word.strip().lower() for word in content.split('\n') if word.strip())
            logger.info(f"Loaded {len(vocabulary)} words from dwyl English words")
        except Exception as e:
            logger.warning(f"Could not load vocabulary: {e}")

        return vocabulary

    def save_combined_dataset(self, vocabulary: Set[str] = None) -> Path:
        """
        Save combined and deduplicated dataset to JSONL.

        Args:
            vocabulary: Optional set of valid words for filtering

        Returns:
            Path to output file
        """
        logger.info(f"Saving combined dataset with {len(self.pairs)} pairs...")

        output_file = self.output_dir / "combined_real_pairs.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for (misspelled, correct), source in tqdm(self.pairs.items(), desc="Writing JSONL"):
                record = {
                    "misspelled": misspelled,
                    "correct": correct,
                    "source": source
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        logger.info(f"Saved to {output_file}")
        return output_file

    def print_statistics(self):
        """Print dataset statistics."""
        logger.info("\n" + "="*60)
        logger.info("DATASET STATISTICS")
        logger.info("="*60)

        total_pairs = len(self.pairs)
        logger.info(f"Total unique pairs: {total_pairs}")

        logger.info("\nPairs by source:")
        for source in sorted(self.stats.keys()):
            count = self.stats[source]
            percentage = (count / total_pairs * 100) if total_pairs > 0 else 0
            logger.info(f"  {source:20s}: {count:6d} ({percentage:5.1f}%)")

        # Count identity vs correction pairs
        identity_pairs = sum(1 for (m, c) in self.pairs if m == c)
        correction_pairs = total_pairs - identity_pairs

        logger.info(f"\nPair types:")
        logger.info(f"  Identity pairs (a->a):     {identity_pairs:6d} ({identity_pairs/total_pairs*100:5.1f}%)")
        logger.info(f"  Correction pairs (a->b):   {correction_pairs:6d} ({correction_pairs/total_pairs*100:5.1f}%)")

        logger.info("="*60 + "\n")

    def run(self, skip_esci: bool = False, skip_github: bool = True):
        """
        Run the complete download and processing pipeline.

        Args:
            skip_esci: Skip ESCI dataset download
            skip_github: Skip GitHub Typo Corpus (default True due to size)
        """
        logger.info("Starting real dataset download and processing...")
        logger.info(f"Output directory: {self.output_dir.absolute()}")

        # Download datasets
        self.download_wikipedia_misspellings()
        self.download_birkbeck_corpus()
        self.download_norvig_testsets()

        if not skip_github:
            self.download_github_typo_corpus()

        if not skip_esci:
            self.download_esci_identity_pairs()

        # Load vocabulary for reference
        vocabulary = self.download_aspell_vocabulary()

        # Save combined dataset
        self.save_combined_dataset(vocabulary)

        # Print statistics
        self.print_statistics()

        logger.info("Download and processing complete!")
        return self.output_dir / "combined_real_pairs.jsonl"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and process real spelling correction datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_real_datasets.py
  python download_real_datasets.py --output-dir datasets
  python download_real_datasets.py --skip-esci
  python download_real_datasets.py --include-github
        """
    )

    parser.add_argument(
        "--output-dir",
        default="real_data",
        help="Output directory for datasets (default: real_data)"
    )

    parser.add_argument(
        "--skip-esci",
        action="store_true",
        help="Skip downloading ESCI identity pairs (faster, needs HuggingFace datasets)"
    )

    parser.add_argument(
        "--include-github",
        action="store_true",
        help="Include GitHub Typo Corpus (warning: large download)"
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for failed downloads (default: 3)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        downloader = DatasetDownloader(
            output_dir=args.output_dir,
            max_retries=args.max_retries
        )

        downloader.run(
            skip_esci=args.skip_esci,
            skip_github=not args.include_github
        )

        logger.info("Success!")
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
