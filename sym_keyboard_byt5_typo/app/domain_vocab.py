# Domain vocabulary utilities

def load_domain_vocab(vocab_path: str) -> set[str]:
    """
    Load domain-specific vocabulary from a text file.
    Each line is a word.
    Returns a set of words.
    """
    vocab = set()
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                word = line.strip().lower()
                if word:
                    vocab.add(word)
                elif line.strip():  # Empty word but line has content
                    print(f"Warning: Empty word on line {line_num} in {vocab_path}")
    except FileNotFoundError:
        print(f"Warning: {vocab_path} not found. Using empty vocab.")
    except UnicodeDecodeError as e:
        print(f"Error: Encoding issue in {vocab_path}: {e}. Using empty vocab.")
    except Exception as e:
        print(f"Error loading vocab from {vocab_path}: {e}. Using empty vocab.")
    return vocab