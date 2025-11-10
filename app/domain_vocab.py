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
            for line in f:
                word = line.strip().lower()
                if word:
                    vocab.add(word)
    except FileNotFoundError:
        print(f"Warning: {vocab_path} not found. Using empty vocab.")
    return vocab