# src/utils.py
from collections import Counter
from typing import List, Tuple

def read_corpus(path: str, lowercase: bool = True) -> List[List[str]]:
    """Return list of token lists (one per line)."""
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if lowercase:
                toks = [t.lower() for t in toks]
            lines.append(toks)
    return lines

def add_sentence_tokens(token_lists: List[List[str]],
                        start_token: str = '<s>',
                        end_token: str = '</s>') -> List[List[str]]:
    return [[start_token] + toks + [end_token] for toks in token_lists]

def build_vocab(token_lists: List[List[str]], min_freq: int = 2, unk_token: str = '<UNK>'):
    """Return vocab set and full frequency counter."""
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    vocab = {tok for tok, c in counter.items() if c >= min_freq}
    vocab.add(unk_token)
    return vocab, counter
