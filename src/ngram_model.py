# src/ngram_model.py
from collections import Counter
import math
from typing import List
from .utils import build_vocab

class NGramModel:
    def __init__(self, n=2, smoothing='none', k=1.0, interp_lambda=0.5, min_freq=2):
        assert n in (1,2)
        self.n = n
        self.smoothing = smoothing  # 'none' | 'addk' | 'interp'
        self.k = float(k)
        self.lambda_ = float(interp_lambda)
        self.min_freq = int(min_freq)

        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.vocab = set()
        self.total_unigrams = 0

    def fit(self, token_lists: List[List[str]]):
        # Build vocab with frequency cutoff (train only)
        vocab, counter = build_vocab(token_lists, min_freq=self.min_freq, unk_token='<UNK>')
        self.vocab = vocab

        # Replace rare tokens with <UNK> when counting
        processed = []
        for toks in token_lists:
            newt = [tok if tok in vocab else '<UNK>' for tok in toks]
            processed.append(newt)

        # Count unigrams and bigrams
        for toks in processed:
            self.unigram_counts.update(toks)
            self.total_unigrams += len(toks)
            if self.n == 2:
                for i in range(1, len(toks)):
                    self.bigram_counts[(toks[i-1], toks[i])] += 1

        # ensure special tokens are present
        self.vocab.update({'<UNK>', '<s>', '</s>'})

    def prob_unigram(self, w: str) -> float:
        if self.smoothing in ('addk', 'interp'):
            V = len(self.vocab)
            c = self.unigram_counts.get(w, 0)
            return (c + self.k) / (self.total_unigrams + self.k * V)
        else:
            c = self.unigram_counts.get(w, 0)
            return c / self.total_unigrams if self.total_unigrams > 0 else 0.0

    def prob_bigram(self, h: str, w: str) -> float:
        if self.smoothing == 'addk':
            V = len(self.vocab)
            c_hw = self.bigram_counts.get((h, w), 0)
            c_h = self.unigram_counts.get(h, 0)
            denom = c_h + self.k * V
            return (c_hw + self.k) / denom if denom > 0 else 0.0
        elif self.smoothing == 'interp':
            V = len(self.vocab)
            c_hw = self.bigram_counts.get((h, w), 0)
            c_h = self.unigram_counts.get(h, 0)
            bigram_part = (c_hw + self.k) / (c_h + self.k * V) if (c_h + self.k * V) > 0 else 0.0
            unigram_part = self.prob_unigram(w)
            return self.lambda_ * bigram_part + (1.0 - self.lambda_) * unigram_part
        else:  # no smoothing
            c_hw = self.bigram_counts.get((h, w), 0)
            c_h = self.unigram_counts.get(h, 0)
            return c_hw / c_h if c_h > 0 else 0.0

    def sentence_neg_logprob(self, toks: List[str]) -> float:
        """Return sum of -log(p) for tokens in sentence; returns inf if 0 prob occurs."""
        total = 0.0
        for i in range(1, len(toks)):
            if self.n == 1:
                p = self.prob_unigram(toks[i])
            else:
                p = self.prob_bigram(toks[i-1], toks[i])
            if p <= 0.0:
                return float('inf')
            total += -math.log(p)
        return total

    def perplexity(self, token_lists: List[List[str]]) -> float:
        N = 0
        total_neg_log = 0.0
        for toks in token_lists:
            N += (len(toks) - 1)   # exclude the initial <s> as target
            val = self.sentence_neg_logprob(toks)
            if val == float('inf'):
                return float('inf')
            total_neg_log += val
        if N == 0:
            return float('inf')
        avg_neg_log = total_neg_log / N
        return math.exp(avg_neg_log)
