# src/run_experiment.py
import argparse
from .utils import read_corpus, add_sentence_tokens
from .ngram_model import NGramModel

def replace_unk(token_lists, vocab):
    return [[tok if tok in vocab else '<UNK>' for tok in toks] for toks in token_lists]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--valid', required=True)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--minfreq', type=int, default=2)
    parser.add_argument('--smoothing', choices=['none','addk','interp'], default='addk')
    parser.add_argument('--k', type=float, default=1.0)
    parser.add_argument('--lambda_', type=float, default=0.5)
    args = parser.parse_args()

    train_lines = read_corpus(args.train)
    train_lines = add_sentence_tokens(train_lines)
    valid_lines = read_corpus(args.valid)
    valid_lines = add_sentence_tokens(valid_lines)

    model = NGramModel(n=args.n, smoothing=args.smoothing, k=args.k, interp_lambda=args.lambda_, min_freq=args.minfreq)
    model.fit(train_lines)

    valid_proc = replace_unk(valid_lines, model.vocab)
    pp = model.perplexity(valid_proc)

    print(f"Model settings: n={args.n}, smoothing={args.smoothing}, k={args.k}, lambda={args.lambda_}, minfreq={args.minfreq}")
    print(f"Vocab size (train): {len(model.vocab)}")
    print(f"Perplexity on validation: {pp:.4f}" if pp != float('inf') else "Perplexity: INF (zero-prob encountered)")

if __name__ == '__main__':
    main()
