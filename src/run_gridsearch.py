# src/run_gridsearch.py
import csv
from .utils import read_corpus, add_sentence_tokens
from .ngram_model import NGramModel

def replace_unk(token_lists, vocab):
    return [[tok if tok in vocab else '<UNK>' for tok in toks] for toks in token_lists]

def evaluate_model(train_file, valid_file, n, smoothing, k=1.0, lambda_=0.5, minfreq=2):
    # Load corpus
    train_lines = read_corpus(train_file)
    train_lines = add_sentence_tokens(train_lines)
    valid_lines = read_corpus(valid_file)
    valid_lines = add_sentence_tokens(valid_lines)

    # Train model
    model = NGramModel(n=n, smoothing=smoothing, k=k,
                       interp_lambda=lambda_, min_freq=minfreq)
    model.fit(train_lines)

    # Replace unknowns in validation
    valid_proc = replace_unk(valid_lines, model.vocab)

    # Compute perplexity
    pp = model.perplexity(valid_proc)
    return pp, len(model.vocab)

def main():
    train_file = "data/train.txt"
    valid_file = "data/val.txt"   # <-- adjust name if yours is val.txt

    results = []

    # Experiments
    experiments = [
        # Unigram
        {"n": 1, "smoothing": "none"},
        {"n": 1, "smoothing": "addk", "k": 1.0},
        # Bigram unsmoothed
        {"n": 2, "smoothing": "none"},
        # Bigram Add-k with different k
        {"n": 2, "smoothing": "addk", "k": 0.1},
        {"n": 2, "smoothing": "addk", "k": 0.5},
        {"n": 2, "smoothing": "addk", "k": 1.0},
        # Interpolation with different λ
        {"n": 2, "smoothing": "interp", "k": 0.1, "lambda_": 0.3},
        {"n": 2, "smoothing": "interp", "k": 0.1, "lambda_": 0.5},
        {"n": 2, "smoothing": "interp", "k": 0.1, "lambda_": 0.7},
    ]

    for exp in experiments:
        n = exp.get("n", 2)
        smoothing = exp.get("smoothing", "addk")
        k = exp.get("k", 1.0)
        lambda_ = exp.get("lambda_", 0.5)
        minfreq = exp.get("minfreq", 2)

        pp, vocab_size = evaluate_model(train_file, valid_file, n, smoothing, k, lambda_, minfreq)

        results.append({
            "n": n,
            "smoothing": smoothing,
            "k": k,
            "lambda": lambda_,
            "minfreq": minfreq,
            "vocab_size": vocab_size,
            "perplexity": pp
        })

        print(f"Done: n={n}, smoothing={smoothing}, k={k}, lambda={lambda_}, minfreq={minfreq} → PP={pp}")

    # Save to CSV
    with open("results/experiment_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to results/experiment_results.csv")

if __name__ == "__main__":
    main()
