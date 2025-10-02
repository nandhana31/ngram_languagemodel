# ngram_languagemodel

This project implements unigram and bigram language models with support for add-k smoothing and interpolation. The models are trained on hotel review text data and evaluated using perplexity on both training and validation sets.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt

Run a single command - python -m src.run_experiment --train data/train.txt --valid data/val.txt --n 2 --minfreq 2 --smoothing addk --k 0.5
Run grid search - python -m src.run_gridsearch

