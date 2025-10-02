# n-gram Language Model

This project implements unigram and bigram language models for natural language processing tasks. The models are trained on hotel review text data and evaluated using perplexity on both training and validation sets. It includes support for **add-k smoothing** and **interpolation** to handle unseen words and sequences.

---

## Usage

### Run a Single Experiment

To train an n-gram model with specific parameters, use the following command. This example trains a bigram ($n=2$) model with a minimum frequency of 2, using add-k smoothing with a value of $k=0.5$.

```bash
python -m src.run_experiment --train data/train.txt --valid data/val.txt --n 2 --minfreq 2 --smoothing addk --k 0.5
```
### Run Grid Search

```bash
python -m src.run_gridsearch
```
