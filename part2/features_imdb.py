"""
Project item 6a (A): input features of IMDB reviews for error analysis (no model needed).

One row per review. Two groups:
  * the tutorial's baseline set (D6, via textstat): lexicon_count, flesch_reading_ease, reading_time,
    flesch_kincaid_grade
  * our own candidates (the final list is decided after the first error analysis, D11):
      n_tokens, unk_rate, negations, contrast_words, exclamations, questions, ratings, upper_ratio,
      shouted_words, pos_words, neg_words, polarity_mix

Model features (predicted label, confidence) are Park's part (6b) and are joined on later.

Usage as a library:   from part2.features_imdb import compute_features;  df = compute_features(texts, known_words)
Usage as a script:    python part2/features_imdb.py --split test --out part2/results/imdb/features_test.csv
                        [--words part2/results/imdb/embeddings/glove840b_imdb_words.txt] [--sample 200]
`known_words` is the set of words that have a pretrained vector (the pruned word list from coverage_imdb.py);
without it, `unk_rate` is left out.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

NEGATIONS = {"not", "no", "nor", "never", "n't", "cannot", "neither", "nothing", "nobody", "nowhere"}
CONTRAST = {"but", "however", "although", "though", "yet", "despite", "nevertheless"}
RATING = re.compile(r"\b\d+(?:\.\d+)?\s*/\s*(?:10|5|100)\b|\b\d+(?:\.\d+)?\s+(?:out of|stars?)\b", re.IGNORECASE)


def clean(text):
    """Same cleaning as Part 1: drop HTML tags, normalise whitespace."""
    return " ".join(re.sub(r"</?(br|i|em|hr)\s*/?>", " ", text).split())


def compute_features(texts, known_words=None):
    """texts: list of cleaned review strings. known_words: optional set of lowercased words with a vector."""
    import nltk
    import textstat
    from nltk.corpus import opinion_lexicon
    from part2.clslib import BasicTokenizer

    try:
        nltk.data.find("corpora/opinion_lexicon")
    except LookupError:
        nltk.download("opinion_lexicon")
    pos = set(opinion_lexicon.positive())
    neg = set(opinion_lexicon.negative())
    tokenizer = BasicTokenizer(lowercase=True, remove_punct=False, remove_stopwords=False)

    rows = []
    for text in texts:
        toks = tokenizer.tokens(text)
        n = max(len(toks), 1)
        letters = [c for c in text if c.isalpha()]
        n_pos = sum(t in pos for t in toks)
        n_neg = sum(t in neg for t in toks)
        row = {
            # tutorial baseline (D6)
            "lexicon_count": textstat.lexicon_count(text),
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "reading_time": textstat.reading_time(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            # our candidates
            "n_tokens": len(toks),
            "negations": sum(t in NEGATIONS for t in toks),
            "contrast_words": sum(t in CONTRAST for t in toks),
            "exclamations": text.count("!"),
            "questions": text.count("?"),
            "ratings": len(RATING.findall(text)),
            "upper_ratio": sum(c.isupper() for c in letters) / max(len(letters), 1),
            "shouted_words": sum(w.isalpha() and w.isupper() and len(w) >= 3 for w in text.split()),
            "pos_words": n_pos,
            "neg_words": n_neg,
            "polarity_mix": min(n_pos, n_neg),  # both polarities present: mixed reviews are expected to be harder
        }
        if known_words is not None:
            row["unk_rate"] = sum(t not in known_words for t in toks) / n
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--out", default=str(here / "results" / "imdb" / "features_test.csv"))
    parser.add_argument("--words", default=None, help="glove840b_imdb_words.txt from coverage_imdb.py (for unk_rate)")
    parser.add_argument("--sample", type=int, default=0, help="first N reviews only (quick test)")
    args = parser.parse_args()

    from datasets import load_dataset

    df = pd.DataFrame(load_dataset("mteb/imdb")[args.split])
    df["text"] = df["text"].map(clean)
    if args.sample:
        df = df.head(args.sample)
    known = None
    if args.words and Path(args.words).exists():
        known = set(Path(args.words).read_text(encoding="utf-8").split("\n")) - {""}
    feats = compute_features(df["text"].tolist(), known)
    feats.insert(0, "label", df["label"].values)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(args.out, index_label="review_id")
    pd.set_option("display.width", 200)
    print(feats.describe().T[["mean", "std", "min", "max"]].round(3))
    print(f"\nSaved {len(feats)} rows to {args.out}")
