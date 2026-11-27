"""Rebuild Part 1 cleaned IMDB folds for Part 3 (same cleaning, dedup, folds, seeds).

Writes <out>/cross_validation_sets.pkl (list over seeds [0,3,5] of
{fold: {training, validation, ...}}) and <out>/test.pkl (list of (text, label)).
The test set is never used for tuning.
Run from repo root:  python part3/utils/build_folds.py --out part3/data
"""
import argparse
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from datasets import load_dataset
import part1.experiment as exp


def clean(text):
    return " ".join(re.sub(r"</?(br|i|em|hr)\s*/?>", " ", text).split())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="part3/data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 3, 5])
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    ds = load_dataset("mteb/imdb")
    tr, te = pd.DataFrame(ds["train"]), pd.DataFrame(ds["test"])
    tr["text"], te["text"] = tr["text"].map(clean), te["text"].map(clean)
    tr = tr.drop_duplicates(subset="text").reset_index(drop=True)  # as in part1/imdb.ipynb
    pairs = list(zip(tr["text"], tr["label"]))
    sets = [exp.cross_validation_split(pairs, k=a.folds, seed=s) for s in a.seeds]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pickle.dump(sets, open(out / "cross_validation_sets.pkl", "wb"))
    pickle.dump(list(zip(te["text"], te["label"])), open(out / "test.pkl", "wb"))
    print(f"train {len(tr)} (deduped), test {len(te)}, seeds {a.seeds}")
