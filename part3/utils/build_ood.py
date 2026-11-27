"""Build the OOD set: Yelp polarity (restaurant/business reviews), binary, model-free.

Labels: 0 = negative (1-2 stars), 1 = positive (4-5 stars); 3 stars are already dropped
by yelp_polarity. Uses the test split only (never seen by any system), cleaned with the
same function as IMDB, and a class-balanced fixed-seed subsample so every system is
scored on the same reviews. Writes a list of (text, label) to <out>/yelp.pkl, which
generate.py reads directly via --input_path.

Run from repo root:  python part3/utils/build_ood.py --n 2000
"""
import argparse
import pickle
import random
import re
from pathlib import Path

from datasets import load_dataset


def clean(text):
    # Same cleaning as IMDB; Yelp also has literal "\n" escapes
    text = text.replace("\\n", " ")
    return " ".join(re.sub(r"</?(br|i|em|hr)\s*/?>", " ", text).split())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="part3/data/robustness")
    ap.add_argument("--n", type=int, default=2000, help="total reviews, split evenly by class")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ds = load_dataset("fancyzhx/yelp_polarity", split="test")
    rng = random.Random(a.seed)
    out = []
    for label in (0, 1):
        idx = [i for i, l in enumerate(ds["label"]) if l == label]
        out += [(clean(ds[i]["text"]), label) for i in rng.sample(idx, a.n // 2)]
    rng.shuffle(out)

    path = Path(a.out)
    path.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(path / "yelp.pkl", "wb"))
    lens = sorted(len(t.split()) for t, _ in out)
    print(f"{len(out)} reviews, {sum(l for _, l in out)} positive, "
          f"median {lens[len(lens) // 2]} words, max {lens[-1]}")
