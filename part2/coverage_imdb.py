"""
Project item 1 (A): does GloVe Common Crawl 840B/300d fit IMDB?

For several tokenizer settings (lowercase, remove_punct, remove_stopwords) it measures, on cleaned and
de-duplicated IMDB train text, how much of the text has a pretrained vector:
  * % of tokens and % of types (distinct words) found in the GloVe vocabulary, both with a lowercased lookup
    and "effective" (the lookup the training pipeline really does: no lowercasing of the token)
  * the most frequent out-of-vocabulary types
  * how many negation tokens survive each tokenizer (not, no, nor, never, n't)
and it probes how GloVe treats punctuation and contractions (tokenizer coherence).

It also does D2 (pruning): in the same pass over the GloVe file it saves vectors only for words that occur
in IMDB (train + test, all settings), plus the mean over the FULL vocabulary, which `PretrainedEmbeddings`
uses as the <unk> vector.

Designed for Snellius (CPU job, see job_coverage.sh). Quick test: add `--sample 500`.
Outputs go to part2/results/imdb/embeddings/.
"""
import argparse
import itertools
import json
import multiprocessing as mp
import re
import sys
import time
from collections import Counter
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from datasets import load_dataset  # noqa: E402
from part2.clslib import BasicTokenizer  # noqa: E402

NEGATIONS = {"not", "no", "nor", "never", "n't"}
PROBES = ["n't", "'s", "'m", "not", "!", ",", "...", "-", "(", ":)", "<br", "/>"]
# (lowercase, remove_punct, remove_stopwords)
CONFIGS = list(itertools.product([True, False], repeat=3))

parser = argparse.ArgumentParser()
parser.add_argument("--glove", required=True, help="glove.840B.300d.zip (or .txt)")
parser.add_argument("--dims", type=int, default=300)
parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "imdb" / "embeddings"))
parser.add_argument("--sample", type=int, default=0, help="use only N reviews per split (quick test)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def clean(text):
    """Same cleaning as Part 1: drop HTML tags, normalise whitespace."""
    return " ".join(re.sub(r"</?(br|i|em|hr)\s*/?>", " ", text).split())


# ---------------------------------------------------------------- data
print("Loading IMDB ...", flush=True)
ds = load_dataset("mteb/imdb")
train = pd.DataFrame(ds["train"])
test = pd.DataFrame(ds["test"])
train["text"] = train["text"].map(clean)
test["text"] = test["text"].map(clean)
train = train.drop_duplicates(subset="text").reset_index(drop=True)
if args.sample:
    train = train.sample(min(args.sample, len(train)), random_state=args.seed)
    test = test.sample(min(args.sample, len(test)), random_state=args.seed)
print(f"train reviews: {len(train)}, test reviews: {len(test)}", flush=True)

TEXTS = {"train": train["text"].tolist(), "test": test["text"].tolist()}


# ---------------------------------------------------------------- tokenize (parallel over settings)
def count_tokens(task):
    split, (lc, rp, rs) = task
    tok = BasicTokenizer(lowercase=lc, remove_punct=rp, remove_stopwords=rs)
    c = Counter()
    for text in TEXTS[split]:
        c.update(tok.tokens(text))
    return split, (lc, rp, rs), c


tasks = [(split, cfg) for cfg in CONFIGS for split in ("train", "test")]
counts = {cfg: {} for cfg in CONFIGS}
t0 = time.time()
print(f"Tokenizing {len(tasks)} (split, setting) tasks with {args.workers} workers ...", flush=True)
with mp.get_context("fork").Pool(args.workers) as pool:  # fork: workers inherit TEXTS
    for split, cfg, c in pool.imap_unordered(count_tokens, tasks):
        counts[cfg][split] = c
        print(f"  done {split:5} lowercase={cfg[0]!s:5} punct={cfg[1]!s:5} stop={cfg[2]!s:5} "
              f"({sum(c.values())} tokens, {len(c)} types)", flush=True)
print(f"Tokenization took {time.time() - t0:.0f}s", flush=True)

# words we want vectors for (lowercased, because PretrainedEmbeddings lowercases its lookups)
wanted = {t.lower() for by_split in counts.values() for c in by_split.values() for t in c}
print(f"IMDB distinct lowercased tokens over all settings: {len(wanted)}", flush=True)


# ---------------------------------------------------------------- one pass over GloVe
def open_lines(path):
    if path.endswith(".zip"):
        with ZipFile(path) as z:
            names = z.namelist()
            assert len(names) == 1, f"expected one member, found {names}"
            with z.open(names[0]) as f:
                yield from TextIOWrapper(f, encoding="utf-8", errors="replace")
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            yield from f


cased = set()  # exact-case vocabulary
lower = set()  # lowercased vocabulary (first occurrence wins, as in PretrainedEmbeddings)
vectors = {}  # lowercased word -> vector, only for words in `wanted`
total = np.zeros(args.dims, dtype=np.float64)
n_unique = 0
t0 = time.time()
print("Scanning GloVe ...", flush=True)
for i, line in enumerate(open_lines(args.glove)):
    parts = line.split()
    if len(parts) - 1 != args.dims:  # malformed line, skipped like in PretrainedEmbeddings
        continue
    w = parts[0]
    lw = w.lower()
    if lw in lower:
        cased.add(w)
        continue
    try:
        vec = np.asarray(parts[1:], dtype=np.float32)
    except ValueError:  # non-numeric field: malformed line, skip
        continue
    cased.add(w)
    lower.add(lw)
    total += vec
    n_unique += 1
    if lw in wanted:
        vectors[lw] = vec
    if (i + 1) % 500_000 == 0:
        print(f"  {i + 1} lines, {time.time() - t0:.0f}s", flush=True)
print(f"GloVe: {len(cased)} cased words, {len(lower)} lowercased unique words ({time.time() - t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- coverage table (train text)
rows, oov_top = [], {}
n_reviews = len(train)
for cfg in CONFIGS:
    c = counts[cfg]["train"]
    n_tok, n_typ = sum(c.values()), len(c)
    known_tok = sum(k for t, k in c.items() if t.lower() in lower)
    known_typ = sum(1 for t in c if t.lower() in lower)
    exact_tok = sum(k for t, k in c.items() if t in cased)
    # what the model really sees: BasicTokenizer.ids() does vocab.index(token) with NO lowercasing,
    # against the lowercased vocabulary, so a capitalised token ("Great") is unknown when lowercase=False
    eff_tok = sum(k for t, k in c.items() if t in lower)
    eff_typ = sum(1 for t in c if t in lower)
    rows.append({
        "lowercase": cfg[0], "remove_punct": cfg[1], "remove_stopwords": cfg[2],
        "tokens_per_review": n_tok / n_reviews,
        "n_types": n_typ,
        "pct_tokens_in_vocab": 100 * known_tok / n_tok,
        "pct_types_in_vocab": 100 * known_typ / n_typ,
        "pct_tokens_exact_case": 100 * exact_tok / n_tok,
        "pct_tokens_effective": 100 * eff_tok / n_tok,
        "pct_types_effective": 100 * eff_typ / n_typ,
        "negations_per_review": sum(c[t] for t in NEGATIONS) / n_reviews,
    })
    oov = Counter({t: k for t, k in c.items() if t.lower() not in lower})
    oov_top[f"lowercase={cfg[0]},punct={cfg[1]},stop={cfg[2]}"] = oov.most_common(50)

df = pd.DataFrame(rows)
df.to_csv(out / "coverage.csv", index=False)
with open(out / "oov_top50.json", "w") as f:
    json.dump(oov_top, f, indent=1)

probes = {p: {"exact": p in cased, "lowercased": p.lower() in lower} for p in PROBES}
with open(out / "probes.json", "w") as f:
    json.dump(probes, f, indent=1)

# ---------------------------------------------------------------- pruned embeddings (D2)
words = sorted(vectors)
np.save(out / "glove840b_imdb.npy", np.stack([vectors[w] for w in words]).astype(np.float32))
(out / "glove840b_imdb_words.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
np.save(out / "glove840b_mean.npy", (total / n_unique).astype(np.float32))  # <unk> vector of the FULL vocabulary
with open(out / "meta.json", "w") as f:
    json.dump({
        "glove": args.glove, "dims": args.dims, "sample": args.sample,
        "train_reviews": n_reviews, "test_reviews": len(test),
        "glove_unique_lowercased": n_unique, "imdb_wanted": len(wanted), "kept_in_pruned_file": len(words),
    }, f, indent=1)

pd.set_option("display.width", 200)
print("\n=== Coverage on IMDB train (cleaned, de-duplicated) ===")
print(df.round(2).to_string(index=False))
print("\n=== Does GloVe contain these tokens? ===")
for p, v in probes.items():
    print(f"  {p!r:8} exact={v['exact']!s:5} lowercased={v['lowercased']}")
print(f"\nPruned file: {len(words)} of {len(wanted)} IMDB types have vectors; saved to {out}")
