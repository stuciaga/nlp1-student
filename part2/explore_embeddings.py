"""
Project item 2 (A): what does the embedding capture, for a sentiment task?

Compares embeddings (e.g. GloVe 840B vs fasttext wiki-news) on sentiment-relevant checks:
  1. Synonyms vs antonyms: cosine similarity of hand-picked sentiment pairs vs random word pairs.
     (Distributional embeddings are known to place antonyms close: "good" and "bad" occur in the same contexts.)
  2. Neighbours: top-10 nearest neighbours of sentiment words (qualitative), and how often a lexicon word's
     lexicon neighbours have the SAME polarity (quantitative, NLTK opinion lexicon).
  3. Linear probe: cross-validated logistic regression predicting lexicon polarity from the vectors.

Neighbour search uses only the `--max-words` most frequent plain-alphabetic words (embedding files are sorted
by frequency), to avoid web junk such as URLs and emoji strings.

Example:
  python part2/explore_embeddings.py \
      --embs glove=/scratch-shared/$USER/nlp1/glove.840B.300d.zip:300 \
             fasttext=/scratch-shared/$USER/nlp1/wiki-news-300d-1M.vec.zip:300
Outputs go to part2/results/imdb/exploration/.
"""
import argparse
import json
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nltk  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from nltk.corpus import opinion_lexicon  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

SYNONYMS = [("good", "great"), ("bad", "terrible"), ("awful", "horrible"), ("excellent", "wonderful"),
            ("boring", "dull"), ("funny", "hilarious"), ("love", "adore"), ("hate", "despise"),
            ("beautiful", "gorgeous"), ("stupid", "dumb")]
ANTONYMS = [("good", "bad"), ("great", "terrible"), ("love", "hate"), ("beautiful", "ugly"),
            ("happy", "sad"), ("best", "worst"), ("excellent", "awful"), ("like", "dislike"),
            ("wonderful", "horrible"), ("brilliant", "stupid")]
QUERY_WORDS = ["good", "bad", "great", "terrible", "boring", "brilliant", "love", "hate", "not", "movie"]

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--embs", nargs="+", required=True, help="name=path:dims, e.g. glove=/x/glove.840B.300d.zip:300")
parser.add_argument("--max-words", type=int, default=200_000)
parser.add_argument("--out", default=str(here / "results" / "imdb" / "exploration"))
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

for r in ["opinion_lexicon"]:
    try:
        nltk.data.find(f"corpora/{r}")
    except LookupError:
        nltk.download(r)
POS = {w for w in opinion_lexicon.positive() if w.isalpha()}
NEG = {w for w in opinion_lexicon.negative() if w.isalpha()}
POS -= NEG  # drop the (rare) words listed in both


def open_lines(path):
    if path.endswith(".zip"):
        with ZipFile(path) as z:
            with z.open(z.namelist()[0]) as f:
                yield from TextIOWrapper(f, encoding="utf-8", errors="replace")
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            yield from f


def load_top(path, dims, max_words):
    """First `max_words` unique lowercase plain-alphabetic words, with unit-normalised vectors."""
    words, vecs, seen = [], [], set()
    for line in open_lines(path):
        parts = line.split()
        if len(parts) - 1 != dims:  # also skips the fasttext header line
            continue
        w = parts[0].lower()
        if not (w.isalpha() and w.isascii()) or w in seen:
            continue
        try:
            v = np.asarray(parts[1:], dtype=np.float32)
        except ValueError:
            continue
        seen.add(w)
        words.append(w)
        vecs.append(v)
        if len(words) >= max_words:
            break
    M = np.stack(vecs)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return words, M


rng = np.random.RandomState(args.seed)
summary, pair_rows, neighbour_lines = {}, [], []
for spec in args.embs:
    name, rest = spec.split("=", 1)
    path, dims = rest.rsplit(":", 1)
    dims = int(dims)
    print(f"[{name}] loading top {args.max_words} words from {path}", flush=True)
    words, M = load_top(path, dims, args.max_words)
    idx = {w: i for i, w in enumerate(words)}
    print(f"[{name}] {len(words)} words loaded", flush=True)

    def cos(a, b):
        return float(M[idx[a]] @ M[idx[b]])

    # 1) synonyms vs antonyms vs random frequent-word pairs
    groups = {}
    for label, pairs in [("synonyms", SYNONYMS), ("antonyms", ANTONYMS)]:
        kept = [(a, b) for a, b in pairs if a in idx and b in idx]
        groups[label] = [cos(a, b) for a, b in kept]
        for a, b in kept:
            pair_rows.append({"embedding": name, "group": label, "pair": f"{a}-{b}", "cosine": cos(a, b)})
    top = min(20_000, len(words))
    ia, ib = rng.randint(0, top, 2000), rng.randint(0, top, 2000)
    groups["random"] = [float(M[i] @ M[j]) for i, j in zip(ia, ib) if i != j]
    res = {g: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for g, v in groups.items()}

    # 2) neighbours
    for q in QUERY_WORDS:
        if q in idx:
            sims = M @ M[idx[q]]
            nn = [words[i] for i in np.argsort(-sims)[1:11]]
            neighbour_lines.append(f"[{name}] {q}: {' '.join(nn)}")
    lex = [(w, 0) for w in sorted(POS) if w in idx] + [(w, 1) for w in sorted(NEG) if w in idx]
    sample = [lex[i] for i in rng.permutation(len(lex))[:1000]]
    lex_pol = {w: p for w, p in lex}
    same = total = 0
    for w, p in sample:
        sims = M @ M[idx[w]]
        for i in np.argsort(-sims)[1:11]:
            nb = words[i]
            if nb in lex_pol:
                total += 1
                same += lex_pol[nb] == p
    res["neighbour_same_polarity_share"] = same / total if total else None
    res["neighbour_lexicon_hits"] = total

    # 3) linear probe on lexicon words
    X = np.stack([M[idx[w]] for w, _ in lex])
    y = np.array([p for _, p in lex])
    res["probe_accuracy"] = float(cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5).mean())
    res["probe_majority_baseline"] = float(max(y.mean(), 1 - y.mean()))
    res["lexicon_words_used"] = len(lex)
    summary[name] = res
    print(f"[{name}] {json.dumps(res, indent=1)}", flush=True)

(out / "summary.json").write_text(json.dumps(summary, indent=1))
(out / "neighbours.txt").write_text("\n".join(neighbour_lines) + "\n", encoding="utf-8")
pd.DataFrame(pair_rows).to_csv(out / "similarity_pairs.csv", index=False)

# figure: cosine of synonyms / antonyms / random pairs, per embedding
fig, axes = plt.subplots(1, len(summary), figsize=(4.5 * len(summary), 3.5), sharey=True, squeeze=False)
df = pd.DataFrame(pair_rows)
for ax, name in zip(axes[0], summary):
    for k, g in enumerate(["synonyms", "antonyms"]):
        vals = df[(df.embedding == name) & (df.group == g)].cosine
        ax.scatter([k] * len(vals), vals, alpha=0.7)
    ax.axhline(summary[name]["random"]["mean"], color="gray", ls="--", label="random pairs (mean)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["synonyms", "antonyms"])
    ax.set_title(name)
    ax.legend(fontsize=7)
axes[0][0].set_ylabel("cosine similarity")
fig.tight_layout()
fig.savefig(out / "synonym_antonym_cosine.png", dpi=150)

print("\n=== neighbours ===")
print("\n".join(neighbour_lines))
print(f"\nSaved to {out}")
