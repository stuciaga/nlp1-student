"""Own stress test: contrast. Label-preserving, model-free.

For severity k, prepend k concessions of the OPPOSITE sentiment plus "However,", then the
original review (whose verdict stays the true label). A bag-of-words system sees many
opposite-sentiment words; a system that understands "however" should not be fooled.
Control: k neutral filler sentences (no sentiment) of similar length, so a drop under
contrast but not under filler is due to the contrast, not just longer input.

Base reviews: fixed class-balanced subsample of the IMDB test set (k=0 is the unperturbed
version of the same reviews). Writes <out>/stress_contrast_k{k}.pkl and
stress_neutral_k{k}.pkl, each a list of (text, label) in the same order.

Run from repo root (after build_folds.py):  python part3/utils/build_stress.py
"""
import argparse
import pickle
import random
from pathlib import Path

# Concessions that sound like the label they are named after (domain-neutral wording)
NEG_CONCESSIONS = [
    "The first half felt slow and the acting seemed wooden.",
    "Some scenes were boring and the dialogue was weak.",
    "I expected to be disappointed by the story and the pacing.",
    "People told me it was terrible and a waste of time.",
    "The opening was dull and I nearly gave up early on.",
    "The script has clumsy moments and a few awful lines.",
]
POS_CONCESSIONS = [
    "The first half was gripping and the acting seemed brilliant.",
    "Some scenes were beautiful and the dialogue was sharp.",
    "I expected to love the story and the pacing.",
    "People told me it was wonderful and well worth the time.",
    "The opening was charming and I was hooked early on.",
    "The script has clever moments and a few superb lines.",
]
NEUTRAL_FILLER = [
    "I watched this on a Tuesday evening after work.",
    "The film runs for a little over two hours.",
    "I saw it with a friend who had a free ticket.",
    "It was released a few years ago in several countries.",
    "The cinema was about half full when we arrived.",
    "I read a bit about the cast before sitting down.",
]


def perturb(text, label, k, rng, neutral=False):
    if k == 0:
        return text
    if neutral:
        return " ".join(rng.sample(NEUTRAL_FILLER, k)) + " " + text
    opposite = POS_CONCESSIONS if label == 0 else NEG_CONCESSIONS  # label 0 = negative review
    return " ".join(rng.sample(opposite, k)) + " However, " + text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_path", default="part3/data/test.pkl")
    ap.add_argument("--out", default="part3/data/robustness")
    ap.add_argument("--n", type=int, default=2000, help="total reviews, split evenly by class")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    test = pickle.load(open(a.test_path, "rb"))
    rng = random.Random(a.seed)
    base = []
    for label in (0, 1):
        pool = [x for x in test if x[1] == label]
        base += rng.sample(pool, a.n // 2)
    rng.shuffle(base)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("contrast", "neutral"):
        for k in a.ks:
            r = random.Random(a.seed + 100 * k)  # same concessions picked per (k, review order)
            data = [(perturb(t, l, k, r, neutral=(kind == "neutral")), l) for t, l in base]
            pickle.dump(data, open(out / f"stress_{kind}_k{k}.pkl", "wb"))
    print(f"{len(base)} base reviews, ks={a.ks}")
    for k in (0, 2):
        t, l = pickle.load(open(out / f"stress_contrast_k{k}.pkl", "rb"))[0]
        print(f"k={k} label={l}: {t[:260]}...")
