"""
Project item 3 (A): embed-and-pool cross-validation on IMDB, with pruned GloVe 840B/300d (D2).

One axis at a time (D4): the baseline config plus ONE changed hyperparameter per run, e.g.
    python part2/run_imdb_embpool.py --axis hidden --values 25 50 100 200 --seeds 0          # screen
    python part2/run_imdb_embpool.py --seeds 0 3 5 --keep-models                             # confirm baseline
Results: part2/results/imdb/embed-and-pool/<tag>/seed-<s>/accuracies.json (same format as Part 1, so the
Part 1 loader/plots work) and, with --keep-models, fold-<k>/model.{json,pt}.

Needs the pruned embeddings from coverage_imdb.py (part2/results/imdb/embeddings/).
Tokenizer defaults keep negations (no stopword removal, no punctuation removal): see the coverage results.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import part1  # noqa: E402
import part2  # noqa: E402
from datasets import load_dataset  # noqa: E402
from part2.pruned_embeddings import PrunedEmbeddings  # noqa: E402

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir", default=str(here / "results" / "imdb" / "embeddings"))
parser.add_argument("--results", default=str(here / "results" / "imdb" / "embed-and-pool"))
parser.add_argument("--seeds", type=int, nargs="+", default=[0])
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--device", default="cuda")
parser.add_argument("--sample", type=int, default=0, help="use N training reviews only (quick test)")
parser.add_argument("--keep-models", action="store_true", help="keep fold-*/model.pt (needed for error analysis)")
# tokenizer (all pruned words are lowercase, so keep lowercase on)
parser.add_argument("--remove-punct", action="store_true")
parser.add_argument("--remove-stopwords", action="store_true")
# baseline hyperparameters
parser.add_argument("--hidden", type=int, default=50)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--trainable", action="store_true", help="fine-tune the embeddings (default: frozen)")
parser.add_argument("--sum-pool", action="store_true", help="sum instead of mean pooling")
# one-axis sweep
parser.add_argument("--axis", choices=["hidden", "dropout", "lr", "wd"], default=None)
parser.add_argument("--values", type=float, nargs="+", default=[])
args = parser.parse_args()


def clean(text):
    """Same cleaning as Part 1: drop HTML tags, normalise whitespace."""
    return " ".join(re.sub(r"</?(br|i|em|hr)\s*/?>", " ", text).split())


class CachingTokenizer:
    """Memoises `ids` per text. `train()` re-tokenizes every text in every fold and seed, and IMDB reviews are long."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cache = {}

    def tokens(self, text):
        return self.tokenizer.tokens(text)

    def ids(self, text):
        if text not in self.cache:
            self.cache[text] = self.tokenizer.ids(text)
        return self.cache[text]


print("Loading embeddings and IMDB ...", flush=True)
emb_model = PrunedEmbeddings(args.emb_dir)
tokenizer = CachingTokenizer(part2.clslib.BasicTokenizer(
    lowercase=True, remove_punct=args.remove_punct, remove_stopwords=args.remove_stopwords,
    language="english", vocab=emb_model.vocab,
))

train = pd.DataFrame(load_dataset("mteb/imdb")["train"])
train["text"] = train["text"].map(clean)
train = train.drop_duplicates(subset="text").reset_index(drop=True)
if args.sample:
    train = train.sample(min(args.sample, len(train)), random_state=0)
pairs = list(zip(train["text"], train["label"]))
outcome_space = part1.clslib.NamedClasses(["negative", "positive"])
print(f"{len(pairs)} training reviews, embeddings {emb_model.vectors.shape}", flush=True)

base = {"hidden": args.hidden, "dropout": args.dropout, "lr": args.lr, "wd": args.wd}
if args.axis is None:
    configs = [("baseline", dict(base))]
else:
    caster = int if args.axis == "hidden" else float
    configs = [(f"{args.axis}-{v:g}", {**base, args.axis: caster(v)}) for v in args.values]

results_folder = Path(args.results)
for tag, hp in configs:
    for seed in args.seeds:
        run_folder = results_folder / tag / f"seed-{seed}"
        run_folder.mkdir(parents=True, exist_ok=True)
        print(f"=== {tag} seed {seed}: {hp}", flush=True)
        t0 = time.time()
        splits = part1.experiment.cross_validation_split(pairs=pairs, k=args.folds, seed=seed)
        out = part2.experiment.cross_validation(
            outcome_space=outcome_space,
            data_splits=splits,
            tokenizer=tokenizer,
            emb_model=emb_model,
            enc_type="embed-and-pool",
            enc_kwargs={"freeze_embeddings": not args.trainable, "average_pool": not args.sum_pool},
            regressor_kwargs={"hidden_size1": hp["hidden"], "hidden_size2": hp["hidden"], "dropout": hp["dropout"]},
            opt_kwargs={"learning_rate": hp["lr"], "weight_decay": hp["wd"],
                        "batch_size": args.batch_size, "num_epochs": args.epochs},
            device=args.device,
            save_to=str(run_folder),
        )
        with open(run_folder / "accuracies.json", "w") as f:
            json.dump({int(k): float(v) for k, v in out["accuracies"].items()}, f, indent=1)
        accs = list(out["accuracies"].values())
        print(f"{tag} seed {seed}: mean acc {sum(accs) / len(accs):.4f}  "
              f"folds {[round(a, 4) for a in accs]}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if not args.keep_models:
            for pt in run_folder.glob("fold-*/model.pt"):
                pt.unlink()  # sweep runs: keep accuracies only, models are large

print("All done.", flush=True)
