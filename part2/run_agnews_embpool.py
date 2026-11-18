"""
T9 (A): embed-and-pool cross-validation on AG News, 3 seeds x 5 folds x 20 epochs.
Mirrors the tutorial cells (dataset, CV splits, tokenizer, `part2.experiment.cross_validation`).
Designed for Snellius (GPU). Results go to part2/results/agnews/embed-and-pool/seed-<s>/fold-<k>/.

Prepare on the login node first (downloads need internet): see prep_agnews.sh.
"""
import argparse
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import part1  # noqa: E402
import part2  # noqa: E402
from datasets import load_dataset  # noqa: E402

EMBEDDINGS = {  # name -> (zip filename, dims, skip_first_line)
    "glove": ("glove.2024.wikigiga.50d.zip", 50, False),
    "fasttext": ("wiki-news-300d-1M.vec.zip", 300, True),
}

parser = argparse.ArgumentParser()
parser.add_argument("--emb", choices=EMBEDDINGS, default="fasttext")
parser.add_argument("--emb-dir", default=".", help="folder containing the embedding zip")
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 3, 5])
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--device", default="cuda")
parser.add_argument("--results", default=str(Path(__file__).resolve().parent / "results" / "agnews"))
args = parser.parse_args()

results_folder = Path(args.results)
results_folder.mkdir(parents=True, exist_ok=True)

fname, dims, skip_first = EMBEDDINGS[args.emb]
emb_path = str(Path(args.emb_dir) / fname)
print(f"Loading embeddings from {emb_path} ({dims}d)", flush=True)
t0 = time.time()
emb_model = part2.clslib.PretrainedEmbeddings(emb_path, dims, skip_first_line=skip_first)
print(f"Embeddings loaded in {time.time() - t0:.0f}s", flush=True)

tokenizer_for_bow = part2.clslib.BasicTokenizer(
    lowercase=True,
    remove_punct=True,
    remove_stopwords=True,
    language="english",
    vocab=emb_model.vocab,
)

outcome_space = part1.clslib.NamedClasses(["world", "sports", "business", "tech/sci-fi"])
dataset = load_dataset("fancyzhx/ag_news")
pairs = list(zip(dataset["train"]["text"], dataset["train"]["label"]))

for seed in args.seeds:
    splits = part1.experiment.cross_validation_split(pairs=pairs, k=args.folds, seed=seed)
    run_folder = f"{results_folder}/embed-and-pool/seed-{seed}"
    print(f"=== seed {seed} -> {run_folder}", flush=True)
    t0 = time.time()
    part2.experiment.cross_validation(
        outcome_space=outcome_space,
        data_splits=splits,
        tokenizer=tokenizer_for_bow,
        emb_model=emb_model,
        enc_type="embed-and-pool",
        enc_kwargs={"freeze_embeddings": True},
        regressor_kwargs={"hidden_size1": 50, "hidden_size2": 50, "dropout": 0.5},
        opt_kwargs={"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "num_epochs": args.epochs},
        device=args.device,
        save_to=run_folder,
    )
    print(f"seed {seed} done in {(time.time() - t0) / 60:.1f} min", flush=True)

print("All seeds done.", flush=True)
