"""
Run MLP and DeepMLP cross-validation experiments on IMDb for both encoders.
Designed to run on Snellius. Results saved to part1/results/.
"""
import sys
import re
import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import Counter
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
import part1

results_folder = Path("part1/results")
results_folder.mkdir(parents=True, exist_ok=True)

# --- Load and clean data ---
def clean(text):
    no_html = re.sub(r"</?(br|i|em|hr)\s*/?>", " ", text)
    return " ".join(no_html.split())

print("Loading IMDb...")
df_raw = load_dataset("mteb/imdb")

import pandas as pd
df_train = pd.DataFrame(df_raw['train'])
df_test  = pd.DataFrame(df_raw['test'])
df_train["text"] = df_train["text"].apply(clean)
df_test["text"]  = df_test["text"].apply(clean)
df_train = df_train.drop_duplicates(subset="text")

train_texts  = df_train["text"].tolist()
train_labels = df_train["label"].tolist()

imdb_outcome_space = part1.clslib.NamedClasses(['negative', 'positive'])

# --- Load tokenizer ---
tokenizer_path = results_folder / "BPE_tokenizer_vs_1000_mf_2.json"
tokenizer = Tokenizer.from_file(str(tokenizer_path))
print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

# --- Encoders ---
class BagOfWords(part1.clslib.Encoder):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __len__(self):
        return self.tokenizer.get_vocab_size()
    def __call__(self, x: str):
        token_ids = self.tokenizer.encode(x).ids
        vector = np.zeros(len(self))
        for tid in token_ids:
            vector[tid] += 1
        return vector

token_indicator_enc = part1.clslib.TokenIndicator(tokenizer)
bag_of_words_enc    = BagOfWords(tokenizer)

class tfidf(part1.clslib.Encoder):
    def __init__(self, tokenizer, training_texts):
        self.tokenizer = tokenizer
        self.count_texts = len(training_texts)

        self.df = np.zeros(self.tokenizer.get_vocab_size())

        for i in training_texts:
            ids = self.tokenizer.encode(i).ids
            unique = set(ids)

            for j in unique:
                self.df[j] += 1

        self.idf = []
        for i in self.df:
            self.idf.append(np.log(self.count_texts / (1 + i)))
    
        self.idf = np.array(self.idf)

    def __len__(self):
        return len(self.idf)

    def __call__(self, x):
        id = self.tokenizer.encode(x).ids
        vector = np.zeros(len(self))

        for i in id:
            vector[i] += 1

        total = len(id)
        if total > 0:
            vector = vector / total
            
        return vector * self.idf

tfidf_enc    = tfidf(tokenizer, train_texts)

# --- Cross-validation splits ---
my_seeds = [0]
num_folds = 5
print("Building CV splits...")
my_cross_validation_sets = [
    part1.experiment.cross_validation_split(
        pairs=list(zip(train_texts, train_labels)),
        k=num_folds,
        seed=seed
    )
    for seed in my_seeds
]

# --- Run MLP experiments ---
my_classifiers = {}

print("Running MLP + TokenIndicator...")
my_classifiers['mlp_indicator-enc'] = []
for run, seed in enumerate(my_seeds):
    run_folder = results_folder / f"mlp_indicator-enc/seed-{seed}"
    my_classifiers['mlp_indicator-enc'].append(
        part1.experiment.cross_validation_experiment(
            my_cross_validation_sets[run], imdb_outcome_space,
            token_indicator_enc, model_option='mlp', save_to=str(run_folder)
        )
    )

print("Running MLP + BagOfWords...")
my_classifiers['mlp_bow-enc'] = []
for run, seed in enumerate(my_seeds):
    run_folder = results_folder / f"mlp_bow-enc/seed-{seed}"
    my_classifiers['mlp_bow-enc'].append(
        part1.experiment.cross_validation_experiment(
            my_cross_validation_sets[run], imdb_outcome_space,
            bag_of_words_enc, model_option='mlp', save_to=str(run_folder)
        )
    )


print("Running tf-idf + BagOfWords...")
my_classifiers['mlp_tfidf-enc'] = []
for run, seed in enumerate(my_seeds):
    run_folder = results_folder / f"mlp_tfidf-enc/seed-{seed}"
    my_classifiers['mlp_tfidf-enc'].append(
        part1.experiment.cross_validation_experiment(
            my_cross_validation_sets[run], imdb_outcome_space,
            tfidf_enc, model_option='mlp', save_to=str(run_folder)
        )
    )


print("Running Deep MLP + TokenIndicator...")
my_classifiers['deep_mlp_indicator-enc'] = []
for run, seed in enumerate(my_seeds):
    run_folder = results_folder / f"deep_mlp_indicator-enc/seed-{seed}"
    my_classifiers['deep_mlp_indicator-enc'].append(
        part1.experiment.cross_validation_experiment(
            my_cross_validation_sets[run], imdb_outcome_space,
            token_indicator_enc, model_option='mlp', save_to=str(run_folder),
            smlp_args={'model_class': part1.smlp.DeepSparseMLP}
        )
    )

print("Running Deep MLP + BagOfWords...")
my_classifiers['deep_mlp_bow-enc'] = []
for run, seed in enumerate(my_seeds):
    run_folder = results_folder / f"deep_mlp_bow-enc/seed-{seed}"
    my_classifiers['deep_mlp_bow-enc'].append(
        part1.experiment.cross_validation_experiment(
            my_cross_validation_sets[run], imdb_outcome_space,
            bag_of_words_enc, model_option='mlp', save_to=str(run_folder),
            smlp_args={'model_class': part1.smlp.DeepSparseMLP}
        )
    )

print("Running Deep MLP + tf-idf...")
my_classifiers['deep_mlp_tfidf-enc'] = []
for run, seed in enumerate(my_seeds):
    run_folder = results_folder / f"deep_mlp_tfidf-enc/seed-{seed}"
    my_classifiers['deep_mlp_tfidf-enc'].append(
        part1.experiment.cross_validation_experiment(
            my_cross_validation_sets[run], imdb_outcome_space,
            tfidf_enc, model_option='mlp', save_to=str(run_folder),
            smlp_args={'model_class': part1.smlp.DeepSparseMLP}
        )
    )


# --- Save accuracies to JSON and print summary ---
import json
for name, runs in my_classifiers.items():
    for run_idx, run in enumerate(runs):
        accs = run['accuracies']
        seed = my_seeds[run_idx]
        out_path = results_folder / f"{name}/seed-{seed}/accuracies.json"
        with open(out_path, 'w') as f:
            json.dump({str(k): v for k, v in accs.items()}, f)
        print(f"{name} seed-{seed}: mean={np.mean(list(accs.values())):.4f} std={np.std(list(accs.values())):.4f}")

print("Done. Results saved to part1/results/")