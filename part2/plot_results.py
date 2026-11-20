"""
Project item 5: box and point plots of fold accuracies (all seeds x folds, no averages only) and the data-size curve.

    python part2/plot_results.py            # reads part2/results/imdb and part1/results, writes part2/results/imdb/plots/

Each point is one fold of one seed. Folders that do not exist yet are skipped with a message.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

here = Path(__file__).resolve().parent
imdb = here / "results" / "imdb"
part1 = here.parent / "part1" / "results"
out = imdb / "plots"  # copy the png and csv files to part2/figures/ to share via git
out.mkdir(parents=True, exist_ok=True)

# label -> folder containing seed-*/accuracies.json
SYSTEMS = {
    "EP frozen d0.5": imdb / "embed-and-pool" / "baseline",
    "EP frozen d0": imdb / "embed-and-pool" / "dropout-0",
    "EP random-init d0": imdb / "embed-and-pool-random-init" / "baseline",
    "EP trainable d0": imdb / "embed-and-pool-trainable" / "baseline",
    "LSTM frozen d0.5": imdb / "lstm" / "baseline",
    "LSTM trainable d0": imdb / "lstm-trainable-d0" / "baseline",
}
for d in sorted(part1.glob("*-enc")):
    SYSTEMS[f"P1 {d.name.replace('-enc', '')}"] = d


def load(folder):
    rows = []
    for f in sorted(Path(folder).glob("seed-*/accuracies.json")):
        seed = int(f.parent.name.split("-")[1])
        for fold, acc in json.loads(f.read_text()).items():
            rows.append({"seed": seed, "fold": int(fold), "accuracy": float(acc)})
    return rows


rows = []
for label, folder in SYSTEMS.items():
    r = load(folder)
    if not r:
        print(f"skip (no results): {label}  {folder}")
        continue
    rows += [{"model": label, **x} for x in r]
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no results found")
df.to_csv(out / "fold_accuracies.csv", index=False)
print(df.groupby("model")["accuracy"].agg(["mean", "std", "min", "max", "count"]).round(4).to_string())

fig, ax = plt.subplots(figsize=(max(8, 1.2 * df["model"].nunique()), 4.5))
order = list(df["model"].unique())
sns.boxplot(data=df, x="model", y="accuracy", order=order, showfliers=False, color="lightgray", ax=ax)
sns.stripplot(data=df, x="model", y="accuracy", order=order, hue="seed", palette="deep", dodge=False, size=4, ax=ax)
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=30)
for t in ax.get_xticklabels():
    t.set_ha("right")
fig.tight_layout()
fig.savefig(out / "box_points.png", dpi=200)
print("saved", out / "box_points.png")

# data-size curve: SIZES maps label -> parent folder holding n<N>/baseline/seed-*/accuracies.json
SIZES = {
    "EP trainable": imdb / "size-ep-trainable",
    "EP random-init": imdb / "size-ep-random",
    "LSTM trainable": imdb / "size-lstm-trainable",
}
rows = []
for label, parent in SIZES.items():
    for d in sorted(parent.glob("n*")):
        n = int(d.name[1:])
        rows += [{"model": label, "n": n, **x} for x in load(d / "baseline")]
if rows:
    s = pd.DataFrame(rows)
    s.to_csv(out / "size_accuracies.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=s, x="n", y="accuracy", hue="model", marker="o", errorbar="sd", ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("training reviews")
    fig.tight_layout()
    fig.savefig(out / "data_size.png", dpi=200)
    print("saved", out / "data_size.png")
else:
    print("skip data-size plot (no size-* results yet)")
