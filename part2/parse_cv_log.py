"""
Recover per-fold accuracies from a cross-validation job log and write accuracies.json per seed.

`part2.experiment.train` saves the model of the epoch with the lowest validation loss (ties go to the later
epoch) and `cross_validation` reports that epoch's validation accuracy. The log prints both per epoch, so we
replay the same rule. Accuracies are as printed (4 decimals).

Usage:
  python part2/parse_cv_log.py logs/agnews_ep_29416769.out part2/results/agnews/embed-and-pool --seeds 0 3 5 --folds 5
"""
import argparse
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("log")
parser.add_argument("out_folder", help="folder holding seed-<s>/ subfolders, e.g. part2/results/agnews/embed-and-pool")
parser.add_argument("--seeds", type=int, nargs="+", required=True)
parser.add_argument("--folds", type=int, default=5)
args = parser.parse_args()

pattern = re.compile(r"Epoch (\d+) \| train loss: [\d.]+ \| train acc: [\d.]+ \| val loss: ([\d.]+) \| val acc: ([\d.]+)")

runs = []  # one list of (val_loss, val_acc) per fold, in the order the folds were trained
for line in Path(args.log).read_text().splitlines():
    m = pattern.search(line)
    if not m:
        continue
    epoch, val_loss, val_acc = int(m.group(1)), float(m.group(2)), float(m.group(3))
    if epoch == 1:
        runs.append([])
    runs[-1].append((val_loss, val_acc))

expected = len(args.seeds) * args.folds
if len(runs) != expected:
    raise SystemExit(f"found {len(runs)} folds in the log, expected {expected} ({len(args.seeds)} seeds x {args.folds} folds)")

best_acc = []
for epochs in runs:
    best_loss, acc = None, None
    for i, (vl, va) in enumerate(epochs):
        if i == 0 or vl <= best_loss:
            best_loss, acc = vl, va
    best_acc.append(acc)

for s, seed in enumerate(args.seeds):
    accs = {fold: best_acc[s * args.folds + fold] for fold in range(args.folds)}
    folder = Path(args.out_folder) / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "accuracies.json").write_text(json.dumps(accs, indent=1))
    print(f"seed {seed}: " + "  ".join(f"{a:.4f}" for a in accs.values()) + f"   mean {sum(accs.values()) / len(accs):.4f}")
