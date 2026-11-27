"""Metrics helper: accuracy, macro-F1, confusion matrix, bootstrap CI, paired bootstrap.

Invalid predictions (-1) count as wrong, get their own confusion column, and are
reported separately as `invalid_rate`.
"""
import numpy as np
from sklearn.metrics import f1_score


def summarize(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    cm = np.zeros((2, 3), dtype=int)  # rows: gold 0/1; cols: pred 0/1/invalid
    for g, p in zip(gold, pred):
        cm[g, 2 if p == -1 else p] += 1
    return {
        "accuracy": float((gold == pred).mean()),
        "macro_f1": float(f1_score(gold, pred, labels=[0, 1], average="macro", zero_division=0)),
        "invalid_rate": float((pred == -1).mean()),
        "confusion": cm,
    }


def bootstrap_ci(gold, pred, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI over reviews for accuracy."""
    correct = (np.asarray(gold) == np.asarray(pred)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(axis=1)
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


def paired_bootstrap(gold, pred_a, pred_b, rope=0.01, n_boot=2000, seed=0):
    """Paired bootstrap on the same reviews. Returns mean diff (a-b), 95% CI, and
    P(a better), P(tie within +-rope), P(b better) over bootstrap samples."""
    gold = np.asarray(gold)
    d = (gold == np.asarray(pred_a)).astype(float) - (gold == np.asarray(pred_b)).astype(float)
    rng = np.random.default_rng(seed)
    diffs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {
        "diff": float(d.mean()),
        "ci": (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))),
        "p_a": float((diffs > rope).mean()),
        "p_tie": float((np.abs(diffs) <= rope).mean()),
        "p_b": float((diffs < -rope).mean()),
    }
