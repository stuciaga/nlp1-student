# Project 3 workplan

IMDB sentiment (binary) with LLMs: prompting and LoRA vs. the Project 1/2 classifiers, plus OOD, paraphrase and one own stress test. Must feed the blog post and be defensible orally.

## 0. Starting point

**Baselines to beat (IMDB, accuracy, 5-fold CV)**

| System                             | CV    | Test (24,678) |
| ---------------------------------- | ----- | ------------- |
| P1 MLP / deep MLP, BoW             | 0.83  | n/a           |
| P2 embed-and-pool, trainable GloVe | 0.898 | 0.885         |
| P2 LSTM, frozen GloVe              | 0.900 | 0.904         |

Reuse from Part 2:
- Cleaned and de-duplicated data (`part1/imdb.ipynb`), folds from `part1.experiment.cross_validation_split`, seeds 0/3/5.
- `predictions_test.csv` (30 models, per-review ids) and `features_test.csv` (length, readability, negations, contrast words, `unk_rate`).
- Part 2 metric style: accuracy, ROPE of 1 point, paired bootstrap. Skip the BDA system comparison (weakly identified).

Tutorial code gaps (`part3/utils/`):
- `generate.py`: prompt and parser are hardcoded to 4 AG News classes. `parse_generated_string` checks `"A" in output`, so "Answer: B" becomes class 0. Rewrite for binary labels and log the invalid rate.
- No CLI flags for sampling or instruction wording.
- `fine_tuning.py`: check whether `r`, `alpha` and `target_modules` have flags.
- Paths hardcoded to `/home/eilia/...` in the `.sh` scripts and argparse defaults.
- Tutorial results are fold 0 only, with no CIs.

## 1. Setup

1. Rebuild `cross_validation_sets.pkl` from the Part 1 cleaned data (same folds and seeds). Keep the test set separate.
2. Parameterise paths and job scripts. Use our own Snellius account and env.
3. Binary prompt and robust parser. Log the invalid rate.
4. Metrics helper: accuracy, macro-F1, confusion matrix, bootstrap 95% CI, paired bootstrap between systems.
5. Tidy results CSV (`model, method, setting, fold, seed, metric, value, gpu_minutes`). Plots read from it.
6. Log resources per run: wall-clock, peak GPU memory, trainable params.
7. **Cost control:** reviews average ~268 tokens. Estimate GPU-hours first. Evaluate on a fixed subsample per fold (e.g. 500-1000 reviews) for ablations, full dev folds only for final systems. Truncate long reviews, and report how many were cut.

## 1b. Work split (A = you, P = Park)

Split by system track. Each person owns one LLM method end to end and runs it through every setting (ID, OOD, paraphrase, stress).

| Item | Owner |
|---|---|
| Setup: folds rebuild, binary prompt and parser, metrics helper | A |
| Setup: path and job-script parameterisation, results CSV schema | P |
| Prompting track (§2) | A |
| LoRA track (§3) | P |
| OOD dataset (§5) | A |
| Paraphrase generation and validation (§5) | P |
| Own stress test (§5) | A |
| Mitigation idea (§6) and extra mile (§7) | P |
| Comparisons (§4) | A builds the merged table, P builds the resource plots |
| Blog and oral prep | Each writes their own items, then cross-reads |

Rules:
- **Agree first and freeze:** folds and seeds, prompt format, and results CSV columns (`model, method, setting, fold, seed, metric, value, gpu_minutes`).
- Separate results folders (`results/prompting/`, `results/lora/`, `results/data/`) and separate notebooks. Don't both edit `project.ipynb`.
- **Datasets are model-free** (OOD, paraphrases, stress inputs are built from text only), saved to `data/robustness/`. Each person then runs their own systems on all of them.
- Final merge: concatenate the CSVs. The comparison and mitigation code reads only from `results/`.
- Handoff points: setup done (both), datasets ready (A: OOD, stress; P: paraphrases), best configs frozen (both), then robustness runs.
- Run mitigation on one system first (e.g. LoRA) and extend to prompting if time allows, so it doesn't block on A.

## 2. Prompting (Q1) [A]

Model: `qwen-0.5b-instruct`. Framing: multiple-choice letter or label word, justified by constrained output and easy parsing. For a second family, add a ~1B model, or use `llama-3b-instruct` and state the 6x size difference.

Ablations (fold 0 pilots, then all folds):
1. Instruction wording, 3-5 variants (needs `--instruction_id`).
2. Number of demonstrations, k ∈ {0, 1, 2, 4, 8}. Context length limits k.
3. Demo choice and order (class balance, shuffled vs. grouped), 3-5 seeds.
4. Sampling: greedy vs. T ∈ {0.3, 0.7, 1.0} (needs new flags), several seeds.
5. Model family.

Output: one table and one plot per ablation, with error bars.

## 3. Fine-tuning (Q2) [P]

LoRA: frozen W plus trainable low-rank `BA`, so gradients and optimizer state cover ~0.1-1% of parameters.

Sweep one factor at a time on fold 0 (same protocol as Part 2), then confirm the best config on all folds and 3 seeds:
- Rank `r` ∈ {2, 4, 8, 16, 64}
- Target modules: q,v / all attention / attention+MLP
- Learning rate ∈ {5e-5, 1e-4, 2e-4, 5e-4}
- Epochs and training-set size. Part 2 had a data-size curve (500 to ~20k), so do the same for LoRA vs. P2 models.
- Max sequence length, as a resource lever.

Record trainable params, memory, train time and accuracy. Plot performance vs. cost and state the trade-off chosen.

## 4. Comparisons (Q3) [A table, P resource plots]

- 0-shot vs. best few-shot vs. LoRA vs. P1/P2 systems, on the same folds and seeds. Mean ± std plus paired bootstrap on the test set. Anything under the 1-point ROPE counts as a tie.
- Report both CV and test. Part 2 found CV overstated embed-and-pool by 1.3 points, so say which is the headline.
- Accuracy vs. train/inference time, memory and params (log cost axis).
- Overlap of errors: do LLMs fail on the same reviews as the P2 models? Join with `features_test.csv`.

## 5. Robustness (Q4)

**OOD [A]**
- Another sentiment domain: SST-2 (short sentences), Yelp or Amazon (reviews of other products). Pick one and justify.
- Run all main systems, including P2 models. Report the drop from IMDB.
- Caveats to state: label mapping (neutral/star ratings → binary), length mismatch, label noise.
- Optionally a small own set (tweets, headlines) with a second-annotator check.

**Paraphrase [P]**
- DIPPER, sampled per class. Long reviews are slow and may exceed the 512 max length, so paraphrase sentence by sentence or use a subsample.
- Check meaning preservation on ~50 pairs (manual or NLI/embedding), and especially that **sentiment** is preserved.
- Report accuracy on paraphrases, flip rate (correct→wrong, wrong→correct) and match rate.
- Also paraphrase the instruction.

**Own stress test [A] (pick one and justify)**
1. Contrast and negation: build inputs with added "but/however" clauses or flipped negation (motivated by the Part 2 finding that contrast words hurt).
2. Typos and casing, with a severity knob.
3. Length: truncate to first/last N sentences.
4. Label-order permutation (positive/negative swapped in the prompt).

Apply the same perturbation to all systems and plot performance vs. severity.

## 6. Mitigation idea (Q5) [P]

Selective prediction: abstain on low confidence (label-token probability, or paraphrase/sample agreement). Under shift, confidence should drop, so abstaining avoids errors without new training data. Alternatives: CoT, self-consistency.

## 7. Extra mile (optional) [P]

Risk-coverage curves on ID, OOD and the stress test, comparing confidence signals.

## 8. Order of work

| Step | A | P | Needs |
|---|---|---|---|
| 1 | Setup: folds, prompt, parser, metrics | Setup: paths, job scripts, CSV schema | n/a |
| 2 | Prompting ablations (fold-0 pilots first); build OOD and stress datasets | LoRA sweeps (start early, queue time); generate paraphrases | 1 |
| 3 | Freeze best prompting config | Freeze best LoRA config | 2 |
| 4 | Run prompting systems on OOD, paraphrase, stress | Run LoRA on OOD, paraphrase, stress | 2, 3, and the other's datasets |
| 5 | Merged comparison table | Resource plots, mitigation (then extra mile) | 4 |
| 6 | Clean notebook, own blog sections, AI-usage note, oral prep, cross-read | same | all |

## 9. Open questions

- Park did slides only in Part 2. Confirm she is up for running Snellius jobs this time.
- Which OOD dataset, and which second model family.
- Snellius SBU budget, and whether the old Part 2 scratch models are still available (expire after 14 days).
- Subsample size for ablations.
