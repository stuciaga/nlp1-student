# Part 3: A's notes (running log for slides and blog)

Scope: A's items in `workplan.md` (setup, prompting track, OOD, own stress test, comparison table).
Add one entry per session: what was done, what was decided and why, numbers, caveats. Mark `TODO(blog)` where a sentence is needed.

## Status

| Item | State |
|---|---|
| Setup: folds rebuild | Done, ran 2026-11-27 |
| Setup: binary prompt + parser | Done, unit-tested; not yet run on a model |
| Setup: metrics helper | Done, unit-tested |
| Prompting track (§2) | Not started |
| OOD dataset (§5) | Dataset built (Yelp polarity, 2,000 reviews); no system run on it yet |
| Own stress test (§5) | Contrast test datasets built (k=0..3 plus neutral control); no system run on them yet |
| Merged comparison table (§4) | Not started |

## Decisions so far

| Decision | Choice | Why |
|---|---|---|
| Output framing | Multiple-choice letter (A = negative, B = positive) | Constrained output, trivial to parse, same format as the tutorial |
| Label order | Configurable (`--label_order`) | Label-order permutation is a ready-made stress test |
| Parser | Strict letter regex, then lone label word; ambiguous = invalid (-1) | Tutorial's `"A" in s` mapped "Answer: B" to class 0 |
| Invalid outputs | Retry up to 3x with sampling, log first-pass and final invalid rate | Report both, since retries can hide a weak prompt |
| `max_new_tokens` | 8 (was 300) | Answer is one letter; saves time |
| Review truncation | 512 tokens, count logged per run | Cost control; must report how many were cut |
| Ablation subsample | `--subsample N`, fixed seed 1234 | Same reviews for every config; N still open (workplan §9) |
| OOD dataset | Yelp polarity test split, 1,000 pos + 1,000 neg, seed 0 | Real domain shift (restaurants vs movies, different vocabulary). SST-2 was rejected: also movie reviews, so it mostly tests length. Caveats for blog: star-to-binary mapping (3 stars dropped upstream), label noise near the 2/4-star boundary, shorter reviews (median 100 words) than IMDB |
| Own stress test | Contrast (workplan option 1), label-preserving: k opposite-sentiment concessions + "However," prepended to the original review, k = 0..3 | Builds on the Part 2 finding that contrast words hurt; applies to every system (P1, P2, prompting, LoRA). Label-preserving, since the original verdict follows "However", so the true label is unchanged and any drop is due to contrast. Control: k neutral filler sentences, to separate contrast from longer input. Base: 2,000 balanced IMDB test reviews (seed 0). Caveats: templated concessions are artificial and few (6 per sentiment); 512-token truncation keeps the start, so the added clauses push a little of the original ending out of long reviews |
| Folds | Part 1 cleaning + dedup, seeds 0/3/5, 5 folds | Same folds as P1/P2 so results are comparable |

## Data facts (for the blog)

- Train after dedup: 24,902 reviews. Test: 24,678 (matches the workplan).
- Folds: about 4,980 validation reviews each; fold 0 is 49.4% positive.
- Test set is never used for tuning.

## Code map

- `part3/utils/build_folds.py`: rebuilds `part3/data/cross_validation_sets.pkl` and `test.pkl` (gitignored).
- `part3/utils/prompts.py`: instruction wordings (ids 0-3), demos, parser.
- `part3/utils/metrics.py`: accuracy, macro-F1, confusion matrix, bootstrap CI, paired bootstrap.
- `part3/utils/generate.py`: flags `--instruction_id --label_order --cv_run --temperature --top_p --subsample --max_review_tokens --demo_path`. Writes `<model>_<type>_<fold>.json` and `.stats.json` (accuracy, invalid rates, truncation, minutes, peak GPU memory).
- `part3/utils/build_ood.py`: writes `part3/data/robustness/yelp.pkl` (list of `(text, label)`; gitignored).
- `part3/utils/build_stress.py`: writes the contrast and neutral-control datasets to `part3/data/robustness/` (gitignored).
- `part3/utils/test_setup.py`: sanity tests for parser and metrics.

## Log

### 2026-11-27
- Built setup (see code map), rebuilt folds, wrote the notes file.
- Shell scripts `generate-0-shot.sh` and `generate-few-shot.sh` no longer hardcode `/home/eilia`. Park owns the rest.
- Not yet verified: any model run. `generate.py` has only been compile-checked.
- Chose Yelp polarity as OOD and built it with `part3/utils/build_ood.py` -> `part3/data/robustness/yelp.pkl` (2,000 reviews, 1,000 positive, median 100 words, max 910).
- Chose the contrast stress test and built it with `part3/utils/build_stress.py` -> `part3/data/robustness/stress_{contrast,neutral}_k{0..3}.pkl` (same 2,000 reviews, same order in every file).
- Next: fold-0 pilot (`--subsample 500`), check invalid rate and 0-shot accuracy; choose OOD dataset and stress test.

## Open questions

- Second model family: ~1B model vs `llama-3b-instruct`.
- Subsample size for ablations.
- Stress test sanity: read ~20 perturbed reviews by hand to confirm the label is really preserved (do this before the blog claims it).
- Label order (`--label_order`) is a free extra stress for prompted LLMs only, if time allows.
- Snellius budget and scratch availability.

## Results

_Nothing yet. Add tables here as `model | setting | fold | acc | CI | invalid | minutes`._

## Blog / slides snippets

- TODO(blog): why a letter instead of a label word.
- TODO(blog): how invalid outputs are counted and retried.
- TODO(blog): AI-usage note (Claude Code helped write the setup code; all choices reviewed by A).
