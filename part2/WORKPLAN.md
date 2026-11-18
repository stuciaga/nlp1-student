# Part 2 Work Plan (You + Park)

Legend: **A** = you, **P** = Park. ⚖️ = **joint decision**, so discuss and agree before anyone proceeds. Record the outcome in section 3.

---

## 0. Heads-up before starting

| Issue | What to do |
|---|---|
| `from part2 import solutions` does not exist in the repo. The tutorial's `analyse_embs`, `demo_similarity`, `demo_topk` and `demo_analogy` cells will just print "You need your own implementation". | **This is part of the assignment.** We write `part2/solutions.py` ourselves, split as in section 1b. |
| Tutorial bug: cell `rm00 = ...` computes `rm00_dt_logprobs` with **`em00`** (the embed-and-pool model), not `rm00`. | Fix to `rm00` before trusting the LSTM classification report. |
| `device='cuda'` is hardcoded in the training cells. | Fine as is: we run training on Snellius (`job_mlp.sh` style). No change needed. |
| `my_seeds = [0]` and `num_folds = 5` by default. | D3 is decided: seeds `[0, 3, 5]`, 5 folds, 20 epochs. Set `my_seeds` accordingly. |
| GloVe download: the tutorial fetches `glove.2024.wikigiga.50d.zip` (50-d). | D1 is decided: we use GloVe Common Crawl 300d, so add a branch for it in `download_pretrained` (see D2 on pruning). |

---

## 1. Tutorial: what to try ourselves

Work through the tutorial in the same order, but each of us **implements** a different half of the "suggested exercise" code and then explains it to the other. The explanatory markdown can be read by both.

| # | Tutorial section | Exercise | Who implements | Both do |
|---|---|---|---|---|
| T1 | Vector semantics (toy skip-gram) | Answer the 2 *Questions* (why the `(good, day)` and `(day, good)` pmfs both changed, and which others changed) | Both, individually, then compare | Questions |
| T2 | Same | *Suggested*: full training loop with several positive and negative pairs | Both | n/a |
| T3 | Structure in embeddings | Opinion-lexicon exercise: PCA/t-SNE plot or k-means of `pos_words` vs `neg_words`, written as `solutions.analyse_embs` | **A** | Answer the 3 *Questions* |
| T4 | Similarity & analogies | `cos_similarity`, `topk_ids`, plus `solutions.demo_similarity`, `demo_topk`, `demo_analogy` (professor/man/woman, nurse, king) | **P** | Read the bias paper (Bolukbasi 2016) and answer the 2 *Questions* |
| T5 | Embed-and-pool | Complete `EmbedAndPoolEncoder.forward` (mask the padding before the mean!) | **A** | Run the dog-bites-man test |
| T6 | RNN encoder | Complete `RNNEncoder.forward` (Elman loop, handle padding) | **P** | Read the LSTM section |
| T7 | LSTM | *Suggested*: bidirectional recurrent encoder (LSTM or GRU, per D5) | **P** | Read `LSTMEncoder` in `clslib.py` |
| T8 | Softmax regressor | Complete `SoftmaxRegressor` (`__init__` and `forward`) | **A** | Run `test_softmax_regressor` |
| T9 | Cross-validation | Run embed-and-pool and LSTM CV on AG News (needs GPU). Save under `results/agnews/` (change `results_folder` and the hardcoded `results/...` paths in the load cells) so it doesn't collide with project runs | **Split**: A runs embed-and-pool, P runs LSTM (one Snellius job each) | Answer the 3 *Questions* |
| T10 | Analysis (BDA) | Run the Bayesian model, check r_hat and divergences, read the forest plots | Both together (30 min screen share) | Analysis *Questions* |

### 1b. The missing `part2/solutions.py` (summary)

The tutorial imports `part2/solutions.py`, which isn't in the repo, so we write it. Four functions plus two helpers, split as follows:

| Function | Owner | File |
|---|---|---|
| `analyse_embs` (do positive and negative words separate in embedding space?) | A | `solutions_a.py` |
| `cos_similarity`, `topk_ids` (the notebook's stubs) | P | `solutions_p.py` |
| `demo_similarity`, `demo_topk`, `demo_analogy` | P | `solutions_p.py` |

---

## 2. Project: split of work

The project reuses the **Project 1 dataset and task: IMDB** (confirmed), and anything from `part1/` can be reused, including the Part 1 baselines for the item 3 comparison. Numbers match the checklist in `project.ipynb`.

| # | Project item | Owner | Depends on | Output |
|---|---|---|---|---|
| 0 | **Shared foundation** (D1 and D3 decided, ⚖️ D2 open): embedding choice, splits/seeds, results folder layout, metric | **Together**, ~1 hr | Nothing | Short agreement in section 3 |
| 1 | Show that GloVe Common Crawl 300d fits IMDB: tokenizer coherence, vocabulary coverage | **A** | D1 | Coverage table (% of tokens and types in vocab) |
| 2 | Explore what the embedding captures: similarities relevant to our task, a comparison against one alternative embedding, known limitations | **A** | 1 | Plots and a 1-page note |
| 3 | Embed-and-pool experiments (embedding size, MLP sizes, dropout, lr, weight decay). Compare with Project 1 | **A** | 1, D3 | Results in `results/imdb/embed-and-pool/` |
| 4 | Recurrent encoder (LSTM or GRU): hidden size, frozen vs trainable embeddings, data-size curve, overfitting, word dropout | **P** | D1, D3 | Results in `results/imdb/lstm/` |
| 5 | Hyperparameter search structure and variance (multiple seeds and folds, box/point plots, no averages-only tables) | **Joint** (D4, decided) | 3 and 4 | One agreed protocol, then each applies it to their own track |
| 6a | Error analysis, **input features** (length, rare/unknown-word rate, negation, etc.) | **A** | 3 and 4 | Feature columns in the shared dataframe |
| 6b | Error analysis, **model features and BDA run** (predicted label, confidence, then fit and read BDA) | **P** | 3 and 4 | Forest plots and interpretation |
| 7 | Clean code and the final write-up (blog): each writes the sections for their own items, then cross-reads | **Both** | All | Blog draft |

### Why this lets us work independently

- The only hard dependency was the embedding choice (D1), which is decided (GloVe Common Crawl 300d), so both tracks can start right away.
- A and P write to **separate result folders** (`results/imdb/embed-and-pool/` and `results/imdb/lstm/`; tutorial runs go in `results/agnews/`) and never edit each other's files. Git conflicts are then limited to notebooks, so **use separate notebooks per track** (for example `a_embed_pool.ipynb` and `p_lstm.ipynb`). Do not both edit `project.ipynb`.
- Item 6 needs both tracks' trained models. A and P can build their feature code on the tutorial's models in the meantime.

### Suggested timeline

| Step | Who | What | Done when |
|---|---|---|---|
| 1 | Both | Tutorial T1–T8 (in parallel) | Both can explain every exercise |
| 2 | Both | Open decisions in section 3 (D2, D5, D8) | Section 3 filled in |
| 3 | A and P in parallel | Items 1–3 (A) and item 4 (P) | Results folders populated |
| 4 | Both | Run the final multi-seed experiments (D4 protocol) | Plots with uncertainty |
| 5 | A and P in parallel | Items 6a and 6b | BDA plots interpreted |
| 6 | Both | Item 7, mock oral Q&A on each other's parts | Blog done |

---

## 3. ⚖️ Decisions still to make (for A and P)

Fill in your pick and a one-line reason. If we agree, it's decided. If not, talk it through, and ask the TA if still stuck. Don't let a disagreement block the other person's work. "Suggested" is only a starting point.

| ID | Question | Options | Suggested | A's pick | P's pick |
|---|---|---|---|---|---|
| D2 | Prune the GloVe file to the IMDB vocabulary? | Yes (load the full file once, save a small `.npy` and word list) / No (load all 2.2M words in every job) | Yes: faster jobs, and cheap frozen-vs-trainable runs. About 30 min of work. | Yes: `coverage_imdb.py` already writes the pruned file (105,657 of 149,563 IMDB types, 127 MB, `<unk>` = full-vocabulary mean). Keep the full zip for item 2. | |
| D5 | LSTM vs GRU, uni- vs bidirectional | LSTM / GRU / bi-LSTM | **Park decides** (she implements it) | n/a | |
| D8 | Who writes which part of the blog? | By track (own items) / by section | By track, then cross-read | By track, then cross-read: each writes the sections for their own items. | |
| D11 | Which extra error-analysis features to add (decide at the end, after looking at errors)? | A adds input features (e.g. unknown-word rate, negation). P adds model features (e.g. probability of the predicted label). | Agree the final list after the first error analysis | | |

## 3b. Decisions already made

| ID | Decision | Outcome |
|---|---|---|
| D0 | Skip-gram training exercise (T2) | **We both do it**, fully. |
| D1 | Embedding model | **GloVe Common Crawl 300d** (`glove.840B.300d`): web text is closest to informal IMDB reviews. We still report coverage and sentiment-structure evidence. |
| D3 | Seeds, folds, epochs | **Tutorial values:** seeds `[0, 3, 5]`, 5 folds, 20 epochs, identical for both tracks. Time one LSTM fold on Snellius first. |
| D4 | Hyperparameter search | **One axis at a time** (1 seed, 5 folds) to screen, then confirm the best settings with 3 seeds. A full grid is too expensive, and the tutorial advises structuring the search. |
| D6 | Error-analysis features (baseline) | **The tutorial's set:** predicted label (one-hot) plus `lexicon_count`, `flesch_reading_ease`, `reading_time`, `flesch_kincaid_grade`. |
| D7 | Repo workflow | **Both push to `stuciaga/nlp1-student`.** `git pull` before every push, commit in small pieces, stay in your own files. |
| D9 | Layout of `solutions.py` | **Separate files** (`solutions_a.py`, `solutions_p.py`), combined into `solutions.py` at the end. |
| D10 | Dataset and task | **IMDB from Project 1.** Anything from `part1/` can be reused, and its results are the baseline for item 3. |

Summary: the foundations (embedding, dataset, seeds, repo, file layout) are settled, so both of us can start now. Open items are about time budget and write-up, not blockers.

---

## 4. Open questions for the TA

- Is the list of recommended pretrained embedding models posted yet? (`project.ipynb` says "we will post a list".)
- Are we allowed to use more than 5 folds, or to run the analysis with sklearn instead of the BDA model?
