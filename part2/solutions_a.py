"""
A's part of the missing `part2/solutions.py` (merged into solutions.py at the end, see WORKPLAN.md, D9).

Owner: A
Used by: tutorial cell calling `solutions.analyse_embs(trained_emb_model, pos_words, neg_words)`

`model` is a `part2.clslib.PretrainedEmbeddings`:
    model.vectors              np.ndarray [V, D]
    model.vocab / model.index(w) / model.word(i)
    model.embed_word(w)        np.ndarray [D]
"""


def analyse_embs(model, pos_words, neg_words):
    """
    Test whether the embedding space contains information about word polarity.

    model: PretrainedEmbeddings
    pos_words: list of str, positive words from the NLTK opinion lexicon (all in the model)
    neg_words: list of str, negative words from the NLTK opinion lexicon (all in the model)

    Suggested approach (see the hint in the tutorial):
      * dimensionality reduction (PCA / t-SNE) and a scatter plot coloured by polarity, and/or
      * k-means with 2 clusters, compared with the true polarity labels (e.g. purity or adjusted Rand index)
    If the clusters don't separate, that does not prove the information is absent. Say so in the write-up.
    """
    raise NotImplementedError("Implement me!")
