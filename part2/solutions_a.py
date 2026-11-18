import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

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
    words = list(pos_words) + list(neg_words)
    X = np.stack([model.embed_word(w) for w in words])  # (N, D)
    y = np.array([0] * len(pos_words) + [1] * len(neg_words))  # 0 = positive, 1 = negative

    # 1) PCA scatter, coloured by true polarity
    Z = PCA(n_components=2).fit_transform(X)
    for lab, name in [(0, "positive"), (1, "negative")]:
        plt.scatter(Z[y == lab, 0], Z[y == lab, 1], s=6, alpha=0.5, label=name)
    plt.legend()
    plt.title("PCA of opinion-lexicon words")
    plt.show()

    # 2) k-means with 2 clusters, compared with true polarity
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    ari = adjusted_rand_score(y, km.labels_)
    agree = (km.labels_ == y).mean()
    purity = max(agree, 1 - agree)  # cluster ids are arbitrary, so check both labelings
    baseline = max(y.mean(), 1 - y.mean())  # always guessing the bigger class
    print(f"k-means: ARI={ari:.3f}, purity={purity:.3f}, majority baseline={baseline:.3f}")

    # 3) supervised probe: is polarity linearly decodable at all?
    acc = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5).mean()
    print(f"linear probe 5-fold accuracy: {acc:.3f}")
