"""
Solutions to the tutorial exercises, imported by the tutorial as `from part2 import solutions`.

`model` is a `part2.clslib.PretrainedEmbeddings`:
    model.vectors              np.ndarray [V, D]
    model.vocab / model.index(w) / model.word(i)
    model.embed_word(w)        np.ndarray [D]
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import cross_val_score


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


def cos_similarity(u, v):
    """
    Return the cosine similarity (a real number) of two embedding vectors u and v.
    """
    dot = np.dot(u, v)

    u_len = np.linalg.norm(u)
    v_len = np.linalg.norm(v)

    result = dot / (u_len * v_len)

    return result


def topk_ids(v, E, k=10):
    """
    v: a D-dimensional vector
    E: embedding matrix, np.ndarray of shape [V, D]
    k: number of nearest neighbours

    Return a list with the ids of the k rows of E nearest to v by cosine similarity.
    Hint: normalise E and v, then one matrix-vector product, instead of looping over rows.
    """
    v_len = np.linalg.norm(v)

    v_norm = v / v_len

    e_len = np.linalg.norm(E, axis=1, keepdims=True)

    e_norm = E / e_len

    score = e_norm @ v_norm

    topk = np.argsort(score, axis=-1)[::-1][:k]

    return topk.tolist()


def demo_similarity(words, model):
    """
    Print the cosine similarity for pairs of the given words (e.g. person, human, place, city, car).
    Expected order from the tutorial: person-person > person-human > person-place > person-city > person-car.
    """
    raise NotImplementedError("Implement me!")


def demo_topk(words, k, model):
    """
    For each word, print its top-k nearest neighbours in the embedding space.
    Exclude the query word itself from its own neighbours.
    """
    raise NotImplementedError("Implement me!")


def demo_analogy(a, b, c, k, model):
    """
    Print the top-k words nearest to emb(a) - emb(b) + emb(c) (e.g. king - man + woman).
    Exclude the input words a, b and c from the results, otherwise they dominate.
    The tutorial calls this with (b, c) and also with (c, b), so make sure the order matters.
    """
    raise NotImplementedError("Implement me!")
