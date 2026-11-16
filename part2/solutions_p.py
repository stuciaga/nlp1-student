"""
P's part of the missing `part2/solutions.py` (merged into solutions.py at the end, see WORKPLAN.md, D9).

Owner: P
Used by: the tutorial cells calling `solutions.demo_similarity`, `demo_topk` and `demo_analogy`.
`cos_similarity` and `topk_ids` are the notebook's stubs. Implement them here so A can reuse them
for project item 2 (embedding exploration).

`model` is a `part2.clslib.PretrainedEmbeddings`:
    model.vectors              np.ndarray [V, D]
    model.vocab / model.index(w) / model.word(i)
    model.embed_word(w)        np.ndarray [D]
"""
import numpy as np

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

    e_len = np.linalg.norm(E, axis = 1, keepdims = True)

    e_norm = E / e_len

    score = e_norm @ v_norm

    topk = np.argsort(score, axis = -1)[::-1][:k]

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
