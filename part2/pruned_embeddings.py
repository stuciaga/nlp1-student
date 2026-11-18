"""
Loader for the pruned embedding files written by coverage_imdb.py (D2).

`PrunedEmbeddings` has the same interface the training code uses on `part2.clslib.PretrainedEmbeddings`
(`vectors`, `vocab`, `pad_token`, `unk_token`, `index`, `word`, ...), so it can be passed to
`part2.experiment.cross_validation` / `load` unchanged.

Layout, as in PretrainedEmbeddings: id 0 = <pad> (zeros), id 1 = <unk> (mean of the FULL GloVe vocabulary,
saved by coverage_imdb.py), then the pruned words (all lowercased). Use tokenizers with lowercase=True.
"""
from pathlib import Path

import numpy as np

from part2.clslib import Vocabulary


class PrunedEmbeddings:
    def __init__(self, folder, unk_token="<unk>", pad_token="<pad>"):
        folder = Path(folder)
        words = (folder / "glove840b_imdb_words.txt").read_text(encoding="utf-8").split("\n")
        words = [w for w in words if w]
        matrix = np.load(folder / "glove840b_imdb.npy").astype(np.float32)
        mean = np.load(folder / "glove840b_mean.npy").astype(np.float32)
        assert len(words) == matrix.shape[0], "word list and matrix are out of sync"

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.num_dims = matrix.shape[1]
        self.vocab = Vocabulary(init=[pad_token, unk_token], default=unk_token)
        for w in words:
            self.vocab.add(w)
        self.vocab.freeze()
        self.vectors = np.concatenate(
            [np.zeros((1, self.num_dims), dtype=np.float32), mean[None, :], matrix], axis=0
        )
        assert self.vectors.shape[0] == len(self.vocab), "Mismatch between vocab and vectors"

    def __contains__(self, word):
        return word.lower() in self.vocab

    def index(self, word):
        return self.vocab.index(word.lower())

    def word(self, i):
        return self.vocab.word(i)

    def __getitem__(self, i):
        return self.vectors[i]

    def embed_word(self, word):
        return self.vectors[self.index(word)]

    def embed_sequence(self, tokens):
        return [self.embed_word(t) for t in tokens]
