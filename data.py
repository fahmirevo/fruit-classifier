import numpy as np


def data_iterator(batch_size=128):
    X = np.load("dataset/X.npy")
    Y = np.load("dataset/Y.npy")

    idxs = np.arange(len(X))

    while True:
        np.random.shuffle(idxs)

        chunks = np.array_split(idxs, (len(idxs) // batch_size) + 1)

        for chunk in chunks:
            yield X[chunk], Y[chunk]
