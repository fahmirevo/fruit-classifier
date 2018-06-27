import numpy as np


def data_iterator(batch_size=128):
    while True:
        file_idxs = np.arange(10)
        np.random.shuffle(file_idxs)

        for file_idx in file_idxs:
            X = np.load('dataset/X_' + str(file_idx) + '.npy')
            Y = np.load('dataset/Y_' + str(file_idx) + '.npy')

            chunk_idxs = np.arange(len(X))
            np.random.shuffle(chunk_idxs)

            chunks = np.array_split(chunk_idxs, (len(chunk_idxs) // batch_size) + 1)
            for chunk in chunks:
                yield X[chunk], Y[chunk]
