import os
import torch
import torchvision.transforms as t
import numpy as np
from PIL import Image
import pickle

data_size = 3917
n_classes = 8

n_classes = len(os.listdir('images'))

X = torch.zeros((data_size, 3, 100, 100))
Y = torch.zeros((data_size, n_classes))

classes = []

count = 0
for i, fruit in enumerate(os.listdir('images')):
    classes.append(fruit)
    for picture in os.listdir('images/' + fruit):
        print(fruit + ' ' + picture)
        im = Image.open('images/' + fruit + '/' + picture)
        x = t.ToTensor()(im)

        X[count] = x
        Y[count, i] = 1
        count += 1
        # x = x.view(1, 3, 100, 100)
        # y = torch.zeros((1, n_classes))
        # y[0, i] = 1

        # if X is not None:
        #     X = torch.cat([X, x])
        #     Y = torch.cat([Y, y])
        # else:
        #     X = x
        #     Y = y

X = np.array(X)
Y = np.array(Y)

idxs = np.arange(len(X))
np.random.shuffle(idxs)
for i, chunk in enumerate(np.array_split(idxs, 10)):
    np.save('dataset/X_' + str(i), X[chunk])
    np.save('dataset/Y_' + str(i), Y[chunk])

with open('classes.txt', 'wb') as f:
    pickle.dump(classes, f)
