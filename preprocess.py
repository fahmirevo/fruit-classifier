import os
import torch
import torchvision.transforms as t
import numpy as np
from PIL import Image
import pickle

n_classes = len(os.listdir('images'))

X = None
Y = None

classes = []

for i, fruit in enumerate(os.listdir('images')):
    classes.append(fruit)
    for picture in os.listdir('images/' + fruit):
        print(fruit + ' ' + picture)
        im = Image.open('images/' + fruit + '/' + picture)
        x = t.ToTensor()(im)
        x = x.view(1, 3, 100, 100)
        y = torch.zeros((1, n_classes))
        y[0, i] = 1

        if X is not None:
            X = torch.cat([X, x])
            Y = torch.cat([Y, y])
        else:
            X = x
            Y = y

np.save('dataset/X.npy', np.array(X))
np.save('dataset/Y.npy', np.array(Y))

with open('classes.txt', 'wb') as f:
    pickle.dump(classes, f)
