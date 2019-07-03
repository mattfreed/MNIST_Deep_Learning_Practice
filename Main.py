import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random

np.random.seed(0)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
assert(X_train.shape[0]==y_train.shape[0]), "The number of images is not equal to the number of labels"
assert(X_test.shape[0]==y_test.shape[0]), "The number of images is not equal to the number of labels"
assert(X_train.shape[1:] == (28,28)), "Thew dimensions are not 28x28"
assert(X_test.shape[1:] == (28,28)), "Thew dimensions are not 28x28"

num_of_samples = []

cols = 5

num_classes = 10

fig, axs =plt.subplots(nrows = num_classes, ncols = cols, figsize = (10,10))
fig.tight_layout()
for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0,len(x_selected-1)),:, :], cmap = plt.get_cmap("gray"))
plt.show()
