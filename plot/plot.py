import os
import sys

sys.path.append(os.path.join(os.getcwd(), os.pardir))

from datasets import digits, phoneme
from sklearn import decomposition, cluster, metrics, neural_network, model_selection
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def vector_distance(x, y):
    d = [y[i] - x[i] for i in range(len(x))]
    s = 0
    for i in range(len(d)):
        s += d[i]*d[i]
    return sqrt(s)

def image_from_features(features):
    image = [[features[y * 8 + x] for x in range(7)] for y in range(8)]
    return image

DATASET = phoneme
AXIS = [0, 1, 2]

# General Options
TITLE = 'Neural Network Classifier'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(DATASET.training_features)):
    features = DATASET.training_features[i]
    label = DATASET.training_labels[i]
    x = features[AXIS[0]]
    y = features[AXIS[1]]
    z = features[AXIS[2]]
    c = 'b' if str(label) == '1' else 'r'

    ax.scatter(x, y, z, c=c, marker='o')
plt.show()