import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, art3d
import pandas as pd

# Constants
title = 'Øving 1.1 B'
csv = 'day_length_weight.csv'
Wx = [-2.24963]
Wy = [33.821503]
b = [[-2106.2793]]


fig = plt.figure()
ax = fig.gca(projection='3d')

# Observed/training input and output
dataset = np.array(pd.read_csv("day_length_weight.csv")).transpose()

x_train = np.expand_dims(dataset[2], 1)
y_train = np.expand_dims(dataset[1], 1)
z_train = np.expand_dims(dataset[0], 1)

ax.scatter(x_train, y_train, z_train, label='$y = f(x, z) = xW1 + zW2 + b$')
ax.set_xlabel('weight')
ax.set_ylabel('length')
ax.set_zlabel('age (days)')


class LinearRegressionModel:
    def __init__(self, W1, W2, b):
        self.W1 = W1
        self.W2 = W2
        self.b = b

    # Predictor
    def f(self, x, y):
        return x * self.W1 + y * self.W2 + self.b

    # Mean Squared Error
    def loss(self, x, y, z):
        return np.mean(np.square(self.f(x, y) - z))


model = LinearRegressionModel(Wx, Wy, b)

x = [ [np.min(x_train)], [np.min(x_train)], [np.max(x_train)], [np.max(x_train)] ]
y = [ [np.min(y_train)], [np.max(y_train)], [np.min(y_train)], [np.max(y_train)] ]
x, y = np.meshgrid(x, y)

z = model.f(x, y)

ax.plot_surface(x, y, z, alpha=0.3, color="orange")

fig.canvas.set_window_title("{}     Loss = {}".format(title, model.loss(x_train, y_train, z_train)))
ax.legend()
plt.show()
