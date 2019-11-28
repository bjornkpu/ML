import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
title = 'Øving 1.1 A'
csv = 'length_weight.csv'
W = 0.23805696
b = -8.483095

fig, ax = plt.subplots()

# Observed/training input and output
x_train = np.mat(pd.read_csv(csv, header=0, dtype=object, usecols=[0]).values.astype(float))
y_train = np.mat(pd.read_csv(csv, header=0, dtype=object, usecols=[1]).values.astype(float))

ax.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
ax.set_xlabel('Length')
ax.set_ylabel('Weight')

class LinearRegressionModel:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return x * self.W + self.b

    # Mean Squared Error
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))


model = LinearRegressionModel(np.mat([[W]]), np.mat([[b]]))

x = np.mat([[np.min(x_train)], [np.max(x_train)]])
ax.plot(x, model.f(x), label='$y = f(x) = xW+b$')
fig.canvas.set_window_title("%s     Loss = %s" % (title, model.loss(x_train, y_train)))
ax.legend()
plt.show()
