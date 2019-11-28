import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
title = 'Øving 1.1 C'
csv = 'day_head_circumference.csv'
W = 0.00303079
b = -0.20893255

fig, ax = plt.subplots()

# Observed/training input and output
x_train = np.mat(pd.read_csv(csv, header=0, dtype=object, usecols=[0]).values.astype(float))
y_train = np.mat(pd.read_csv(csv, header=0, dtype=object, usecols=[1]).values.astype(float))

ax.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
ax.set_xlabel('Head circ.')
ax.set_ylabel('Day')

class NonLinearRegressionModel:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        def sigma(z):
            return 1/(1+np.exp(-z))
        return 20*sigma(x*self.W + self.b)+31

    # Mean Squared Error
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))


model = NonLinearRegressionModel(np.mat([[W]]), np.mat([[b]]))

#x = np.arange(np.min(x_train),np.max(x_train)).reshape(int(np.max(x_train)-np.min(x_train)), 1)
x = np.linspace(np.min(x_train), np.max(x_train)).reshape(-1, 1)
ax.plot(x, model.f(x), label='$y = f(x) = xW+b$')
fig.canvas.set_window_title("%s     Loss = %s" % (title, model.loss(x_train, y_train)))
ax.legend()
plt.show()
