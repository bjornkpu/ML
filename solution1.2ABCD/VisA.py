import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import models, data

# Constants
title = 'Øving 1.2 A - NOT'
W = -23.045763
b = 11.176368
fig, ax = plt.subplots()

# Observed/training input and output
x_train, y_train = data.a_not()

ax.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
ax.set_xlabel('intup')
ax.set_ylabel('output')


model = models.NOT_visualize(np.mat([[W]]), np.mat([[b]]))
x = np.mat(np.linspace(np.min(x_train), np.max(x_train),20)).reshape(20,1)

ax.plot(x, model.f(x), label='$y = f(x) = xW+b$')
fig.canvas.set_window_title("%s     Loss = %s" % (title, model.loss(x_train, y_train)))
ax.legend()
plt.show()
