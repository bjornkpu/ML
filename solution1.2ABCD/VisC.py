import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import models, data

# Constants
title = 'Øving 1.2 C - XOR'
W1 = [-9.91930103302002, 12.406449317932129], [-9.91930103302002, 12.406449317932129]
W2 = [[23.752224], [22.959755]]
b1 = [[14.904374122619629, -6.0293803215026855]]
b2 = [[-34.61781]]

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Observed/training input and output
x_train, y_train = data.c_xor()

ax1.plot(x_train[:,0].A.squeeze(),x_train[:,1].A.squeeze(),y_train[:,0].A.squeeze(), 'o',label='$(\\hat x^{(i)},\\hat y^{(i)})$')
ax2.plot(x_train[:,0].A.squeeze(),x_train[:,1].A.squeeze(),y_train[:,0].A.squeeze(), 'o',label='$(\\hat x^{(i)},\\hat y^{(i)})$')
ax1.set_xlabel('Input1')
ax2.set_xlabel('Input1')
ax1.set_ylabel('Input2')
ax2.set_ylabel('Input2')
ax1.set_zlabel('Output')
ax2.set_zlabel('Output')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_zticks([0, 1])
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_zticks([0, 1])

model = models.XOR_visualize(np.mat(W1), np.mat(W2), np.mat(b1), np.mat(b2))

# create x,y
x1_grid, x2_grid = np.meshgrid(np.linspace(np.min(x_train[:,0].A.squeeze()), np.max(x_train[:,0].A.squeeze()), 10), np.linspace(np.min(x_train[:,1].A.squeeze()), np.max(x_train[:,1].A.squeeze()), 10))
h1_grid=np.empty([10,10])
h2_grid=np.empty([10,10])
f2_grid=np.empty([10,10])
f_grid=np.empty([10,10])
for i in range(0,x1_grid.shape[0]):
    for j in range(0,x2_grid.shape[1]):
        h = model.f1([[x1_grid[i,j],x2_grid[i,j]]])
        h1_grid[i,j] = h[0,0]
        h2_grid[i,j] = h[0,1]
        f2_grid[i,j] = model.f2([[x1_grid[i,j],x2_grid[i,j]]])
        f_grid[i,j] = model.f([[x1_grid[i,j],x2_grid[i,j]]])


# plot the surface
ax1.plot_wireframe(x1_grid, x2_grid, h1_grid, alpha='0.6', color='lightgreen')
ax1.plot_wireframe(x1_grid, x2_grid, h2_grid, alpha='0.6', color='darkgreen')
ax2.plot_wireframe(x1_grid, x2_grid, f_grid, alpha='0.6', color='darkgreen')

fig.canvas.set_window_title("%s     Loss = %s" % (title, model.loss(x_train, y_train)))
ax1.legend()
plt.show()
