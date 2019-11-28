import numpy as np

def a_not():
    x_train = np.mat([[1], [0], [0], [1], [0], [1], [1]])
    y_train = np.mat([[0], [1], [1], [0], [1], [0], [0]])
    return x_train, y_train

def b_nand():
    x_train = np.mat([[1, 1], [0, 1], [1, 0], [0, 0], [1, 0], [1,1], [0,1]])
    y_train = np.mat([[0], [1], [1], [1], [1], [0], [1]])
    return x_train, y_train

def c_xor():
    x_train = np.mat([[1, 1], [0, 1], [1, 0], [0, 0], [1, 0], [1,1], [0,1]])
    y_train = np.mat([[0], [1], [1], [0], [1], [0], [1]])
    return x_train, y_train
