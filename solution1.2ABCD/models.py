import numpy as np
import tensorflow as tf


def sigmoid(t):
    return 1 / (1 + np.exp(-t))



'''
CALCULATE
'''
# A
class NOT_model:
    def __init__(self):
        # Model input
        self.x = tf.compat.v1.placeholder(tf.float32)
        self.y = tf.compat.v1.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        #logits
        logits=tf.matmul(self.x, self.W) + self.b
        # Predictor
        f = tf.nn.sigmoid(logits)

        # Mean Squared Error

        self.loss = tf.losses.sigmoid_cross_entropy(self.y, logits)

# B
class NAND_model:
    def __init__(self):
        # Model input
        self.x = tf.compat.v1.placeholder(tf.float32)
        self.y = tf.compat.v1.placeholder(tf.float32)

        # Model variables
        self.W1 = tf.Variable([[-10., 10.],[-10., 10.]])
        self.W2 = tf.Variable([[10.],[10.]])
        self.b1 = tf.Variable([[10., -10.]])
        self.b2 = tf.Variable([[0.0]])

        #logits
        logit1=tf.matmul(self.x, self.W1) + self.b1

        # Predictor
        f1 = tf.nn.sigmoid(logit1)
        logit2=tf.matmul(f1, self.W2) + self.b2
        f2 = tf.nn.sigmoid(logit2)

        # Sigmoid cross entropy
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, logit2)

# C
class XOR_model:
    def __init__(self):
        # Model input
        self.x = tf.compat.v1.placeholder(tf.float32)
        self.y = tf.compat.v1.placeholder(tf.float32)

        # Model variables
        self.W1 = tf.Variable([[-10., 10.],[-10., 10.]])
        self.W2 = tf.Variable([[10.],[10.]])
        self.b1 = tf.Variable([[10., -10.]])
        self.b2 = tf.Variable([[0.0]])

        #logits
        logit1=tf.matmul(self.x, self.W1) + self.b1

        # Predictor
        f1 = tf.nn.sigmoid(logit1)
        logit2=tf.matmul(f1, self.W2) + self.b2
        f2 = tf.nn.sigmoid(logit2)

        # Sigmoid cross entropy
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, logit2)

# D
class MNIST_model:
    def __init__(self):
        # Model input
        self.x = tf.compat.v1.placeholder(tf.float32, [None, 784])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, 10])

        # Model variables
        self.W1 = tf.Variable(tf.random.normal([784, 40]))
        self.b1 = tf.Variable(tf.zeros([40]))

        f1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.x, self.W1), self.b1))

        self.W2 = tf.Variable(tf.random.normal([40, 10]))
        self.b2 = tf.Variable(tf.zeros([10]))

        f2 = tf.nn.softmax(tf.nn.bias_add(tf.matmul(f1, self.W2), self.b2))

        # Loss
        self.loss = -tf.reduce_sum(self.y * tf.math.log(f2))

        # Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f2,1), tf.argmax(self.y,1)), tf.float32))


'''
VISUALIZE
'''
# A
class NOT_visualize:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return sigmoid(x * self.W + self.b)

    # Mean Squared Error
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))
# B
class NAND_visualize:
    def __init__(self, W1, W2, b1, b2):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

    #layer1
    def f1(self, x):
        return sigmoid(x * self.W1 + self.b1)
    #layer2
    def f2(self, h):
        return sigmoid(h * self.W2 + self.b2)

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    # Mean Squared Error
    def loss(self, x, y):
        return -np.mean(np.multiply(y,
            np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))

# C
class XOR_visualize:
    def __init__(self, W1, W2, b1, b2):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

    #layer1
    def f1(self, x):
        return sigmoid(x * self.W1 + self.b1)
    #layer2
    def f2(self, h):
        return sigmoid(h * self.W2 + self.b2)

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    # Mean Squared Error
    def loss(self, x, y):
        return -np.mean(np.multiply(y,
            np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))
# D
