'''
Lag en ikke-lineær modell som predikerer hodeomkrets ut fra alder (i
dager) gitt observasjonene i day_head_circumference.csv
Bruk følgende modell prediktor: f (x) = 20σ(xW + b) + 31
'''
import numpy as np
import tensorflow as tf
import pandas as pd
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"     # Ignore GPU. comment to use tensorflow-gpu
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Ignore warnings

# Constants
optimizer = 0.0000000339
csv = 'day_head_circumference.csv'

# Load data
x_train = np.mat(pd.read_csv(csv, header=0, dtype=object, usecols=[0]).values.astype(float))
y_train = np.mat(pd.read_csv(csv, header=0, dtype=object, usecols=[1]).values.astype(float))

class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.compat.v1.placeholder(tf.float32)
        self.y = tf.compat.v1.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        def sigma(z):
            return 1/(1+tf.math.exp(-z))
        f = 20*sigma(tf.matmul(self.x, self.W) + self.b)+31

        # Mean Squared Error
        self.loss = tf.reduce_mean(tf.square(f - self.y))


print("Running...")

model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.compat.v1.train.GradientDescentOptimizer(optimizer).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.compat.v1.Session()

# Initialize tf.Variable objects
session.run(tf.compat.v1.global_variables_initializer())

runtimer = 0
lastLoss = 0
while True:
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})
    if runtimer % 10000 == 0:
        W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
        print("(epoch = %s | opt = %s)   W = %s, b = %s, loss = %s" % (runtimer,optimizer,W, b, loss))
        if lastLoss == loss:
            print("\nModel trained!\nEpoch: {:^10} W = {}, b = {}\nFinal Loss: {}".format(runtimer,W, b, loss))
            break
        else:
            lastLoss = loss
    runtimer +=1

session.close()
