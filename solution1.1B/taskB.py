'''
Lag en lineer modell som predikerer alder (i dager) ut fra lengde og
vekt gitt observasjonene i day_length_weight.csv
'''
import numpy as np
import tensorflow as tf
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"     # Ignore GPU. comment to use tensorflow-gpu
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Ignore warnings

# Constants
optimizer = 0.00011586142973004287
#epocs = [10000]
csv = 'day_length_weight.csv'

# Load data
def read_csv(collumn):
    return np.mat(pd.read_csv(csv, header=0, dtype=object, usecols=[collumn]).values.astype(float))

x_train, y_train, z_train = read_csv(1), read_csv(0), read_csv(2)

class LinearRegressionModel3d:
    def __init__(self):
        # Model input
        self.x = tf.compat.v1.placeholder(tf.float32)
        self.y = tf.compat.v1.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        f = tf.matmul(self.x, self.W) + self.b

        # Mean Squared Error
        self.loss = tf.reduce_mean(tf.square(f - self.y))

print("Running...")
model = LinearRegressionModel3d()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.compat.v1.train.GradientDescentOptimizer(optimizer).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.compat.v1.Session()

# Initialize tf.Variable objects
session.run(tf.compat.v1.global_variables_initializer())

# Evaluate training accuracy

runtimer = 0
lastLoss = 0
while True:
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})
    if runtimer % 10000 == 0:
        W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
        print("(epoch = %s | opt = %s)   W = %s, b = %s, loss = %s" % (runtimer,optimizer,' '.join(map(str, W)), b, loss))
        if lastLoss == loss:
            print("\nModel trained!\nEpoch: {:^10} W = {}, b = {}\nFinal Loss: {}".format(runtimer,' '.join(map(str, W)), b, loss))
            break
        else:
            lastLoss = loss
    runtimer +=1

session.close()
