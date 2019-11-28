'''
Lag en modell som predikerer tilsvarende NOT-operatoren.
Visualiser resultatet etter optimalisering av modellen.
'''
import numpy as np
import tensorflow as tf
import pandas as pd
import os, models, data
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"     # Ignore GPU. comment to use tensorflow-gpu
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Ignore warnings

optimizer = 10
x_train, y_train = data.a_not()
model = models.NOT_model()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.compat.v1.train.GradientDescentOptimizer(optimizer).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.compat.v1.Session()

# Initialize tf.Variable objects
session.run(tf.compat.v1.global_variables_initializer())

runtimer = 0
while True:
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})
    if runtimer % 10000 == 0:
        W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
        print("(epoch = %s | opt = %s)   W = %s, b = %s, loss = %s" % (runtimer,optimizer,W, b, loss))

    runtimer +=1

session.close()
