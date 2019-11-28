'''
Lag en modell som predikerer tilsvarende NAND-operatoren.
Visualiser resultatet etter optimalisering av modellen.
'''
import numpy as np
import tensorflow as tf
import pandas as pd
import os, models, data
os.environ["CUDA_VISIBLE_DEVICES"]="-1"     # Ignore GPU. comment to use tensorflow-gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Ignore warnings

optimizer = 1
x_train, y_train = data.b_nand()
model = models.NAND_model()

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
        W1, W2, b1, b2, loss = session.run([model.W1, model.W2, model.b1, model.b2, model.loss], {model.x: x_train, model.y: y_train})
        print("(epoch = {} | opt = {})".format(runtimer,optimizer))
        print("W1 = [{}, {}], [{}, {}]\nW2 = [{}, {}]".format(W1[0,0],W1[0,1],W1[1,0],W1[1,1],W2[0],W2[1]))
        print("b1 = [[{}, {}]]\nb2 = {}\nloss = {}\n".format(b1[0,0],b1[0,1], b2, loss))
    runtimer +=1

session.close()
