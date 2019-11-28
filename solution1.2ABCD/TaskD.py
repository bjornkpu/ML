'''
Lag en modell med prediktoren f (x) = softmax(xW + b) som
klassifiserer handskrevne tall. Se mnist for eksempel lasting av
MNIST datasettet, og visning og lagring av en observasjon. Du
skal oppnå en nøyaktighet på 0.9 eller over. Lag 10 .png bilder
som viser W etter optimalisering.
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, models, data
os.environ["CUDA_VISIBLE_DEVICES"]="-1"     # Ignore GPU. comment to use tensorflow-gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Ignore warnings

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
(x_train_, y_train_), (x_test_, y_test_) = tf.keras.datasets.mnist.load_data()

x_train = x_train_.reshape(x_train_.shape[0], x_train_.shape[1]*x_train_.shape[2])
x_test = x_test_.reshape(x_test_.shape[0], x_test_.shape[1]*x_test_.shape[2])
y_train = np.zeros((y_train_.size, 10))
y_train[np.arange(y_train_.size), y_train_] = 1

x_train, x_test  = x_train.astype('float32'), x_test.astype('float32')

y_test = np.zeros((y_test_.size, 10))
y_test[np.arange(y_test_.size), y_test_] = 1

# Load model from file models.py
model = models.MNIST_model()
optimizer = 0.1

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.compat.v1.train.AdamOptimizer(optimizer).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.compat.v1.Session()

# Initialize tf.Variable objects
session.run(tf.compat.v1.global_variables_initializer())

runtimer = 0
while True:
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})
    if runtimer % 1 == 0:
        W1, W2, b1, b2, loss, accuracy = session.run([model.W1, model.W2, model.b1, model.b2, model.loss, model.accuracy], {model.x: x_train, model.y: y_train})
        if accuracy > 0.9:
            print("epoch = {} | opt = {} | accuracy = {:.3}% | loss = {}".format(runtimer, optimizer, accuracy*100, loss))
            print("\nDone!    Accuracy over 90% reached!\n")
            break
        print("epoch = {} | opt = {} | accuracy = {:.3} | loss = {}".format(runtimer,optimizer, accuracy*100, loss))
    runtimer +=1


session.close()

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(W1[:,i].reshape(28,28))
    plt.title("W: {}".format(i))
    plt.xticks([])
    plt.yticks([])

plt.show()
