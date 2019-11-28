'''
I stedet for MNIST datasettet, bruk Fashion MNIST. Lag en
passende modell. Hva accuracy klarer du å oppnå?

Epoch 5/5
60000/60000 [==============================] - 184s 3ms/sample - loss: 0.2145 - acc: 0.9229
10000/10000 [==============================] - 7s 663us/sample - loss: 0.2153 - acc: 0.9212
Test accuracy: 0.9212

Process returned 0 (0x0)	execution time : 1070.065 s
'''
import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Ignore warnings

tf.logging.set_verbosity(tf.logging.INFO)

(x_train_, y_train_), (x_test_, y_test_) = tf.keras.datasets.fashion_mnist.load_data()
x_train_=x_train_/255.0
x_test_=x_test_/255.0

model = keras.Sequential()
max_pool = l.MaxPooling2D((2,2),(2,2),padding='same')
model.add(l.Reshape(target_shape=(28,28,1),input_shape=(28, 28)))

model.add(l.Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(l.BatchNormalization())
model.add(max_pool)
model.add(l.Dropout(0.25))
# second CONV => RELU => CONV => RELU => POOL layer set
model.add(l.Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(l.BatchNormalization())
model.add(l.Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(l.BatchNormalization())
model.add(max_pool)
model.add(l.Dropout(0.25))
# first (and only) set of FC => RELU layers
model.add(l.Flatten())
model.add(l.Dense(512, activation='relu'))
model.add(l.BatchNormalization())
model.add(l.Dropout(0.5))
# softmax classifier
model.add(l.Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_, y_train_, epochs=5, batch_size=100)

test_loss, test_acc = model.evaluate(x_test_, y_test_)

print('Test accuracy:', test_acc)
