'''
Prøv selv og utvid modellen du laget i b). Hva accuracy klarer
du å oppnå?
Prøv eksempelvis å legge til ReLU og/eller Dropout

Epoch 5/5
60000/60000 [==============================] - 16s 273us/sample - loss: 0.0324 - acc: 0.9900
10000/10000 [==============================] - 1s 81us/sample - loss: 0.0373 - acc: 0.9880
Test accuracy: 0.988

Process returned 0 (0x0)	execution time : 87.522 s

'''
import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train_, y_train_), (x_test_, y_test_) = tf.keras.datasets.mnist.load_data()
x_train_=x_train_/255.0
x_test_=x_test_/255.0

max_pool = l.MaxPooling2D((2,2),(2,2),padding='same')

model = keras.Sequential([
    l.Reshape(target_shape=(1,28,28),
              input_shape=(28, 28)),
    l.Conv2D(32,
            5,
            padding='same',
            activation=tf.nn.relu),
    max_pool,
    l.Conv2D(64,
            5,
            padding='same',
            activation=tf.nn.relu),
    max_pool,
    l.Flatten(),
    l.Dense(1024, activation=tf.nn.relu),
    l.Dropout(0.4),
    l.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_, y_train_, epochs=5)

test_loss, test_acc = model.evaluate(x_test_, y_test_)

print('Test accuracy:', test_acc)
