'''
Ta utgangspunkt i compute2.py i ntnu-tdat3025/cnn/mnist. Kjør
først dette eksempelet og se hva accuracy modellen oppnår.
a) Utvid modellen som vist nedenfor. Ca hva accuracy oppnår
denne modellen?

From github:
epoch 19
accuracy 0.9677

With keras:
Epoch 5/5
60000/60000 [==============================] - 11s 177us/sample - loss: 0.0729 - acc: 0.9787
10000/10000 [==============================] - 1s 66us/sample - loss: 0.0744 - acc: 0.9772
Test accuracy: 0.9772

Process returned 0 (0x0)	execution time : 56.786 s

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
            (5,5),
            strides=(1, 1),
            padding='same',
            input_shape=(28,28,1)),
    max_pool,
    l.Conv2D(64,
            (5,5),
            strides=(1, 1),
            padding='same'),
    max_pool,
    l.Flatten(),
    l.Dense(10,
            activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_, y_train_, epochs=5)

test_loss, test_acc = model.evaluate(x_test_, y_test_)

print('Test accuracy:', test_acc)
