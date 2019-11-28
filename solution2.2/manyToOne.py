import numpy as np
import tensorflow as tf

batch_size = 1

ch_enc = np.eye(13,13)
encoding_size = np.shape(ch_enc)[1]

index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']
char_to_index = dict((char, i) for i, char in enumerate(index_to_char))

em_enc = np.eye(7,7)
encoding_size_emo = np.shape(em_enc)[1]

index_to_em = ['🎩', '🐀', '🐱', '🏢', '🙃', '🧢', '👦']
em_to_index = dict((em, i) for i, em in enumerate(index_to_em))

x_train = [[
    [ch_enc[1], ch_enc[2], ch_enc[3], ch_enc[0]], # 'hat '
    [ch_enc[4], ch_enc[2], ch_enc[3], ch_enc[0]], # 'rat '
    [ch_enc[5], ch_enc[2], ch_enc[3], ch_enc[0]], # 'cat '
    [ch_enc[6], ch_enc[7], ch_enc[2], ch_enc[3]], # 'flat'
    [ch_enc[8], ch_enc[2], ch_enc[3], ch_enc[3]], # 'matt'
    [ch_enc[5], ch_enc[2], ch_enc[9], ch_enc[0]], # 'cap '
    [ch_enc[10], ch_enc[11], ch_enc[12], ch_enc[0]], # 'son '
]]

y_train = np.mat([
    em_enc[0],
    em_enc[1],
    em_enc[2],
    em_enc[3],
    em_enc[4],
    em_enc[5],
    em_enc[6]
])

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(None, encoding_size)))
model.add(tf.keras.layers.Dense(encoding_size_emo, activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=optimizer)


def on_epoch_end(epoch, data):
    if epoch % 10 == 9:
        print("epoch", epoch)
        print("loss", data['loss'])

        # Generate text from the initial text ' h'
        text = ['ht', 'rats', 'atc', 'flap','mats','rap', 'sot', 'hat', 'rat', 'cat', 'flat', 'matt', 'cap', 'son']
        for i, ord in enumerate(text):
            x = np.zeros((1, 4, encoding_size))
            for t, char in enumerate(text[i]):
                x[0, t, char_to_index[char]] = 1
            y = model.predict(x)[-1]
            text[i] +=":  "+ index_to_em[y.argmax()]
            print(text[i])


model.fit(x_train, y_train, batch_size=batch_size, epochs=500, verbose=False, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)])
