"""
log-loss training   = 0.151
log-loss validation = 1.311
accuracy training   = 0.950
accuracy validation = 0.834
"""

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, GaussianNoise, Dropout
from keras.layers import Conv2D, MaxPool2D, AlphaDropout, BatchNormalization
from helpers import NeptuneCallback
from deepsense import neptune

ctx = neptune.Context()
ctx.tags.append('cifar10')
ctx.tags.append('fun')

num_classes = 10
epochs = 50

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (1, 1), activation='relu'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(GaussianNoise(0.3))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(AlphaDropout(0.25))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(AlphaDropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(AlphaDropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
ctx.channel_send('n_layers', len(model.layers))
ctx.channel_send('n_parameters', model.count_params())

model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=128,
          validation_data=(x_test, y_test),
          verbose=2,
          callbacks=[NeptuneCallback(x_test, y_test, images_per_epoch=20)])
