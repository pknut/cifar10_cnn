"""
log-loss training   = 1.726
log-loss validation = 4.258
accuracy training   = 0.533
accuracy validation = 0.540
"""
import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, GaussianNoise, Dropout
from keras.layers import Conv2D, MaxPool2D, AlphaDropout, BatchNormalization
from deepsense import neptune
from neptune_call import NeptuneCallback

ctx = neptune.Context()
ctx.tags.append('cifar100')
ctx.tags.append('fine')
ctx.tags.append('fun')


num_classes = 100
epochs = 100

(x_train, y_train), (x_test, y_test) = cifar100.load_data('fine')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(AlphaDropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(AlphaDropout(0.5))
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
