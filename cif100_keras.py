"""
log-loss training   = 1.072
log-loss validation = 1.632
accuracy training   = 0.680
accuracy validation = 0.582
"""
import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, GaussianNoise, Dropout
from keras.layers import Conv2D, MaxPool2D, AlphaDropout, BatchNormalization
from keras.callbacks import EarlyStopping
from deepsense import neptune
from neptune_call import NeptuneCallback

ctx = neptune.Context()
ctx.tags.append('cifar100')
ctx.tags.append('fine')
ctx.tags.append('fun')


num_classes = 100
epochs = 200

(x_train, y_train), (x_test, y_test) = cifar100.load_data('fine')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.5))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

optimizer = keras.optimizers.Adam(lr=0.0001)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
ctx.channel_send('n_layers', len(model.layers))
ctx.channel_send('n_parameters', model.count_params())

early_stopping = EarlyStopping(patience=10)

model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=128,
          validation_data=(x_test, y_test),
          verbose=2,
          callbacks=[NeptuneCallback(x_test, y_test, images_per_epoch=20), early_stopping])
