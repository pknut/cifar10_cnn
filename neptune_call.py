from keras.callbacks import Callback
from deepsense import neptune

ctx = neptune.Context()

class NeptuneCallback(Callback):
    def __init__(self, x_test, y_test, images_per_epoch=-1):
        self.epoch_id = 0
        self.images_per_epoch = images_per_epoch
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        ctx.channel_send('Log-loss training', self.epoch_id, logs['loss'])
        ctx.channel_send('Log-loss validation', self.epoch_id, logs['val_loss'])
        ctx.channel_send('Accuracy training', self.epoch_id, logs['acc'])
        ctx.channel_send('Accuracy validation', self.epoch_id, logs['val_acc'])

