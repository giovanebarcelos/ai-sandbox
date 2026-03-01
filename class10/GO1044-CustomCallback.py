# GO1044-CustomCallback
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            self.model.save(f'model_epoch_{epoch}.keras')
        if logs.get('loss') != logs.get('loss'):  # NaN check
            self.model.stop_training = True
