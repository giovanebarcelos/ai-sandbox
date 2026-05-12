# GO1050-Scheduler
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# MÉTODO 1: Callback com função customizada
# Define a taxa de aprendizado em função da época: mantém o valor original nas
# primeiras 10 épocas e reduz 5% por época a partir daí, evitando oscilações tardias.
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.95

callback = LearningRateScheduler(scheduler)
model.fit(X_train, y_train, callbacks=[callback])

# MÉTODO 2: Schedule no Optimizer
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# MÉTODO 3: ReduceLROnPlateau (automático!) ⭐
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)
model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[reduce_lr])
