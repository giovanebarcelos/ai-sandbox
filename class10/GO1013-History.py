# GO1013-History
# Executa o treino do modelo com mini-batches de 128 amostras por até 50 épocas,
# usando o conjunto de validação e os callbacks para controle automático do processo.
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)
