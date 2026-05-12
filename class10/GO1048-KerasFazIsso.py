# GO1048-KerasFazIsso
# Lembra que o Keras gerencia automaticamente o modo de inferência (training=False)
# ao chamar model.predict(), aplicando corretamente Dropout e BatchNormalization.
# Keras faz isso automaticamente!
model.predict(X_test)
