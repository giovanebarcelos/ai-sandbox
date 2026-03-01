# GO1021-SalvarModeloCompleto
# Salvar modelo completo
model.save('my_model.keras')

# Carregar modelo completo
model = keras.models.load_model('my_model.keras')

# Usar imediatamente
predictions = model.predict(X_test)
