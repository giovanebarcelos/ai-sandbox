# GO1021-SalvarModeloCompleto
# Demonstra como salvar o modelo completo (arquitetura + pesos + otimizador) no
# formato .keras e recarregá-lo para uso imediato em predições.
# Salvar modelo completo
model.save('my_model.keras')

# Carregar modelo completo
model = keras.models.load_model('my_model.keras')

# Usar imediatamente
predictions = model.predict(X_test)
