# GO1022-SalvarSóPesos
# Salvar só pesos (mais leve)
model.save_weights('model_weights.weights.h5')

# Carregar pesos (precisa criar arquitetura antes!)
model = create_model()  # Recriar arquitetura
model.load_weights('model_weights.weights.h5')
