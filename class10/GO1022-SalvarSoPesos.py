# GO1022-SalvarSóPesos
# Mostra como salvar apenas os pesos do modelo (arquivo mais leve) e como restaurá-los,
# exigindo que a arquitetura seja recriada manualmente antes de carregar os pesos.
# Salvar só pesos (mais leve)
model.save_weights('model_weights.weights.h5')

# Carregar pesos (precisa criar arquitetura antes!)
model = create_model()  # Recriar arquitetura
model.load_weights('model_weights.weights.h5')
