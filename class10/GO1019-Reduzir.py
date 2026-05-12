# GO1019-Reduzir
# Lista de boas práticas de otimização: reduzir o learning rate, normalizar as
# entradas e usar inicialização adequada para estabilizar o treinamento.
optimizer=Adam(learning_rate=0.0001)  # Reduzir LR
X_train = X_train / 255.0             # Normalizar
kernel_initializer='he_normal'        # Boa inicialização
