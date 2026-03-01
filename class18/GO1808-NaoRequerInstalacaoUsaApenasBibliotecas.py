# GO1808-NãoRequerInstalaçãoUsaApenasBibliotecas
Q_network = build_model()   # Rede principal
Q_target = build_model()    # Rede target
Q_target.set_weights(Q_network.get_weights())

# Calcular target usando rede fixa
y = r + gamma * max_a' Q_target(s', a')
loss = MSE(y, Q_network(s, a))

# Atualizar target network a cada C steps
if step % C == 0:
    Q_target.set_weights(Q_network.get_weights())
