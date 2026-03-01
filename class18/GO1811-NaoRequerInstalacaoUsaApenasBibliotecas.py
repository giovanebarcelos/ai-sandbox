# GO1811-NãoRequerInstalaçãoUsaApenasBibliotecas
# Política estocástica π_θ(a|s) parametrizada por θ
policy_network = Sequential([
    Dense(64, activation='relu', input_shape=(state_size,)),
    Dense(action_size, activation='softmax')  # Probabilidades
])

# Coletar episódio completo
states, actions, rewards = [], [], []
for step in episode:
    action_probs = policy_network(state)
    action = np.random.choice(action_size, p=action_probs)
    states.append(state)
    actions.append(action)
    rewards.append(reward)

# Calcular returns Gt = Σ γ^k r_{t+k}
returns = []
G = 0
for r in reversed(rewards):
    G = r + gamma * G
    returns.insert(0, G)

# Atualizar política (gradient ascent)
for s, a, G in zip(states, actions, returns):
    with tf.GradientTape() as tape:
        action_probs = policy_network(s)
        loss = -tf.math.log(action_probs[a]) * G  # Negativo para ascent
    grads = tape.gradient(loss, policy_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))
