# GO1810-NãoRequerInstalaçãoUsaApenasBibliotecas
# Treinamento
for episode in range(500):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for step in range(500):
        # ε-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = Q_network.predict(state, verbose=0)
            action = np.argmax(q_values[0])

        # Executar ação
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Armazenar em buffer
        replay_buffer.append((state, action, reward, next_state, done))
        total_reward += reward

        # Treinar se buffer tem dados suficientes
        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)

            states = np.vstack([x[0] for x in batch])
            actions = np.array([x[1] for x in batch])
            rewards = np.array([x[2] for x in batch])
            next_states = np.vstack([x[3] for x in batch])
            dones = np.array([x[4] for x in batch])

            # Calcular targets
            targets = Q_network.predict(states, verbose=0)
            q_next = Q_target.predict(next_states, verbose=0)

            for i in range(batch_size):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + gamma * np.max(q_next[i])

            # Treinar
            Q_network.fit(states, targets, epochs=1, verbose=0)

        state = next_state
        if done:
            break

    # Atualizar target network a cada 10 episódios
    if episode % 10 == 0:
        Q_target.set_weights(Q_network.get_weights())

    # Decair epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode}: Reward = {total_reward}, Epsilon = {epsilon:.3f}")

# CartPole resolvido quando média últimos 100 episódios ≥ 195
