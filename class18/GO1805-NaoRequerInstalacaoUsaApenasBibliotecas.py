# GO1805-NãoRequerInstalaçãoUsaApenasBibliotecas
Q = np.zeros((n_states, n_actions))
epsilon = 1.0
alpha = 0.1
gamma = 0.99
for episode in range(num_episodes):
    s = env.reset()
    done = False
    while not done:
        if np.random.random() < epsilon:
            a = np.random.choice(n_actions)
        else:
            a = np.argmax(Q[s])
        s_next, reward, done, info = env.step(a)
        Q[s,a] = Q[s,a] + alpha * (reward + gamma * np.max(Q[s_next]) - Q[s,a])
        s = s_next
    epsilon = max(0.01, epsilon * 0.995)
