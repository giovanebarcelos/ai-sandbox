# GO1807-NãoRequerInstalaçãoUsaApenasBibliotecas
replay_buffer = deque(maxlen=100000)

# Durante treino, SAMPLE aleatório de mini-batch
for step in episode:
    s, a, r, s_next, done = step
    replay_buffer.append((s, a, r, s_next, done))

    # Treinar com batch aleatório (quebra correlação temporal)
    if len(replay_buffer) > batch_size:
        batch = random.sample(replay_buffer, batch_size=32)
        train_on_batch(batch)
