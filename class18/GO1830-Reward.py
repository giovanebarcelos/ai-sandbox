# GO1830-Reward
reward = + 1.0 * approach_target_speed    # Andar na velocidade certa
         - 10.0 * colisão                 # Evitar acidente!
         - 0.5 * jerk                     # Dirigir suavemente
         - 0.1 * mudar_faixa_desnecessário  # Não ficar trocando
         + 0.5 * ultrapassar_carro_lento  # Eficiência
