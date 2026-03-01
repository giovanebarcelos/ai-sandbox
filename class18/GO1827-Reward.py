# GO1827-Reward
reward = -distancia_objetivo          # Aproximar do destino
       - 10 * colisão                 # Evitar bater
       + 100 * chegou_objetivo        # Chegou!
       - 0.01 * tempo                 # Minimizar tempo
