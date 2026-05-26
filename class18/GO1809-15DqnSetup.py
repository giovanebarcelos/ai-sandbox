"""
GO1809 - Configuração DQN para CartPole
========================================
Configura o ambiente e a rede neural para DQN no CartPole.
Usa gymnasium (substituto do gym descontinuado).

Instalação:
    pip install gymnasium tensorflow

Conceito: DQN (Deep Q-Network) usa uma rede neural para aproximar
Q(s,a) em espaços de estados contínuos onde uma Q-table seria inviável.
CartPole tem 4 dimensões de estado contínuo — impossível discretizar.

Arquitetura:
  Entrada: vetor de estado (4 dims no CartPole)
  Saídas:  Q-value para cada ação possível (2 no CartPole)
  Treinamento: MSE entre target e predição
"""

import sys
import subprocess


def instalar_dependencias():
    """Instala gymnasium e tensorflow se necessário."""
    for pkg in ["gymnasium", "tensorflow"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Instalando {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def setup_dqn_cartpole():
    """
    Configura o ambiente CartPole e a rede DQN.
    Retorna (env, Q_network, Q_target, replay_buffer, hyperparams).
    """
    try:
        import gymnasium as gym
        from tensorflow import keras
        from keras import layers
        from collections import deque

        # ── Ambiente ─────────────────────────────────────────
        env = gym.make('CartPole-v1')
        state_size = env.observation_space.shape[0]   # 4
        action_size = env.action_space.n               # 2

        # ── Arquitetura da rede ───────────────────────────────
        def build_model():
            model = keras.Sequential([
                # Camada de entrada: 4 features do CartPole
                layers.Dense(64, activation='relu', input_shape=(state_size,)),
                layers.Dense(64, activation='relu'),
                # Saída: Q-value para cada ação (sem ativação = linear)
                layers.Dense(action_size, activation='linear'),
            ])
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
            )
            return model

        Q_network = build_model()
        Q_target = build_model()
        # Inicializar target com mesmos pesos que a rede principal
        Q_target.set_weights(Q_network.get_weights())

        # ── Hyperparâmetros ───────────────────────────────────
        hyperparams = {
            "state_size": state_size,
            "action_size": action_size,
            "replay_buffer_maxlen": 10000,
            "batch_size": 32,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "target_update_freq": 10,  # Atualizar target a cada N episódios
        }

        replay_buffer = deque(maxlen=hyperparams["replay_buffer_maxlen"])

        return env, Q_network, Q_target, replay_buffer, hyperparams

    except ImportError as e:
        return None, None, None, None, str(e)


if __name__ == "__main__":
    print("=" * 60)
    print("GO1809 - CONFIGURACAO DQN PARA CARTPOLE")
    print("=" * 60)

    instalar_dependencias()

    env, Q_network, Q_target, replay_buffer, hyperparams = setup_dqn_cartpole()

    if env is None:
        print(f"\nErro ao configurar: {hyperparams}")
        print("Execute: pip install gymnasium tensorflow")
        sys.exit(1)

    print("\nAMBIENTE CARTPOLE:")
    print(f"  Espaco de observacao: {env.observation_space}")
    print(f"  Espaco de acao      : {env.action_space}")
    print()
    print("  Variaveis de estado (4):")
    print("    [0] posicao do carrinho   [-4.8, +4.8] m")
    print("    [1] velocidade do carrinho[-inf, +inf] m/s")
    print("    [2] angulo do polo        [-24°, +24°] rad")
    print("    [3] velocidade angular    [-inf, +inf] rad/s")
    print()
    print("  Acoes (2):")
    print("    0: empurrar para ESQUERDA")
    print("    1: empurrar para DIREITA")
    print()
    print("  Recompensa: +1 por cada passo que o polo fica em pe")
    print("  Termina:    polo > 15° OU carrinho fora de [-2.4, +2.4]")
    print("  Resolvido:  media >= 195 por 100 episodios")

    print()
    print("CONFIGURACAO DA REDE DQN:")
    Q_network.summary()

    print()
    print("HYPERPARAMETROS:")
    for k, v in hyperparams.items():
        print(f"  {k:25s}: {v}")

    print()
    print("VERIFICACAO: Q_network e Q_target tem pesos identicos?")
    import numpy as np
    pesos_identicos = all(
        np.allclose(w1, w2)
        for w1, w2 in zip(Q_network.get_weights(), Q_target.get_weights())
    )
    print(f"  Pesos identicos: {pesos_identicos}  <- CORRETO (Target = copia)")

    print()
    print("  Proximo passo: treinar o DQN (ver GO1817-26ProjetoDqnCartpoleSetup.py)")

    env.close()
