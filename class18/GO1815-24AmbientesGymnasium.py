"""
GO1815 - Ambientes Gymnasium: Visão Geral e Demo
================================================
Demonstra os principais ambientes do Gymnasium com simulação de interação.
Com Gymnasium instalado: roda o ambiente real.
Sem Gymnasium: mostra informações detalhadas de cada ambiente.

Instalação:
    pip install gymnasium
    pip install gymnasium[atari] ale-py  # Para Atari

Conceito: Gymnasium (substituto do OpenAI Gym descontinuado) é a biblioteca
padrão de ambientes para RL em Python. Todos os ambientes seguem a
mesma interface: reset() → step(action) → (obs, reward, term, trunc, info)
"""

import sys
import subprocess


def instalar_gymnasium():
    try:
        import gymnasium
        return True
    except ImportError:
        print("Instalando gymnasium...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gymnasium"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True


def descrever_ambientes() -> list:
    """Retorna descrições dos principais ambientes Gymnasium."""
    return [
        {
            "id": "CartPole-v1",
            "nome": "CartPole (Polo no Carrinho)",
            "objetivo": "Equilibrar vara em carrinho empurrando esq/dir",
            "obs": "Box(4,) — [pos_carrinho, vel_carrinho, angulo, vel_angular]",
            "acoes": "Discrete(2) — 0=esquerda, 1=direita",
            "recompensa": "+1 por cada passo com vara em pé",
            "resolvido": "Media >= 195 por 100 episódios",
            "dificuldade": "Fácil",
        },
        {
            "id": "MountainCar-v0",
            "nome": "Mountain Car (Carro na Montanha)",
            "objetivo": "Carro usa momentum para subir a montanha",
            "obs": "Box(2,) — [posicao, velocidade]",
            "acoes": "Discrete(3) — 0=esq, 1=nada, 2=dir",
            "recompensa": "-1 por passo (incentiva subir rápido)",
            "resolvido": "Chegar ao topo em <= 110 passos",
            "dificuldade": "Moderado",
        },
        {
            "id": "LunarLander-v2",
            "nome": "Lunar Lander (Pouso Lunar)",
            "objetivo": "Pousar nave espacial entre as bandeiras",
            "obs": "Box(8,) — [x, y, vel_x, vel_y, angle, vel_ang, leg_l, leg_r]",
            "acoes": "Discrete(4) — 0=nada, 1=motor_esq, 2=motor_ppal, 3=motor_dir",
            "recompensa": "±200 pouso, -100 crash, -0.3/frame motor",
            "resolvido": "Media >= 200 por 100 episódios",
            "dificuldade": "Difícil",
        },
        {
            "id": "ALE/Breakout-v5",
            "nome": "Atari Breakout",
            "objetivo": "Rebater bola para destruir tijolos",
            "obs": "Box(210,160,3) — pixels RGB",
            "acoes": "Discrete(4) — acoes de joystick",
            "recompensa": "+1 por tijolo destruído",
            "resolvido": "Score > 30 por episódio",
            "dificuldade": "Muito difícil (requer CNN)",
        },
    ]


def demonstrar_interface_gymnasium():
    """
    Demonstra a interface padrão do Gymnasium com CartPole.
    """
    try:
        import gymnasium as gym
        import numpy as np

        print("\n  Executando ambiente REAL CartPole-v1:")
        # Sem render_mode para rodar sem display
        env = gym.make('CartPole-v1')

        # Reset: retorna (observacao, info)
        obs, info = env.reset(seed=42)
        print(f"  env.reset() → obs shape={obs.shape}")
        print(f"  obs = {obs.round(3)}")
        print(f"  Obs: [pos={obs[0]:.3f}, vel={obs[1]:.3f}, "
              f"ang={obs[2]:.3f}, ang_vel={obs[3]:.3f}]")

        print(f"\n  Espaco de observacao: {env.observation_space}")
        print(f"  Espaco de acao      : {env.action_space}")

        # Simular 5 passos com ação aleatória
        print("\n  5 passos com politica aleatoria:")
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            print(f"    passo {step+1}: action={action}, reward={reward:.1f}, "
                  f"done={done}, obs={obs.round(2)}")
            if done:
                break

        print(f"\n  Total reward: {total_reward:.1f}")
        env.close()

    except ImportError:
        print("\n  gymnasium nao instalado.")
        print("  Execute: pip install gymnasium")


def demo_sem_gymnasium():
    """Simulação da interface para demonstração sem a biblioteca."""
    import numpy as np

    print("\n  SIMULACAO da interface Gymnasium (sem instalar):")
    print()

    # Simular CartPole
    np.random.seed(42)
    obs = np.array([0.04, -0.01, 0.02, -0.03])
    print(f"  env.reset() → obs = {obs}")

    for step in range(5):
        action = np.random.choice([0, 1])
        # Dinâmica simplificada
        obs = obs + np.random.randn(4) * 0.05
        reward = 1.0
        done = abs(obs[2]) > 0.3  # termina se angulo > 17°
        print(f"  step {step+1}: action={action}, reward={reward}, "
              f"done={done}, obs={obs.round(3)}")
        if done:
            break


if __name__ == "__main__":
    print("=" * 60)
    print("GO1815 - AMBIENTES GYMNASIUM")
    print("=" * 60)

    # ─── Descrição dos ambientes ──────────────────────────────
    print("\nPRINCIPAIS AMBIENTES:")
    print("─" * 60)

    ambientes = descrever_ambientes()
    for amb in ambientes:
        print(f"\n  [{amb['id']}]  {amb['nome']}")
        print(f"  Objetivo   : {amb['objetivo']}")
        print(f"  Observacao : {amb['obs']}")
        print(f"  Acoes      : {amb['acoes']}")
        print(f"  Recompensa : {amb['recompensa']}")
        print(f"  Dificuldade: {amb['dificuldade']}")

    # ─── Interface padrão ─────────────────────────────────────
    print()
    print("─" * 60)
    print("INTERFACE PADRAO GYMNASIUM:")
    print("─" * 60)
    print()
    print("  env = gym.make('CartPole-v1')")
    print("  obs, info = env.reset()       # Iniciar episodio")
    print("  while not done:")
    print("      action = policy(obs)       # Escolher acao")
    print("      obs, reward, term, trunc, info = env.step(action)")
    print("      done = term or trunc")
    print("  env.close()")

    # ─── Tentar executar ambiente real ────────────────────────
    try:
        import gymnasium
        demonstrar_interface_gymnasium()
    except ImportError:
        demo_sem_gymnasium()

    print()
    print("─" * 60)
    print("NOTA: gymnasium substituiu gym (descontinuado em 2023)")
    print("─" * 60)
    print("  gym.make(...)                 # ANTIGO (deprecado)")
    print("  gymnasium.make(...)           # CORRETO (atual)")
    print()
    print("  Migracoes necessarias:")
    print("  - import gym → import gymnasium as gym")
    print("  - env.step() retorna 5 valores (adicionado 'truncated')")
    print("  - env.reset() retorna (obs, info)")
