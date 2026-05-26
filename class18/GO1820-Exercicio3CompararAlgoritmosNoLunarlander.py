"""
GO1820 - Exercício 3: Comparar Algoritmos no LunarLander
=========================================================
Compara DQN, PPO e A2C no ambiente LunarLander-v2.
Requer: pip install gymnasium stable-baselines3 matplotlib

LunarLander-v2:
  - Estado: 8 dimensões (posição, velocidade, ângulo, contato dos pés)
  - Ações: 4 (nada, motor esq, motor principal, motor dir)
  - Resolvido: média >= 200 por 100 episódios
  - Recompensa: +100/200 pouso suave, -100 crash

Stable-Baselines3 oferece implementações prontas e testadas de:
  - DQN: Deep Q-Network (Mnih et al. 2015)
  - PPO: Proximal Policy Optimization (Schulman et al. 2017)
  - A2C: Advantage Actor-Critic (Mnih et al. 2016)
"""

import sys
import subprocess
import numpy as np
import time


def instalar_deps():
    for pkg in ["gymnasium", "stable-baselines3", "matplotlib"]:
        try:
            pkg_import = pkg.replace("-", "_")
            __import__(pkg_import)
        except ImportError:
            print(f"Instalando {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )


def treinar_algoritmo(nome: str, timesteps: int = 100_000) -> dict:
    """Treina um algoritmo e retorna métricas."""
    from stable_baselines3 import PPO, A2C, DQN
    import gymnasium as gym

    algoritmos = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
    AlgoritmoClass = algoritmos[nome]

    env = gym.make("LunarLander-v2")
    inicio = time.time()

    model = AlgoritmoClass('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=timesteps)

    tempo = time.time() - inicio

    # Avaliar
    rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        total_r = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total_r += reward
        rewards.append(total_r)

    env.close()
    return {
        "nome": nome,
        "reward_medio": np.mean(rewards),
        "reward_std": np.std(rewards),
        "reward_max": max(rewards),
        "tempo_treino": tempo,
        "rewards": rewards,
        "model": model,
    }


def salvar_grafico(resultados: list) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Gráfico 1: Comparação de rewards
        ax1 = axes[0]
        nomes = [r["nome"] for r in resultados]
        medias = [r["reward_medio"] for r in resultados]
        stds = [r["reward_std"] for r in resultados]
        cores = ["steelblue", "tomato", "seagreen"][:len(nomes)]

        bars = ax1.bar(nomes, medias, color=cores, alpha=0.8, edgecolor='black')
        ax1.errorbar(nomes, medias, yerr=stds, fmt='none', color='black',
                     capsize=5, linewidth=2)
        ax1.axhline(y=200, color='gold', linestyle='--', linewidth=2, label='Meta (200)')
        ax1.set_ylabel("Recompensa Média (20 testes)")
        ax1.set_title("Comparação de Algoritmos — LunarLander-v2")
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        for bar, media in zip(bars, medias):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{media:.1f}', ha='center', va='bottom', fontweight='bold')

        # Gráfico 2: Boxplot das distribuições
        ax2 = axes[1]
        data = [r["rewards"] for r in resultados]
        ax2.boxplot(data, labels=nomes, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.axhline(y=200, color='red', linestyle='--', label='Meta (200)')
        ax2.set_ylabel("Recompensa por Episódio")
        ax2.set_title("Distribuição de Recompensas (20 episódios de teste)")
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig("GO1820_Exercicio3_Comparacao.png", dpi=120, bbox_inches='tight')
        print("  Grafico salvo: GO1820_Exercicio3_Comparacao.png")
    except Exception as e:
        print(f"  Grafico nao salvo: {e}")


def mostrar_comparacao_sem_sb3():
    """Mostra como a comparação funciona sem as dependências."""
    print("\nCOMO USAR stable-baselines3 (PPO, A2C, DQN):")
    print()
    print("  from stable_baselines3 import PPO, A2C, DQN")
    print("  import gymnasium as gym")
    print()
    print("  env = gym.make('LunarLander-v2')")
    print()
    print("  # Treinar PPO")
    print("  model_ppo = PPO('MlpPolicy', env, verbose=1)")
    print("  model_ppo.learn(total_timesteps=100_000)")
    print()
    print("  # Treinar A2C")
    print("  model_a2c = A2C('MlpPolicy', env, verbose=1)")
    print("  model_a2c.learn(total_timesteps=100_000)")
    print()
    print("  # Treinar DQN")
    print("  model_dqn = DQN('MlpPolicy', env, verbose=1)")
    print("  model_dqn.learn(total_timesteps=100_000)")
    print()
    print("  # Avaliar (greedy/determinístico)")
    print("  obs, _ = env.reset()")
    print("  action, _ = model_ppo.predict(obs, deterministic=True)")

    print()
    print("COMPARACAO ESPERADA (100k timesteps):")
    print()
    print(f"  {'Algoritmo':>10} | {'Reward Médio':>13} | {'Tempo':>10} | Notas")
    print("  " + "-" * 60)
    esperado = [
        ("PPO", 180, "~2 min",  "Melhor custo-benefício"),
        ("A2C",  90, "~1 min",  "Rápido mas menos estável"),
        ("DQN",  50, "~3 min",  "Mais amostras necessárias"),
    ]
    for nome, reward, tempo, nota in esperado:
        print(f"  {nome:>10} | {reward:>13} | {tempo:>10} | {nota}")


if __name__ == "__main__":
    print("=" * 60)
    print("EXERCICIO 3 - COMPARAR ALGORITMOS NO LUNARLANDER")
    print("=" * 60)

    instalar_deps()

    try:
        from stable_baselines3 import PPO, A2C, DQN
        import gymnasium

        algoritmos_treinar = ["PPO", "A2C", "DQN"]
        timesteps = 100_000

        print(f"\nTreinando {len(algoritmos_treinar)} algoritmos ({timesteps:,} timesteps cada)...")
        print("Isso pode levar 5-10 minutos dependendo do hardware.\n")

        resultados = []
        for nome in algoritmos_treinar:
            print(f"  Treinando {nome}...", end="", flush=True)
            res = treinar_algoritmo(nome, timesteps=timesteps)
            resultados.append(res)
            print(f" done. Reward médio: {res['reward_medio']:.1f} "
                  f"(±{res['reward_std']:.1f}), tempo: {res['tempo_treino']:.0f}s")

        # Tabela de resultados
        print()
        print("─" * 60)
        print("RESULTADOS FINAIS:")
        print("─" * 60)
        print(f"\n  {'Algoritmo':>10} | {'Reward Médio':>13} | {'Std':>7} | "
              f"{'Máximo':>8} | {'Tempo(s)':>9}")
        print("  " + "-" * 60)
        for r in sorted(resultados, key=lambda x: -x["reward_medio"]):
            print(f"  {r['nome']:>10} | {r['reward_medio']:>13.1f} | "
                  f"{r['reward_std']:>7.1f} | {r['reward_max']:>8.1f} | "
                  f"{r['tempo_treino']:>9.1f}")

        melhor = max(resultados, key=lambda x: x["reward_medio"])
        print(f"\n  Melhor algoritmo: {melhor['nome']} "
              f"(reward={melhor['reward_medio']:.1f})")

        salvar_grafico(resultados)

    except ImportError as e:
        print(f"Dependencia faltando: {e}")
        print("Execute: pip install stable-baselines3 gymnasium matplotlib")
        mostrar_comparacao_sem_sb3()
