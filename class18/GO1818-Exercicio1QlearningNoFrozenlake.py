"""
GO1818-Exercicio1 - Q-Learning no FrozenLake
=============================================
Exercício completo: implementar Q-Learning no FrozenLake e analisar resultados.
Requer: pip install gymnasium matplotlib

FrozenLake: grid 4x4 onde o agente deve ir do início (S) ao objetivo (G)
sem cair nos buracos (H). Versão is_slippery=False: determinístico.

Tarefa do exercício:
1. Criar Q-table: Q = np.zeros((n_states, n_actions))
2. Implementar loop de treinamento com ε-greedy
3. Update rule: Q[s,a] += alpha * (r + gamma * max Q[s'] - Q[s,a])
4. Treinar 10.000 episódios
5. Testar política aprendida
6. Comparar is_slippery=True vs False
"""

import sys
import subprocess
import numpy as np


def instalar_deps():
    for pkg in ["gymnasium", "matplotlib"]:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )


def treinar_q_learning_frozenlake(
    is_slippery: bool = False,
    num_episodes: int = 10000,
    alpha: float = 0.1,
    gamma: float = 0.99,
) -> dict:
    """
    Treina Q-Learning no FrozenLake.
    Retorna estatísticas e Q-table treinada.
    """
    import gymnasium as gym

    env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
    n_states = env.observation_space.n   # 16 (grid 4x4)
    n_actions = env.action_space.n       # 4 (esq, baixo, dir, cima)

    # 1. Inicializar Q-table com zeros
    Q = np.zeros((n_states, n_actions))

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9995

    recompensas = []
    sucessos_por_bloco = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 2. Politica ε-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            # 3. Executar ação
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 4. Update rule do Q-Learning
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        recompensas.append(total_reward)

        # Guardar taxa de sucesso a cada 1000 episódios
        if (episode + 1) % 1000 == 0:
            bloco = recompensas[-1000:]
            sucessos_por_bloco.append(sum(bloco) / len(bloco))

    env.close()

    return {
        "Q": Q,
        "recompensas": recompensas,
        "sucessos_por_bloco": sucessos_por_bloco,
        "n_states": n_states,
        "n_actions": n_actions,
        "is_slippery": is_slippery,
    }


def testar_politica(Q: np.ndarray, is_slippery: bool, n_testes: int = 100) -> float:
    """Testa a política aprendida e retorna taxa de sucesso."""
    import gymnasium as gym

    env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
    sucessos = 0
    for _ in range(n_testes):
        state, _ = env.reset()
        done = False
        while not done:
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        if reward > 0:
            sucessos += 1
    env.close()
    return sucessos / n_testes


def visualizar_q_table(Q: np.ndarray) -> None:
    """Exibe a Q-table como grid 4x4 com ações ótimas."""
    acoes_nomes = ['E', 'B', 'D', 'C']  # Esquerda, Baixo, Direita, Cima
    print("  Q-table (acao otima por estado):")
    print("  Grid 4x4 (S=inicio, H=buraco, F=livre, G=objetivo):")
    grid = "SFFHFFFHHFFFHFFG"  # FrozenLake padrão
    print()
    for i in range(4):
        linha_grid = "  "
        linha_acao = "  "
        for j in range(4):
            s = i * 4 + j
            celula = grid[s]
            melhor_a = acoes_nomes[int(np.argmax(Q[s]))]
            if celula in ("H", "G"):
                linha_acao += f"  {celula}  "
            else:
                linha_acao += f"  {melhor_a}  "
            linha_grid += f"  {celula}  "
        print(linha_grid)
        print(linha_acao)
    print()


def comparar_slippery():
    """Compara determinístico vs estocástico."""
    print()
    print("─" * 60)
    print("COMPARACAO: Determinístico vs Estocástico")
    print("─" * 60)

    for slippery in [False, True]:
        tipo = "ESTOCÁSTICO (is_slippery=True)" if slippery else "DETERMINÍSTICO (is_slippery=False)"
        print(f"\n  Treinando: {tipo}")

        resultado = treinar_q_learning_frozenlake(is_slippery=slippery, num_episodes=10000)

        taxa = testar_politica(resultado["Q"], slippery, n_testes=200)
        print(f"  Taxa de sucesso (200 testes): {taxa:.1%}")

        print(f"  Progresso (taxa por 1000 ep): "
              + " → ".join(f"{s:.0%}" for s in resultado["sucessos_por_bloco"]))

        if not slippery:
            visualizar_q_table(resultado["Q"])


def salvar_grafico(resultado_det, resultado_est):
    """Salva gráfico de comparação."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, res, titulo in zip(
            axes,
            [resultado_det, resultado_est],
            ["Determinístico (is_slippery=False)", "Estocástico (is_slippery=True)"],
        ):
            # Média móvel de 500 episódios
            recomp = res["recompensas"]
            n = 500
            media_movel = [np.mean(recomp[max(0, i-n):i+1]) for i in range(len(recomp))]
            ax.plot(media_movel, linewidth=1.5, color='steelblue')
            ax.set_xlabel("Episódio")
            ax.set_ylabel("Taxa de Sucesso (media 500 ep)")
            ax.set_title(titulo)
            ax.set_ylim(0, 1.1)
            ax.axhline(y=0.7, color='red', linestyle='--', label='Meta 70%')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle("Q-Learning no FrozenLake", fontsize=14)
        plt.tight_layout()
        plt.savefig("GO1818_Exercicio1_FrozenLake.png", dpi=120, bbox_inches='tight')
        print("\n  Grafico salvo: GO1818_Exercicio1_FrozenLake.png")
    except Exception as e:
        print(f"\n  Grafico nao salvo: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("EXERCICIO 1 - Q-LEARNING NO FROZENLAKE")
    print("=" * 60)

    instalar_deps()

    try:
        import gymnasium

        print("\nTreinando Q-Learning no FrozenLake...")

        # Treinamento determinístico
        print("\n1. FrozenLake DETERMINISTICO (is_slippery=False)")
        res_det = treinar_q_learning_frozenlake(is_slippery=False, num_episodes=10000)
        taxa_det = testar_politica(res_det["Q"], False, n_testes=200)
        print(f"   Taxa de sucesso: {taxa_det:.1%}  (meta: > 70%)")
        print(f"   Status: {'PASSOU!' if taxa_det > 0.7 else 'Precisa melhorar'}")
        visualizar_q_table(res_det["Q"])

        # Treinamento estocástico
        print("\n2. FrozenLake ESTOCASTICO (is_slippery=True)")
        res_est = treinar_q_learning_frozenlake(is_slippery=True, num_episodes=10000)
        taxa_est = testar_politica(res_est["Q"], True, n_testes=200)
        print(f"   Taxa de sucesso: {taxa_est:.1%}  (mais difícil!)")

        # Salvar gráfico
        salvar_grafico(res_det, res_est)

        print("\nRESUMO:")
        print(f"  Determinístico: {taxa_det:.1%} de sucesso")
        print(f"  Estocástico   : {taxa_est:.1%} de sucesso")
        print()
        print("  Por que é mais difícil com is_slippery=True?")
        print("  O agente escolhe acao A mas pode executar B ou C aleatoriamente.")
        print("  Q-Learning ainda converge mas precisa de mais episodios.")

    except ImportError:
        print("gymnasium nao instalado. Execute: pip install gymnasium")
