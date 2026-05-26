"""
GO1811 - REINFORCE: Policy Gradient Monte Carlo
================================================
Demonstra o algoritmo REINFORCE (Williams, 1992) com exemplo executável.
Requer apenas numpy.

REINFORCE é o algoritmo de Policy Gradient mais simples:
1. Coletar episódio completo seguindo política π_θ(a|s)
2. Calcular retornos G_t para cada timestep
3. Atualizar política: θ ← θ + α ∇_θ log π_θ(a|s) * G_t

A política π_θ(a|s) é uma rede neural com saída softmax.
Update rule: maximizar log-probabilidade das ações boas (G_t > 0)
             e minimizar log-probabilidade das ações ruins (G_t < 0)

Diferença crítica vs. Q-Learning:
  Q-Learning: aprende VALORES (Q-table)
  REINFORCE:  aprende POLÍTICAS (parâmetros θ diretamente)
"""

import numpy as np


class PoliticaLinear:
    """
    Política π_θ(a|s) como rede linear + softmax.
    Para demonstração didática (sem TensorFlow).
    """

    def __init__(self, state_size: int, action_size: int):
        np.random.seed(42)
        self.W = np.random.randn(state_size, action_size) * 0.01
        self.action_size = action_size

    def predict_probs(self, state: np.ndarray) -> np.ndarray:
        """Retorna π_θ(a|s) — distribuição de probabilidade sobre ações."""
        logits = state @ self.W
        # Softmax: exp(logits) / Σexp(logits)
        logits -= np.max(logits)  # Estabilidade numérica
        exp = np.exp(logits)
        return exp / exp.sum()

    def sample_action(self, state: np.ndarray) -> tuple:
        """Amostra ação da política e retorna (acao, probs)."""
        probs = self.predict_probs(state)
        action = int(np.random.choice(self.action_size, p=probs))
        return action, probs

    def update(self, states: list, actions: list, returns: list,
               lr: float = 0.01) -> float:
        """
        Atualização REINFORCE:
        θ ← θ + α * ∇_θ log π_θ(a|s) * G_t

        Equivalente a: maximizar E[log π(a|s) * G_t]
        """
        total_loss = 0.0
        for s, a, G in zip(states, actions, returns):
            probs = self.predict_probs(s)

            # Gradiente do log π(a|s): (1_{a=i} - π(i|s)) para cada i
            grad = np.zeros(self.action_size)
            for i in range(self.action_size):
                if i == a:
                    grad[i] = 1 - probs[i]
                else:
                    grad[i] = -probs[i]

            # Atualizar pesos: gradient ASCENT ponderado por G_t
            self.W += lr * np.outer(s, grad * G)
            total_loss += -np.log(probs[a] + 1e-8) * G

        return total_loss / len(states)


def calcular_retornos(rewards: list, gamma: float) -> list:
    """
    Calcula retornos descontados G_t para cada timestep.
    G_t = R_t + γ R_{t+1} + γ² R_{t+2} + ...
    """
    G = 0.0
    retornos = []
    for r in reversed(rewards):
        G = r + gamma * G
        retornos.insert(0, G)
    return retornos


def treinar_reinforce(num_episodes: int = 300, gamma: float = 0.99) -> dict:
    """
    Treina política no ambiente CartPole simplificado 1D.
    """
    # Ambiente: equilíbrio de vara (simplificado 2D)
    # Estado: [angulo, velocidade_angular]
    state_size, action_size = 2, 2
    politica = PoliticaLinear(state_size, action_size)

    recompensas = []
    perdas = []

    for ep in range(num_episodes):
        # ── Coletar episódio completo ─────────────────────────
        # Estado inicial: ângulo e velocidade aleatórios
        angle = np.random.uniform(-0.1, 0.1)
        ang_vel = 0.0
        state = np.array([angle, ang_vel])

        states_ep, actions_ep, rewards_ep = [], [], []
        done = False
        t = 0

        while not done and t < 200:
            action, probs = politica.sample_action(state)

            # Dinâmica simplificada: empurrar esquerda (0) ou direita (1)
            force = 1.0 if action == 1 else -1.0
            ang_vel = ang_vel + 0.01 * force - 0.001 * np.sin(angle)
            angle = angle + ang_vel

            next_state = np.array([angle, ang_vel])

            reward = 1.0 if abs(angle) < 0.3 else 0.0
            done = abs(angle) > 0.5

            states_ep.append(state)
            actions_ep.append(action)
            rewards_ep.append(reward)

            state = next_state
            t += 1

        # ── Atualizar política (Monte Carlo — episódio completo) ──
        retornos = calcular_retornos(rewards_ep, gamma)

        # Normalizar retornos (estabilidade)
        retornos = np.array(retornos)
        retornos = (retornos - retornos.mean()) / (retornos.std() + 1e-8)

        loss = politica.update(states_ep, actions_ep, retornos.tolist(), lr=0.005)
        perdas.append(loss)
        recompensas.append(sum(rewards_ep))

    return {"politica": politica, "recompensas": recompensas, "perdas": perdas}


if __name__ == "__main__":
    print("=" * 60)
    print("GO1811 - REINFORCE: POLICY GRADIENT MONTE CARLO")
    print("=" * 60)

    print("\nCONCEITO:")
    print()
    print("  # Politica estocastica π_θ(a|s) — saida softmax")
    print("  policy_network = Sequential([..., Dense(action_size, softmax)])")
    print()
    print("  # Coletar episodio")
    print("  for step in episode:")
    print("      action_probs = policy_network(state)")
    print("      action = np.random.choice(action_size, p=action_probs)")
    print("      states, actions, rewards += step")
    print()
    print("  # Calcular retornos G_t = Σ γ^k r_{t+k}")
    print("  returns = []")
    print("  G = 0")
    print("  for r in reversed(rewards):")
    print("      G = r + gamma * G")
    print("      returns.insert(0, G)")
    print()
    print("  # Atualizar politica (gradient ascent)")
    print("  loss = -E[log π(a|s) * G_t]")

    print()
    print("─" * 60)
    print("EXECUTANDO REINFORCE (300 episodios, versao simplificada):")
    print("─" * 60)

    np.random.seed(7)
    resultado = treinar_reinforce(num_episodes=300)
    recompensas = resultado["recompensas"]

    print()
    print("  Progresso do aprendizado:")
    for bloco in range(6):
        inicio = bloco * 50
        fim = inicio + 50
        if fim <= len(recompensas):
            media = np.mean(recompensas[inicio:fim])
            barra = "#" * int(media / 3)
            print(f"  Ep {inicio+1:3d}-{fim:3d}: media={media:6.1f}  {barra}")

    print()
    print("  Vantagens do REINFORCE:")
    print("  + Aprende π(a|s) diretamente (sem Q-values)")
    print("  + Funciona para acoes discretas E continuas")
    print("  + Convergencia garantida para maximo local")
    print()
    print("  Desvantagens:")
    print("  - Alta variancia (requer muitos episodios)")
    print("  - Monte Carlo: precisa episodio completo (nao online)")
    print("  - Lento para problemas complexos")
    print()
    print("  Solucao: REINFORCE com Baseline (ver GO1824) ou A2C (ver GO1825)")
    print("           Baseline V(s) reduz variancia significativamente")
