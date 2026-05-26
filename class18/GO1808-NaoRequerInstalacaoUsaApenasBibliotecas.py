"""
GO1808 - Target Network: Estabilizando o DQN
=============================================
Demonstra o conceito de Target Network com exemplo executável.
Requer apenas numpy.

Problema sem Target Network: ao treinar Q_network, os targets
(r + γ max Q(s',a')) mudam a cada passo — a rede está "perseguindo
um alvo móvel". Isso causa instabilidade e divergência.

Solução (Mnih et al. 2015, Nature): Target Network
1. Manter duas redes: Q_network (principal) e Q_target (cópia)
2. Calcular targets usando Q_target (pesos FIXOS)
3. A cada C passos: Q_target.weights ← Q_network.weights

Intuição: como aprender a mover-se em direção a um alvo fixo,
não a um alvo que também se move.
"""

import numpy as np


class RedeSimulada:
    """
    Simula uma rede neural Q com pesos como arrays numpy.
    Representa Q(s,a) como uma função linear simples para demonstração.
    """

    def __init__(self, n_entradas: int = 4, n_saidas: int = 2, seed: int = None):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(n_entradas, n_saidas) * 0.1
        self.b = np.zeros(n_saidas)

    def predict(self, estado: np.ndarray) -> np.ndarray:
        """Forward pass: Q-values para todas as ações."""
        return estado @ self.W + self.b

    def get_weights(self) -> list:
        """Retorna cópia dos pesos."""
        return [self.W.copy(), self.b.copy()]

    def set_weights(self, weights: list) -> None:
        """Copia pesos de outra rede (hard update)."""
        self.W = weights[0].copy()
        self.b = weights[1].copy()

    def soft_update(self, source: 'RedeSimulada', tau: float = 0.01) -> None:
        """
        Soft update: θ_target ← τ·θ_network + (1-τ)·θ_target
        Alternativa ao hard update — atualização mais suave e estável.
        """
        self.W = tau * source.W + (1 - tau) * self.W
        self.b = tau * source.b + (1 - tau) * self.b


def calcular_loss_com_target(
    Q_network: RedeSimulada,
    Q_target: RedeSimulada,
    batch: list,
    gamma: float = 0.99,
) -> float:
    """
    Calcula MSE loss usando Q_target para os targets (pesos fixos).
    batch: lista de (estado, acao, recompensa, prox_estado, done)
    """
    total_loss = 0.0
    for estado, acao, recompensa, prox_estado, done in batch:
        # Target: usa Q_target (pesos fixos) — ESTÁVEL
        if done:
            target = recompensa
        else:
            target = recompensa + gamma * np.max(Q_target.predict(prox_estado))

        # Predição: usa Q_network (pesos atualizados) — APRENDE
        q_pred = Q_network.predict(estado)[acao]

        total_loss += (target - q_pred) ** 2

    return total_loss / len(batch)


if __name__ == "__main__":
    print("=" * 60)
    print("GO1808 - TARGET NETWORK: ESTABILIZANDO O DQN")
    print("=" * 60)

    print("\nCONCEITO:")
    print()
    print("  Q_network = build_model()    # Rede principal (aprende)")
    print("  Q_target  = build_model()    # Rede target (pesos fixos)")
    print("  Q_target.set_weights(Q_network.get_weights())")
    print()
    print("  # Calcular target usando rede FIXA")
    print("  y = r + gamma * max_a' Q_target(s', a')")
    print("  loss = MSE(y, Q_network(s, a))")
    print()
    print("  # Atualizar target a cada C passos (hard update)")
    print("  if step % C == 0:")
    print("      Q_target.set_weights(Q_network.get_weights())")

    # ─── Demonstração ─────────────────────────────────────────
    print()
    print("─" * 60)
    print("DEMONSTRACAO: Evolucao dos pesos e loss")
    print("─" * 60)

    np.random.seed(42)
    n_entradas, n_acoes = 4, 2
    C = 10  # Atualizar target a cada C passos

    Q_network = RedeSimulada(n_entradas, n_acoes, seed=1)
    Q_target = RedeSimulada(n_entradas, n_acoes, seed=1)

    print(f"\n  Pesos iniciais identicos (Q_target = copia de Q_network)")
    print(f"  C = {C} (atualizar target a cada {C} passos)")

    # Simular 30 passos de treinamento
    historico_loss = []
    for step in range(30):
        # Batch simulado
        batch = []
        for _ in range(8):
            s = np.random.randn(n_entradas) * 0.5
            a = np.random.choice(n_acoes)
            r = np.random.randn() * 0.5
            sp = s + np.random.randn(n_entradas) * 0.1
            done = np.random.random() < 0.1
            batch.append((s, a, r, sp, done))

        # Calcular loss com target network
        loss = calcular_loss_com_target(Q_network, Q_target, batch)
        historico_loss.append(loss)

        # Atualização dos pesos (simulada — descida de gradiente simplificada)
        lr = 0.01
        for s, a, r, sp, done in batch:
            target = r if done else r + 0.99 * np.max(Q_target.predict(sp))
            pred = Q_network.predict(s)[a]
            erro = target - pred
            Q_network.W[:, a] += lr * erro * s
            Q_network.b[a] += lr * erro

        # Hard update a cada C passos
        if (step + 1) % C == 0:
            Q_target.set_weights(Q_network.get_weights())
            print(f"\n  [Passo {step + 1:3d}] TARGET ATUALIZADO | loss={loss:.4f}")
        elif step % 5 == 0:
            print(f"  [Passo {step + 1:3d}]                    | loss={loss:.4f}")

    # ─── Soft vs Hard Update ──────────────────────────────────
    print()
    print("─" * 60)
    print("HARD UPDATE vs SOFT UPDATE:")
    print("─" * 60)

    Q_net = RedeSimulada(4, 2, seed=10)
    Q_tgt_hard = RedeSimulada(4, 2, seed=10)
    Q_tgt_soft = RedeSimulada(4, 2, seed=10)

    # Modificar Q_net
    Q_net.W += 1.0  # Simular aprendizado significativo

    # Hard update
    Q_tgt_hard.set_weights(Q_net.get_weights())

    # Soft update (tau=0.1)
    Q_tgt_soft.soft_update(Q_net, tau=0.1)

    diff_hard = np.mean(np.abs(Q_tgt_hard.W - Q_net.W))
    diff_soft = np.mean(np.abs(Q_tgt_soft.W - Q_net.W))

    print(f"\n  Após modificar Q_network (delta pesos = 1.0):")
    print(f"  Hard update (C=10): diferença = {diff_hard:.4f} (copia exata)")
    print(f"  Soft update (τ=0.1): diferença = {diff_soft:.4f} (10% do delta)")

    print()
    print("  Hard update: mais agressivo, atualiza completamente")
    print("  Soft update: mais suave, usado em SAC e TD3 (continuos)")
    print()
    print("  Ver GO1817 para DQN completo com Target Network no CartPole.")
