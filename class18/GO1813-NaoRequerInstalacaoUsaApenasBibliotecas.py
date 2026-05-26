"""
GO1813 - Actor-Critic: Atualizações do Actor e Critic
======================================================
Demonstra as regras de atualização do Actor-Critic com exemplos numéricos.
Requer apenas numpy.

Actor-Critic combina dois componentes:
1. ACTOR (π_θ): aprende QUAL ação tomar — gradient ascent
   ∇_θ J = E[∇_θ log π_θ(a|s) · A(s,a)]

2. CRITIC (V_ϕ): aprende QUANTO vale cada estado — minimizar MSE
   Loss_critic = MSE(V_ϕ(s), retorno_real)

O Critic fornece a Advantage para o Actor, reduzindo a variância.
Isso é mais estável que REINFORCE puro (sem baseline).

Tipos de Actor-Critic:
  - A2C (Advantage Actor-Critic): TD advantage, single actor
  - A3C (Asynchronous): múltiplos workers paralelos
  - PPO (Proximal Policy Optimization): clipping de gradiente
  - SAC (Soft Actor-Critic): para ações contínuas
"""

import numpy as np


def update_actor_manual(
    log_prob_a: float,
    advantage: float,
    theta_antes: float,
    lr_actor: float = 0.01,
) -> tuple:
    """
    Simula uma atualização do Actor com gradient ascent.
    θ ← θ + α * ∇_θ log π_θ(a|s) * A(s,a)

    log_prob_a: log π_θ(a|s) — log-prob da ação tomada
    advantage : A(s,a) — vantagem estimada pelo Critic
    theta_antes: valor de θ antes da atualização (escalar para simplificar)
    """
    # Gradiente do log π (simplificado como escalar para demonstração)
    grad_log_pi = log_prob_a  # Em 1D: d/dθ log π ≈ log_prob
    grad_j = grad_log_pi * advantage  # ∇J = ∇log π * A

    # Ascent: maximizar J
    theta_novo = theta_antes + lr_actor * grad_j

    return theta_novo, grad_j


def update_critic_manual(
    v_pred: float,
    retorno_real: float,
    phi_antes: float,
    lr_critic: float = 0.05,
) -> tuple:
    """
    Simula uma atualização do Critic com gradient descent.
    ϕ ← ϕ - α_c * ∇_ϕ MSE(V_ϕ(s), G_t)

    v_pred      : V_ϕ(s) — valor predito pelo Critic
    retorno_real: G_t — retorno real (ou TD target)
    phi_antes   : valor de ϕ antes (escalar para simplificar)
    """
    td_error = retorno_real - v_pred          # Erro de predição
    loss = td_error ** 2                       # MSE

    # Gradiente: ∂MSE/∂V_pred = -2 * td_error
    grad_critic = -2 * td_error

    # Descent: minimizar MSE
    phi_novo = phi_antes - lr_critic * grad_critic  # subtrair gradiente

    return phi_novo, td_error, loss


def simular_treinamento_a2c(num_steps: int = 20, gamma: float = 0.99) -> list:
    """
    Simula vários passos de treinamento A2C.
    Demonstra a evolução das atualizações Actor e Critic.
    """
    np.random.seed(42)

    # Parâmetros escalares simplificados (em produção: pesos de rede)
    theta = 0.0    # Parâmetro do Actor
    phi = 0.0      # Parâmetro do Critic

    historico = []
    for step in range(num_steps):
        # Simular observação de um step
        reward = np.random.choice([-1.0, +1.0, +10.0], p=[0.6, 0.3, 0.1])
        v_pred = phi + 0.1 * np.random.randn()   # V_ϕ(s) com ruído
        v_prox = v_pred + 0.05 * np.random.randn()   # V(s') estimado

        # TD target e Advantage
        td_target = reward + gamma * v_prox
        advantage = td_target - v_pred

        # Log-prob da ação (simplificado)
        log_prob = -0.5 * np.random.rand()

        # Atualizar Critic primeiro
        phi_novo, td_error, critic_loss = update_critic_manual(
            v_pred, td_target, phi, lr_critic=0.05
        )

        # Atualizar Actor
        theta_novo, grad_j = update_actor_manual(
            log_prob, advantage, theta, lr_actor=0.01
        )

        historico.append({
            "step": step + 1,
            "reward": reward,
            "advantage": advantage,
            "critic_loss": critic_loss,
            "delta_actor": theta_novo - theta,
            "delta_critic": phi_novo - phi,
        })

        theta = theta_novo
        phi = phi_novo

    return historico


if __name__ == "__main__":
    print("=" * 60)
    print("GO1813 - ACTOR-CRITIC: ATUALIZACOES")
    print("=" * 60)

    print("\nFORMULAS:")
    print()
    print("  ACTOR (gradient ascent — maximizar J):")
    print("  ∇_θ J = E[∇_θ log π_θ(a|s) · A(s,a)]")
    print("  θ ← θ + α_actor · ∇_θ log π_θ(a|s) · A(s,a)")
    print()
    print("  CRITIC (gradient descent — minimizar MSE):")
    print("  Loss = MSE(V_ϕ(s), retorno_real)")
    print("  ϕ ← ϕ - α_critic · ∂Loss/∂ϕ")

    # ─── Exemplo numérico manual ─────────────────────────────
    print()
    print("─" * 60)
    print("EXEMPLO NUMERICO: um passo de Actor-Critic")
    print("─" * 60)

    print("\n  Situação:")
    print("  s: estado atual, V(s)=3.0, V(s')=2.5")
    print("  a: acao tomada, log π(a|s)=-0.5")
    print("  r: recompensa = +5.0")
    print("  gamma = 0.99")

    gamma = 0.99
    V_s, V_sp, reward = 3.0, 2.5, 5.0
    log_prob = -0.5

    td_target = reward + gamma * V_sp
    advantage = td_target - V_s

    print()
    print(f"  TD target = r + γ*V(s') = {reward} + {gamma}*{V_sp} = {td_target:.3f}")
    print(f"  Advantage = TD_target - V(s) = {td_target:.3f} - {V_s} = {advantage:.3f}")
    print(f"  A(s,a) > 0 → acao foi MELHOR que a media → REFORCAR")

    # Atualizar Actor
    theta_init = 0.5
    theta_novo, grad_j = update_actor_manual(log_prob, advantage, theta_init, lr_actor=0.01)
    print()
    print("  UPDATE ACTOR:")
    print(f"  θ ← θ + α * ∇log π * A(s,a)")
    print(f"  θ ← {theta_init} + 0.01 * {log_prob} * {advantage:.3f}")
    print(f"  θ ← {theta_novo:.5f}  (δ = {theta_novo - theta_init:+.5f})")

    # Atualizar Critic
    phi_init = 3.0
    phi_novo, td_error, critic_loss = update_critic_manual(V_s, td_target, phi_init, lr_critic=0.05)
    print()
    print("  UPDATE CRITIC:")
    print(f"  Loss = MSE(V(s), TD_target) = ({V_s} - {td_target:.3f})² = {critic_loss:.4f}")
    print(f"  ϕ ← ϕ - α_c * ∂Loss/∂ϕ")
    print(f"  ϕ ← {phi_novo:.5f}  (convergindo para TD_target={td_target:.3f})")

    # ─── Simulação de múltiplos passos ────────────────────────
    print()
    print("─" * 60)
    print("SIMULACAO: 20 PASSOS DE TREINAMENTO A2C")
    print("─" * 60)

    historico = simular_treinamento_a2c(num_steps=20)
    print(f"\n  {'Step':>5} | {'Reward':>7} | {'Advantage':>10} | "
          f"{'Critic Loss':>12} | {'ΔActor':>10}")
    print("  " + "-" * 60)
    for h in historico[::4]:  # Mostrar 1 em cada 4
        print(f"  {h['step']:>5} | {h['reward']:>7.1f} | {h['advantage']:>10.4f} | "
              f"{h['critic_loss']:>12.4f} | {h['delta_actor']:>+10.6f}")

    print()
    print("─" * 60)
    print("ACTOR vs CRITIC — PAPEIS NO APRENDIZADO:")
    print("─" * 60)
    papeis = [
        ("Actor  (θ)", "Aprende a POLITICA π(a|s): qual acao tomar"),
        ("Critic (ϕ)", "Aprende o VALOR V(s): quanto vale cada estado"),
        ("Advantage  ", "CRITICA do Critic para o Actor: foi bom ou ruim?"),
        ("Iteracao   ", "Actor e Critic se melhoram mutuamente"),
    ]
    for nome, desc in papeis:
        print(f"  {nome}: {desc}")
    print()
    print("  Ver GO1825-Codigo.py para implementacao completa com TensorFlow.")
