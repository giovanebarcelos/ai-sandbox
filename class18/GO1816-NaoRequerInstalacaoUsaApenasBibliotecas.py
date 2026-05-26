"""
GO1816 - Interface de Interação com Ambientes Gymnasium
========================================================
Explica e demonstra cada método da API Gymnasium: reset, step, render, close.
Requer apenas numpy (sem gymnasium para demonstração básica).

A API Gymnasium é o "contrato" entre agentes e ambientes:
  reset()  → inicia novo episódio, retorna (obs, info)
  step(a)  → executa ação, retorna (obs, reward, terminated, truncated, info)
  render() → visualiza o ambiente (opcional)
  close()  → libera recursos

Esta API padrão permite que o mesmo código de agente (DQN, PPO etc.)
funcione em QUALQUER ambiente Gymnasium — do CartPole ao Atari.
"""

import numpy as np


class CartPoleSimulado:
    """
    Simulação simplificada do CartPole para demonstração da API Gymnasium.
    Não requer gymnasium instalado.
    """

    def __init__(self):
        # Limites do CartPole real
        self.angle_limit = 0.2095    # ~12 graus em radianos
        self.pos_limit = 2.4         # metros

        # Estado interno
        self._state = None
        self._step_count = 0
        self._max_steps = 500

    @property
    def observation_space(self):
        """Box(4,) — 4 valores contínuos."""
        class Box:
            shape = (4,)
            def __repr__(self):
                return "Box(4,) — [posicao, velocidade, angulo, vel_angular]"
        return Box()

    @property
    def action_space(self):
        """Discrete(2) — 0=esquerda, 1=direita."""
        class Discrete:
            n = 2
            def sample(self):
                return np.random.choice(2)
            def __repr__(self):
                return "Discrete(2) — 0=esquerda, 1=direita"
        return Discrete()

    def reset(self, seed: int = None) -> tuple:
        """
        Inicia novo episódio.
        Retorna: (observacao, info) — padrão Gymnasium >= 0.26
        """
        if seed is not None:
            np.random.seed(seed)
        # CartPole inicializa com valores pequenos aleatórios
        self._state = np.random.uniform(-0.05, 0.05, 4)
        self._step_count = 0
        info = {}
        return self._state.copy(), info

    def step(self, action: int) -> tuple:
        """
        Executa ação no ambiente.
        Retorna: (observation, reward, terminated, truncated, info)

        Diferença entre terminated e truncated:
          terminated: condição de fim natural (polo caiu, objetivo alcancado)
          truncated:  limite de passos atingido (timeout)
        """
        assert action in [0, 1], f"Acao invalida: {action}"

        # Dinâmica do CartPole (simplificada)
        pos, vel, ang, ang_vel = self._state

        # Física do pêndulo invertido
        force = 10.0 if action == 1 else -10.0
        ang_acc = (9.8 * np.sin(ang) - np.cos(ang) * force) / 1.5
        ang_vel = ang_vel + 0.02 * ang_acc
        ang = ang + 0.02 * ang_vel

        vel = vel + 0.02 * (force - ang_acc * np.cos(ang) * 0.5)
        pos = pos + 0.02 * vel

        self._state = np.array([pos, vel, ang, ang_vel])
        self._step_count += 1

        # Recompensa: +1 por cada passo com vara em pé
        reward = 1.0

        # terminated: polo caiu ou saiu dos limites
        terminated = bool(
            abs(ang) > self.angle_limit or abs(pos) > self.pos_limit
        )

        # truncated: limite de tempo atingido
        truncated = self._step_count >= self._max_steps

        info = {"step": self._step_count}
        observation = self._state.copy()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Renderiza estado atual (versão texto)."""
        pos, vel, ang, ang_vel = self._state
        print(f"  [Render] pos={pos:+.3f}m  vel={vel:+.3f}  "
              f"ang={ang:+.3f}rad ({np.degrees(ang):+.1f}°)  "
              f"ang_vel={ang_vel:+.3f}  step={self._step_count}")

    def close(self):
        """Libera recursos (não necessário para simulação)."""
        pass


if __name__ == "__main__":
    print("=" * 60)
    print("GO1816 - API GYMNASIUM: reset, step, render, close")
    print("=" * 60)

    env = CartPoleSimulado()

    # ─── Espaços ──────────────────────────────────────────────
    print("\nESPACOS DO CARTPOLE:")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.action_space      = {env.action_space}")

    # ─── Reset ────────────────────────────────────────────────
    print()
    print("─" * 60)
    print("1. env.reset() — iniciar novo episodio")
    print("─" * 60)
    observation, info = env.reset(seed=42)
    print(f"  observation = {observation.round(4)}")
    print(f"  Significado: [posicao, velocidade, angulo, vel_angular]")
    print(f"  info = {info}")

    # ─── Step ─────────────────────────────────────────────────
    print()
    print("─" * 60)
    print("2. env.step(action) — executar acao")
    print("─" * 60)

    acoes = [1, 1, 0, 1, 0]
    for i, action in enumerate(acoes):
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"  step({action}): reward={reward:.1f}  "
              f"term={terminated}  trunc={truncated}  done={done}")
        print(f"    obs = {obs.round(3)}")

    # ─── Render ───────────────────────────────────────────────
    print()
    print("─" * 60)
    print("3. env.render() — visualizar estado")
    print("─" * 60)
    env.render()

    # ─── Episódio completo ────────────────────────────────────
    print()
    print("─" * 60)
    print("4. EPISODIO COMPLETO com politica aleatoria")
    print("─" * 60)
    observation, info = env.reset(seed=99)
    total_reward = 0
    passos = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        passos += 1
        if done:
            razao = "polo caiu" if terminated else "timeout"
            print(f"  Encerrado apos {passos} passos ({razao})")
            print(f"  Total reward: {total_reward:.1f}")
            break

    env.close()

    # ─── Diferença terminated vs truncated ───────────────────
    print()
    print("─" * 60)
    print("TERMINATED vs TRUNCATED:")
    print("─" * 60)
    print("  terminated: episodio terminou por CONDICAO NATURAL")
    print("    Ex: pole caiu, objetivo alcancado, agente morreu")
    print()
    print("  truncated:  episodio cortado por LIMITE DE TEMPO")
    print("    Ex: 500 passos atingidos sem terminar naturalmente")
    print()
    print("  Por que importa?")
    print("  - Se truncated: o proximo estado NAO e terminal")
    print("  -   → usar bootstrap: target = r + γ V(s')")
    print("  - Se terminated: proximo estado e terminal")
    print("  -   → target = r  (sem valor futuro)")
    print()
    print("  done = terminated or truncated  (para simplificar)")
    print("  Mas para targets corretos: verificar separadamente!")
