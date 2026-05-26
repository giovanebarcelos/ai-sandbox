"""
GO1807 - Experience Replay: Quebrando a Correlação Temporal
============================================================
Demonstra o Experience Replay do DQN com exemplo executável.
Requer apenas numpy.

Problema: em RL online, amostras consecutivas são altamente correlacionadas
(s_t e s_{t+1} são quase idênticos). Treinar uma rede com amostras
correlacionadas causa instabilidade ("catastrophic forgetting").

Solução (Lin, 1992 / Mnih et al. 2013): Replay Buffer
1. Armazenar transições (s, a, r, s', done) em um buffer
2. A cada passo, amostrar um MINI-BATCH aleatório do buffer
3. Treinar na rede com o mini-batch

Vantagens:
  + Quebra correlação temporal (amostras aleatórias do passado)
  + Usa cada transição múltiplas vezes (mais eficiente em dados)
  + Estabiliza treinamento da rede neural
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Buffer circular para Experience Replay.
    Implementação do Replay Buffer do DQN clássico.
    """

    def __init__(self, maxlen: int = 10000):
        # deque com maxlen descarta automaticamente os mais antigos
        self.buffer = deque(maxlen=maxlen)

    def push(self, estado, acao: int, recompensa: float,
             prox_estado, done: bool) -> None:
        """Armazena uma transição."""
        self.buffer.append((estado, acao, recompensa, prox_estado, done))

    def sample(self, batch_size: int) -> list:
        """
        Amostra um mini-batch ALEATÓRIO.
        Aleatoriedade é essencial para quebrar a correlação temporal.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer tem {len(self.buffer)} amostras, "
                f"mas batch_size={batch_size}"
            )
        return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)


def demonstrar_correlacao(buffer: ReplayBuffer, n_amostras: int = 6) -> None:
    """
    Mostra a diferença entre amostragem sequencial e aleatória.
    """
    transitions_list = list(buffer.buffer)

    print("  Sequencial (CORRELACIONADO):")
    for t in transitions_list[-n_amostras:]:
        s, a, r, sp, d = t
        print(f"    s={s:.2f}, a={a}, r={r:+.1f}, s'={sp:.2f}")

    print()
    print("  Aleatório via Replay Buffer (DECORRELACIONADO):")
    sample = buffer.sample(n_amostras)
    for t in sample:
        s, a, r, sp, d = t
        print(f"    s={s:.2f}, a={a}, r={r:+.1f}, s'={sp:.2f}")


def simular_preenchimento_buffer() -> ReplayBuffer:
    """
    Simula um agente coletando transições e armazenando no buffer.
    """
    np.random.seed(42)
    random.seed(42)

    buf = ReplayBuffer(maxlen=1000)

    # Simular 200 passos de interação com um ambiente contínuo simples
    # Estado = posição (float), Ação = 0 ou 1
    estado = 0.0
    for step in range(200):
        acao = random.choice([0, 1])

        # Transição simulada (estado contínuo)
        prox_estado = estado + (0.1 if acao == 1 else -0.1) + np.random.randn() * 0.02
        prox_estado = float(np.clip(prox_estado, -1, 1))

        # Recompensa: positiva quando próximo de 0.5
        recompensa = float(1.0 - abs(prox_estado - 0.5) * 2)
        done = step == 199

        buf.push(estado, acao, recompensa, prox_estado, done)
        estado = prox_estado

    return buf


if __name__ == "__main__":
    print("=" * 60)
    print("GO1807 - EXPERIENCE REPLAY")
    print("=" * 60)

    print("\nCONCEITO:")
    print()
    print("  # Armazenar transicoes no buffer")
    print("  replay_buffer = deque(maxlen=100000)")
    print("  replay_buffer.append((s, a, r, s_prox, done))")
    print()
    print("  # A cada passo: amostrar ALEATORIO")
    print("  if len(replay_buffer) > batch_size:")
    print("      batch = random.sample(replay_buffer, batch_size)")
    print("      train_on_batch(batch)  # Quebra correlacao temporal!")

    # ─── Demonstrar com buffer preenchido ─────────────────────
    print()
    print("─" * 60)
    print("DEMONSTRACAO: Correlacao sequencial vs. aleatoria")
    print("─" * 60)

    buf = simular_preenchimento_buffer()
    print(f"\n  Buffer preenchido: {len(buf)} transicoes")
    demonstrar_correlacao(buf, n_amostras=5)

    # ─── Simular ciclo de treinamento DQN ─────────────────────
    print()
    print("─" * 60)
    print("CICLO DE TREINAMENTO COM REPLAY BUFFER:")
    print("─" * 60)

    batch_size = 32
    n_treinos = 5

    print(f"\n  batch_size = {batch_size}")
    print(f"  Realizando {n_treinos} passos de treinamento:")

    for t in range(n_treinos):
        batch = buf.sample(batch_size)

        # Desempacotar batch
        estados = np.array([b[0] for b in batch])
        acoes = np.array([b[1] for b in batch])
        recompensas = np.array([b[2] for b in batch])
        prox_estados = np.array([b[3] for b in batch])

        # Simular perda média (em produção: treinar a rede neural)
        td_errors = abs(recompensas - np.mean(recompensas))
        loss_simulado = float(np.mean(td_errors ** 2))

        print(f"    Treino {t + 1}: "
              f"r_media={np.mean(recompensas):+.3f}, "
              f"loss_simulado={loss_simulado:.5f}")

    # ─── Estatísticas do buffer ───────────────────────────────
    print()
    print("─" * 60)
    print("ESTATISTICAS DO REPLAY BUFFER:")
    print("─" * 60)

    todas = list(buf.buffer)
    recomp = [t[2] for t in todas]
    print(f"\n  Capacidade   : {buf.buffer.maxlen}")
    print(f"  Preenchido   : {len(buf)} ({len(buf)/buf.buffer.maxlen:.0%})")
    print(f"  Recomp. média: {np.mean(recomp):.3f}")
    print(f"  Recomp. min  : {min(recomp):.3f}")
    print(f"  Recomp. max  : {max(recomp):.3f}")

    print()
    print("  Por que usar deque(maxlen=N)?")
    print("  - Quando buffer enche, descarta transicoes antigas")
    print("  - Mantém só as N transicoes mais recentes (mais relevantes)")
    print("  - Complexidade O(1) para inserção e remoção")
    print()
    print("  Ver GO1817 para DQN completo com Replay Buffer no CartPole.")
