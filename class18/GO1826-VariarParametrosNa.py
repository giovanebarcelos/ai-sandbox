"""
GO1826 - Variação de Parâmetros na Simulação (Domain Randomization)
====================================================================
Demonstra Domain Randomization para treinar políticas robóticas robustas.
Requer apenas numpy.

Problema: se treinarmos um robô apenas em um ambiente fixo (atrito=1.0,
altura=0.0), a política aprende a explorar especificamente aquele ambiente.
No mundo real, condições variam — a política "quebra".

Solução: Domain Randomization (Tobin et al. 2017 — OpenAI)
Variar parâmetros do ambiente durante o treinamento de forma que a política
aprenda a funcionar bem em QUALQUER configuração do intervalo.

Usado em: robótica sim-to-real, veículos autônomos, drones.
"""

import numpy as np


class AmbienteRoboticoVariavel:
    """
    Ambiente de robô com parâmetros variáveis para Domain Randomization.
    A cada reset(), os parâmetros físicos são sorteados do intervalo de variação.
    """

    def __init__(self, randomize: bool = True):
        self.randomize = randomize
        # Parâmetros padrão (sem randomização)
        self.terrain_friction = 1.0
        self.terrain_height = 0.0
        self.motor_noise_std = 0.0

    def reset(self, seed: int = None) -> np.ndarray:
        """
        Inicializa episódio com parâmetros variados (se randomize=True).
        """
        if seed is not None:
            np.random.seed(seed)

        if self.randomize:
            # Domain Randomization: sortear parâmetros aleatoriamente
            self.terrain_friction = np.random.uniform(0.5, 1.5)
            self.terrain_height = np.random.uniform(-0.1, 0.1)
            self.motor_noise_std = np.random.uniform(0.0, 0.05)
        else:
            # Ambiente fixo (sem variação)
            self.terrain_friction = 1.0
            self.terrain_height = 0.0
            self.motor_noise_std = 0.0

        # Estado inicial: posição (x, y, z) + velocidade
        estado = np.array([0.0, 0.0, self.terrain_height, 0.0, 0.0, 0.0])
        return estado

    def step(self, acao: np.ndarray) -> tuple:
        """
        Executa ação com parâmetros físicos atuais.
        acao: vetor 3D de torques nos motores
        """
        # Adicionar ruído ao motor (realismo)
        acao_com_ruido = acao + np.random.normal(0, self.motor_noise_std, size=len(acao))

        # Simular movimento (simplificado)
        deslocamento = acao_com_ruido * self.terrain_friction
        nova_pos = deslocamento.sum()

        # Recompensa: quanto se moveu na direção certa
        recompensa = float(nova_pos > 0) * self.terrain_friction
        done = abs(nova_pos) > 5.0

        return np.array([nova_pos, 0.0, self.terrain_height, 0.0, 0.0, 0.0]), recompensa, done

    def info(self) -> dict:
        return {
            "terrain_friction": self.terrain_friction,
            "terrain_height": self.terrain_height,
            "motor_noise_std": self.motor_noise_std,
        }


def simular_treino_com_randomizacao(n_episodios: int = 10) -> None:
    """
    Mostra como os parâmetros variam a cada episódio.
    """
    env = AmbienteRoboticoVariavel(randomize=True)

    print("  Parametros variados a cada episodio (Domain Randomization):")
    print(f"  {'Ep':>4} | {'Atrito':>8} | {'Altura':>8} | {'Ruído Motor':>12}")
    print("  " + "-" * 45)

    for ep in range(n_episodios):
        env.reset(seed=ep * 7)
        info = env.info()
        print(f"  {ep+1:>4} | {info['terrain_friction']:>8.3f} | "
              f"{info['terrain_height']:>8.3f} | {info['motor_noise_std']:>12.4f}")


def comparar_robustez(n_testes: int = 100) -> None:
    """
    Compara política treinada com vs sem randomização.
    """
    # Simular taxa de sucesso em novo ambiente (atrito=0.7, altura=-0.08)
    # Política com randomização foi exposta a essa variação no treino
    # Política sem randomização foi treinada só em atrito=1.0, altura=0.0

    np.random.seed(42)
    sucessos_com = sum(np.random.random() < 0.82 for _ in range(n_testes))
    sucessos_sem = sum(np.random.random() < 0.31 for _ in range(n_testes))

    print(f"\n  Novo ambiente: atrito=0.7, altura=-0.08 (nao visto no treino)")
    print(f"  Com Domain Randomization : {sucessos_com}/{n_testes} ({sucessos_com/n_testes:.0%})")
    print(f"  Sem Domain Randomization : {sucessos_sem}/{n_testes} ({sucessos_sem/n_testes:.0%})")


if __name__ == "__main__":
    print("=" * 60)
    print("GO1826 - DOMAIN RANDOMIZATION PARA ROBOTICA")
    print("=" * 60)

    print("\nCONCEITO:")
    print()
    print("  # Variar parametros na simulacao para generalizar")
    print("  terrain_friction = np.random.uniform(0.5, 1.5)  # Variacao de atrito")
    print("  terrain_height   = np.random.uniform(-0.1, 0.1) # Altura variavel")
    print("  motor_noise      = np.random.normal(0, 0.05)    # Ruido nos motores")
    print()
    print("  Objetivo: politica aprende a funcionar bem em QUALQUER")
    print("  combinacao dos parametros dentro dos intervalos definidos.")

    print()
    print("─" * 60)
    print("SIMULACAO: PARAMETROS POR EPISODIO")
    print("─" * 60)
    simular_treino_com_randomizacao(n_episodios=10)

    print()
    print("─" * 60)
    print("COMPARACAO DE ROBUSTEZ:")
    print("─" * 60)
    comparar_robustez(n_testes=100)

    print()
    print("─" * 60)
    print("APLICACOES DE DOMAIN RANDOMIZATION:")
    print("─" * 60)
    aplicacoes = [
        ("OpenAI Dactyl (2019)", "Mao robótica aprende cubo rubik em simulação"),
        ("Boston Dynamics      ", "Robos caminham em terrenos variados"),
        ("Carros autônomos     ", "Variar clima, iluminação, tráfego"),
        ("Drones               ", "Variar vento, payload, falhas de motor"),
    ]
    for nome, desc in aplicacoes:
        print(f"  {nome}: {desc}")

    print()
    print("  Sim-to-Real Gap: diferença entre simulação e mundo real.")
    print("  Domain Randomization reduz esse gap ao forçar a política")
    print("  a aprender comportamentos robustos a variações.")
