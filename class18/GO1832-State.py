"""
GO1832 - Estado para Data Center (Otimização de Recursos)
==========================================================
Demonstra o espaço de estados para otimização de data center com RL.
Requer apenas numpy.

Contexto: Google (2016) usou DeepMind RL para reduzir consumo de energia
de seus data centers em 40%. O agente aprende a ajustar resfriamento
e alocação de recursos para minimizar PUE (Power Usage Effectiveness).

Estado do data center: centenas/milhares de métricas de servidores.
O agente observa esse "snapshot" e decide como realocar carga ou ajustar cooling.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EstadoDataCenter:
    """
    Representação do estado de um data center com N servidores.
    Em produção: N pode ser 500+ servidores.
    """
    cpu_usage: np.ndarray          # [0, 100] % por servidor
    memory_usage: np.ndarray       # [0, 100] % por servidor
    network_bandwidth: np.ndarray  # [0, 100] % por servidor
    pending_tasks: int             # Tarefas aguardando alocação
    sla_violations: int            # Violações de SLA na última hora
    temperatura_media: float       # Temperatura média (°C)
    pue: float                     # Power Usage Effectiveness

    @property
    def n_servidores(self) -> int:
        return len(self.cpu_usage)

    def to_vector(self) -> np.ndarray:
        """Converte para vetor para a rede neural."""
        return np.concatenate([
            self.cpu_usage / 100.0,
            self.memory_usage / 100.0,
            self.network_bandwidth / 100.0,
            [self.pending_tasks / 1000.0],
            [self.sla_violations / 100.0],
            [self.temperatura_media / 50.0],
            [self.pue / 2.0],
        ])

    def summary(self) -> dict:
        """Resumo das métricas principais."""
        return {
            "n_servidores": self.n_servidores,
            "cpu_media": float(np.mean(self.cpu_usage)),
            "cpu_max": float(np.max(self.cpu_usage)),
            "mem_media": float(np.mean(self.memory_usage)),
            "pending_tasks": self.pending_tasks,
            "sla_violations": self.sla_violations,
            "temperatura": self.temperatura_media,
            "pue": self.pue,
        }


def gerar_estado_data_center(
    n_servidores: int = 10,
    carga: str = "normal",
    seed: int = 42,
) -> EstadoDataCenter:
    """
    Gera um estado de data center simulado.
    carga: "baixa", "normal", "alta", "sobrecarga"
    """
    np.random.seed(seed)

    params = {
        "baixa":       (20, 15, 15,  10, 0, 18.0, 1.5),
        "normal":      (45, 50, 40, 100, 2, 22.0, 1.8),
        "alta":        (75, 70, 70, 350, 8, 28.0, 2.1),
        "sobrecarga":  (90, 88, 85, 800, 25, 35.0, 2.5),
    }
    cpu_m, mem_m, bw_m, tasks, sla, temp, pue = params.get(carga, params["normal"])

    return EstadoDataCenter(
        cpu_usage=np.clip(np.random.normal(cpu_m, 10, n_servidores), 0, 100),
        memory_usage=np.clip(np.random.normal(mem_m, 8, n_servidores), 0, 100),
        network_bandwidth=np.clip(np.random.normal(bw_m, 15, n_servidores), 0, 100),
        pending_tasks=int(np.random.poisson(tasks)),
        sla_violations=int(np.random.poisson(max(1, sla))),
        temperatura_media=temp + np.random.randn() * 1.5,
        pue=pue + np.random.randn() * 0.05,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("GO1832 - ESTADO PARA DATA CENTER (RL)")
    print("=" * 60)

    print("\nESTRUTURA DO ESTADO:")
    print()
    print("  state = {")
    print("    'cpu_usage_per_server':    [45%, 67%, 23%, ...]  # N valores")
    print("    'memory_usage_per_server': [...]")
    print("    'network_bandwidth':       [...]")
    print("    'pending_tasks':           342")
    print("    'sla_violations':          12")
    print("  }")
    print()
    print("  Dimensão do estado: 3 × N_servidores + 2")
    print("  Para N=500: vetor de 1502 dimensões!")

    print()
    print("─" * 60)
    print("ESTADOS EM DIFERENTES CARGAS:")
    print("─" * 60)

    for carga in ["baixa", "normal", "alta", "sobrecarga"]:
        estado = gerar_estado_data_center(n_servidores=10, carga=carga)
        s = estado.summary()
        print(f"\n  [{carga.upper():10s}]")
        print(f"    CPU média: {s['cpu_media']:5.1f}%  (máx: {s['cpu_max']:.1f}%)")
        print(f"    Memória: {s['mem_media']:5.1f}%")
        print(f"    Tarefas pendentes: {s['pending_tasks']}")
        print(f"    Violações de SLA: {s['sla_violations']}")
        print(f"    Temperatura: {s['temperatura']:.1f}°C")
        print(f"    PUE: {s['pue']:.2f}  "
              f"({'eficiente' if s['pue'] < 1.8 else 'alto consumo'})")

    # Vetor para rede neural
    estado = gerar_estado_data_center(n_servidores=5, carga="normal")
    vetor = estado.to_vector()
    print()
    print("─" * 60)
    print(f"VETOR NORMALIZADO (N=5 servidores): dim={len(vetor)}")
    print("─" * 60)
    print(f"  {vetor.round(3)}")

    print()
    print("─" * 60)
    print("ACOES DO AGENTE DE DATA CENTER:")
    print("─" * 60)
    acoes = [
        "Migrar carga do servidor X para Y",
        "Aumentar cooling na zona A",
        "Reduzir frequência de CPU dos servidores ociosos",
        "Priorizar filas de alta prioridade",
        "Ativar/desativar servidores em standby",
    ]
    for i, acao in enumerate(acoes, 1):
        print(f"  {i}. {acao}")

    print()
    print("  Google DeepMind resultado (2016):")
    print("  → Redução de 40% no consumo de energia de resfriamento")
    print("  → PUE médio caiu de 2.0 para 1.6")
