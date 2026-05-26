"""
GO1828 - Representação de Estado para Direção Autônoma
========================================================
Demonstra o espaço de estados para um agente de direção autônoma.
Requer apenas numpy.

Conceito: o estado captura tudo o que o agente precisa saber para tomar
a decisão ótima (Propriedade de Markov). Um estado bem-projetado:
- Contém informação suficiente (não ambíguo)
- É eficiente (sem redundâncias)
- É observável pelo agente (sensores reais)

Para direção autônoma, o estado inclui:
- Velocidade própria e do tráfego
- Posição na pista
- Distâncias e velocidades dos veículos próximos
- Sinalização e meta de velocidade
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class EstadoVeiculoAutonomo:
    """Representação do estado para direção autônoma."""
    ego_speed: float       # Velocidade própria (km/h)
    lane: int              # Faixa atual (1=direita, 2=centro, 3=esquerda)
    front_car_distance: float  # Distância para carro da frente (m)
    front_car_speed: float     # Velocidade do carro da frente (km/h)
    left_lane_clear: bool      # Faixa esquerda livre
    right_lane_clear: bool     # Faixa direita livre
    target_speed: float        # Velocidade alvo (km/h)

    def to_array(self) -> np.ndarray:
        """Converte para vetor numpy (formato para rede neural)."""
        return np.array([
            self.ego_speed / 120.0,          # Normalizado
            self.lane / 3.0,                  # Normalizado
            self.front_car_distance / 200.0,  # Normalizado
            self.front_car_speed / 120.0,     # Normalizado
            float(self.left_lane_clear),      # 0 ou 1
            float(self.right_lane_clear),     # 0 ou 1
            self.target_speed / 120.0,        # Normalizado
        ])

    def situacao(self) -> str:
        """Interpreta a situação atual."""
        partes = []
        if self.front_car_distance < 30:
            partes.append("PERIGO: carro muito perto!")
        elif self.front_car_distance < 60:
            partes.append("Carro próximo à frente")
        if self.ego_speed < self.target_speed - 10:
            partes.append("Abaixo da velocidade alvo")
        elif self.ego_speed > self.target_speed + 5:
            partes.append("Acima da velocidade alvo")
        if not partes:
            partes.append("Situação normal")
        return " | ".join(partes)


def simular_cenarios() -> List[EstadoVeiculoAutonomo]:
    """Cria cenários típicos de direção autônoma."""
    return [
        EstadoVeiculoAutonomo(
            ego_speed=65, lane=2, front_car_distance=50,
            front_car_speed=55, left_lane_clear=True,
            right_lane_clear=False, target_speed=70,
        ),
        EstadoVeiculoAutonomo(
            ego_speed=90, lane=3, front_car_distance=15,
            front_car_speed=60, left_lane_clear=False,
            right_lane_clear=True, target_speed=80,
        ),
        EstadoVeiculoAutonomo(
            ego_speed=70, lane=1, front_car_distance=200,
            front_car_speed=80, left_lane_clear=True,
            right_lane_clear=False, target_speed=100,
        ),
    ]


if __name__ == "__main__":
    print("=" * 60)
    print("GO1828 - ESTADO PARA DIRECAO AUTONOMA")
    print("=" * 60)

    print("\nESTRUTURA DO ESTADO:")
    print()
    campos = [
        ("ego_speed",           "Velocidade do veículo próprio (km/h)"),
        ("lane",                "Faixa atual [1=direita, 2=centro, 3=esquerda]"),
        ("front_car_distance",  "Distância ao carro da frente (metros)"),
        ("front_car_speed",     "Velocidade do carro da frente (km/h)"),
        ("left_lane_clear",     "Faixa esquerda livre [True/False]"),
        ("right_lane_clear",    "Faixa direita livre [True/False]"),
        ("target_speed",        "Velocidade alvo do trajeto (km/h)"),
    ]
    for campo, desc in campos:
        print(f"  {campo:25s}: {desc}")

    print()
    print("─" * 60)
    print("ACOES POSSIVEIS:")
    print("─" * 60)
    acoes = [
        ("0 - stay_lane",     "Manter faixa e velocidade"),
        ("1 - change_left",   "Mudar para faixa da esquerda"),
        ("2 - change_right",  "Mudar para faixa da direita"),
        ("3 - accelerate",    "Acelerar (+5 km/h)"),
        ("4 - brake",         "Frear (-5 km/h)"),
    ]
    for acao, desc in acoes:
        print(f"  {acao:20s}: {desc}")

    print()
    print("─" * 60)
    print("CENARIOS DE ESTADO:")
    print("─" * 60)

    for i, estado in enumerate(simular_cenarios(), 1):
        print(f"\n  Cenário {i}: {estado.situacao()}")
        print(f"    Velocidade própria: {estado.ego_speed} km/h "
              f"(alvo: {estado.target_speed} km/h)")
        print(f"    Faixa: {estado.lane} | "
              f"Esq. livre: {estado.left_lane_clear} | "
              f"Dir. livre: {estado.right_lane_clear}")
        print(f"    Carro à frente: {estado.front_car_distance}m, "
              f"{estado.front_car_speed} km/h")

        # Vetor para a rede neural
        vetor = estado.to_array()
        print(f"    Vetor normalizado: {vetor.round(3)}")

    print()
    print("─" * 60)
    print("POR QUE NORMALIZAR O ESTADO?")
    print("─" * 60)
    print("  Redes neurais convergem melhor com entradas em [-1, 1]")
    print("  Sem normalizar: distância(200m) domina velocidade(0.5) no gradiente")
    print("  Com normalizar: todos os features na mesma escala [0, 1]")
