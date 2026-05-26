"""
GO1837 - Recompensa para Market Making (PnL + Risco de Inventário)
==================================================================
Demonstra a função de recompensa para um agente de market making.
Requer apenas numpy.

O que é Market Making?
  Corretoras e traders especializados postam ordens de compra (bid) e venda
  (ask) simultaneamente, ganhando o spread (diferença bid-ask). O risco é
  acumular muito estoque de um ativo (inventário) se o mercado mover contra.

Componentes da recompensa:
  + PnL (Profit and Loss): lucro direto com o spread
  - Risco de inventário: penaliza posições grandes (exposição ao mercado)
  - Taxa de cancelamento: penaliza cancelar ordens excessivamente
  + Rebate de liquidez: exchanges pagam traders que adicionam liquidez
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class EstadoMarketMaker:
    """Estado do agente de market making."""
    inventario: float        # Posição atual (+: long, -: short)
    pnl_acumulado: float     # Lucro/Prejuízo acumulado
    spread_bid_ask: float    # Spread atual (em R$)
    volatilidade: float      # Volatilidade do ativo (desvio padrão)
    volume_ordens: int       # Volume de ordens executadas


def calcular_reward_market_maker(
    pnl: float,              # Lucro/Prejuízo do período (R$)
    inventario: float,       # Posição atual (em unidades do ativo)
    preco_ativo: float,      # Preço atual do ativo (para escalar inventário)
    ordens_canceladas: int,  # Número de ordens canceladas
    rebate_liquidez: float,  # Rebate pago pela exchange (R$)
    peso_inventario: float = 0.1,   # Quanto penalizar inventário excessivo
    custo_cancelamento: float = 0.01,  # Custo por cancelamento
) -> dict:
    """
    Calcula a recompensa multi-componente para market making.

    O agente precisa equilibrar:
    - Maximizar PnL com spreads
    - Minimizar exposição ao risco direcional (inventário)
    - Evitar cancelamentos excessivos (sinalização ruim para a exchange)
    - Maximizar rebates por fornecer liquidez
    """
    # PnL direto: lucro com o spread bid-ask
    r_pnl = pnl

    # Penalidade de inventário: posição grande = risco direcional
    # Normaliza pelo valor do ativo para ficar em escala monetária
    valor_inventario = abs(inventario) * preco_ativo
    r_inventario = -peso_inventario * valor_inventario

    # Penalidade por cancelar ordens: exchanges podem banir por cancel ratio alto
    r_cancelamento = -custo_cancelamento * ordens_canceladas

    # Rebate: exchanges pagam quem adiciona liquidez (maker fee negativo)
    r_rebate = rebate_liquidez

    total = r_pnl + r_inventario + r_cancelamento + r_rebate

    return {
        "pnl": r_pnl,
        "risco_inventario": r_inventario,
        "custo_cancelamentos": r_cancelamento,
        "rebate_liquidez": r_rebate,
        "total": total,
        "inventario_atual": inventario,
    }


def simular_sessao_market_making(
    n_ciclos: int = 10,
    preco_inicial: float = 100.0,
    seed: int = 7,
) -> None:
    """
    Simula uma sessão de market making com o agente ajustando spreads.
    A cada ciclo, algumas ordens são executadas e o inventário muda.
    """
    np.random.seed(seed)
    preco = preco_inicial
    inventario = 0.0
    pnl_total = 0.0

    print(f"  {'Ciclo':>6} | {'Preço':>8} | {'PnL':>7} | "
          f"{'Inventário':>11} | {'Reward':>8}")
    print("  " + "-" * 52)

    for i in range(1, n_ciclos + 1):
        # Variação de preço (passeio aleatório)
        preco *= (1 + np.random.normal(0, 0.002))

        # PnL: ganho com spread (aprox. 0.05% do valor negociado)
        volume_executado = np.random.uniform(0, 500)
        pnl = volume_executado * 0.0005

        # Inventário: acumula se um lado do mercado domina
        delta_inv = np.random.normal(0, 2.0)  # Variação estocástica
        inventario += delta_inv

        # Cancelamentos ocasionais
        cancelamentos = np.random.poisson(1)

        # Rebate da exchange
        rebate = volume_executado * 0.0002

        comp = calcular_reward_market_maker(
            pnl=pnl,
            inventario=inventario,
            preco_ativo=preco,
            ordens_canceladas=cancelamentos,
            rebate_liquidez=rebate,
        )
        pnl_total += pnl

        inv_str = f"{inventario:+.1f}"
        print(f"  {i:>6} | R${preco:>6.2f} | {pnl:>+6.2f} | "
              f"{inv_str:>11} | {comp['total']:>+8.3f}")

    print(f"\n  PnL acumulado: R${pnl_total:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("GO1837 - RECOMPENSA MARKET MAKING (PNL + INVENTARIO)")
    print("=" * 60)

    print("\nFORMULA:")
    print()
    print("  reward = + pnl                    # Lucro com spread")
    print("           - inventario_risk * 0.1  # Penaliza posicao grande")
    print("           - order_cancel_fee * 0.01# Penaliza cancelamentos")
    print("           + liquidity_rebate        # Exchange paga liquidez")

    print()
    print("─" * 60)
    print("CENARIOS:")
    print("─" * 60)

    PRECO = 100.0  # Preço do ativo

    cenarios = [
        # (descricao, pnl, inventario, cancelamentos, rebate)
        ("Operação ideal (spread, baixo inv.)",  2.50,   1.0,  0, 0.50),
        ("Inventário alto (risco direcional)",   2.50,  50.0,  0, 0.50),
        ("Muitos cancelamentos (penalizado)",    2.50,   1.0, 30, 0.50),
        ("Prejuízo + sem rebate",               -1.00,   5.0,  2, 0.00),
        ("Alta liquidez (rebate generoso)",      3.00,   2.0,  1, 2.00),
    ]

    for desc, pnl, inv, canc, reb in cenarios:
        comp = calcular_reward_market_maker(pnl, inv, PRECO, canc, reb)
        print(f"\n  [{desc}]")
        for k, v in comp.items():
            if k not in ("total", "inventario_atual"):
                print(f"    {k:<25}: {v:>+8.3f}")
        print(f"    {'TOTAL':<25}: {comp['total']:>+8.3f}")

    print()
    print("─" * 60)
    print("SIMULACAO DE SESSAO (10 CICLOS):")
    print("─" * 60)
    print()
    simular_sessao_market_making(n_ciclos=10)

    print()
    print("─" * 60)
    print("DESAFIOS DO RL EM MARKET MAKING:")
    print("─" * 60)
    print("  + Dados de alta frequência disponíveis (tick-by-tick)")
    print("  + Feedback imediato (PnL em tempo real)")
    print("  - Latência: decisões em microssegundos")
    print("  - Adversarial: outros traders reagem ao comportamento do agente")
    print("  - Risco de inventário noturno (overnight risk)")
    print("  - Regulação: manipulação de mercado é crime")
