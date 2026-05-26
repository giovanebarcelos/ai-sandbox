"""
GO1835 - Ações de Rebalanceamento de Portfólio
===============================================
Demonstra o espaço de ações para gestão de portfólio com RL.
Requer apenas numpy.

Para cada ativo no portfólio: a ação é um valor contínuo em [-1.0, +1.0]:
  -1.0 = vender tudo deste ativo
   0.0 = manter posição atual
  +1.0 = comprar o máximo possível

Exemplo: actions = [0.2, -0.5, 0.1, ...]
  0.2  = comprar 20% mais de AAPL
  -0.5 = vender 50% da posição em GOOGL
  0.1  = comprar 10% mais de AMZN
"""

import numpy as np
from typing import Dict, List


def rebalancear_portfolio(
    holdings: Dict[str, float],  # {ativo: qtd_acoes}
    prices: Dict[str, float],    # {ativo: preco}
    cash: float,
    acoes: np.ndarray,           # valor em [-1, 1] para cada ativo
    ativos: List[str],
    custo_transacao: float = 0.001,  # 0.1% por transação
) -> tuple:
    """
    Executa o rebalanceamento do portfólio baseado nas ações do agente.
    Retorna (novo_holdings, novo_cash, custo_total_transacoes).
    """
    novo_holdings = holdings.copy()
    novo_cash = float(cash)
    custo_total = 0.0

    for ativo, acao in zip(ativos, acoes):
        preco = prices[ativo]
        qtd_atual = holdings.get(ativo, 0.0)

        if acao > 0:  # Comprar
            valor_compra = novo_cash * acao
            qtd_comprar = valor_compra / preco
            custo = valor_compra * custo_transacao
            novo_holdings[ativo] = qtd_atual + qtd_comprar
            novo_cash -= valor_compra + custo
            custo_total += custo
        elif acao < 0:  # Vender
            qtd_vender = qtd_atual * abs(acao)
            valor_venda = qtd_vender * preco
            custo = valor_venda * custo_transacao
            novo_holdings[ativo] = qtd_atual - qtd_vender
            novo_cash += valor_venda - custo
            custo_total += custo

    # Garantir que cash não seja negativo
    novo_cash = max(0.0, novo_cash)

    return novo_holdings, novo_cash, custo_total


def calcular_valor_portfolio(holdings: Dict[str, float],
                             prices: Dict[str, float],
                             cash: float) -> float:
    """Calcula o valor total do portfólio."""
    return cash + sum(holdings.get(a, 0) * prices[a] for a in prices)


if __name__ == "__main__":
    print("=" * 60)
    print("GO1835 - ACOES DE REBALANCEAMENTO DE PORTFOLIO")
    print("=" * 60)

    print("\nESPACO DE ACOES (Continuo):")
    print()
    print("  Para cada ativo: acao em [-1.0, +1.0]")
    print("    -1.0 = vender TUDO")
    print("     0.0 = manter posicao atual")
    print("    +1.0 = comprar o maximo possível")
    print()
    print("  actions = [0.2, -0.5, 0.1]")
    print("    0.2  = comprar 20% mais de AAPL")
    print("    -0.5 = vender 50% de GOOGL")
    print("    0.1  = comprar 10% mais de AMZN")

    # Estado inicial
    ativos = ["AAPL", "GOOGL", "AMZN"]
    prices = {"AAPL": 150.0, "GOOGL": 2800.0, "AMZN": 3200.0}
    holdings = {"AAPL": 100.0, "GOOGL": 5.0, "AMZN": 3.0}
    cash = 5000.0

    valor_inicial = calcular_valor_portfolio(holdings, prices, cash)

    print()
    print("─" * 60)
    print("ESTADO INICIAL DO PORTFOLIO:")
    print("─" * 60)
    for ativo in ativos:
        v = holdings.get(ativo, 0) * prices[ativo]
        pct = v / valor_inicial * 100
        print(f"  {ativo:6s}: {holdings.get(ativo,0):6.1f} acoes × "
              f"R${prices[ativo]:8.2f} = R${v:10.2f}  ({pct:.1f}%)")
    print(f"  {'Caixa':6s}: R${cash:10.2f}  ({cash/valor_inicial*100:.1f}%)")
    print(f"  {'TOTAL':6s}: R${valor_inicial:10.2f}")

    # Aplicar diferentes estratégias
    estrategias = [
        ("Comprar agressivo",   np.array([0.5, 0.3, 0.2])),
        ("Vender agressivo",    np.array([-0.5, -0.3, -0.2])),
        ("Rebalancear neutro",  np.array([0.1, -0.1, 0.0])),
        ("Concentrar em AAPL", np.array([0.8, -0.8, -0.5])),
    ]

    print()
    print("─" * 60)
    print("RESULTADO DE DIFERENTES ESTRATEGIAS:")
    print("─" * 60)

    for nome, acoes in estrategias:
        novo_h, novo_c, custo = rebalancear_portfolio(
            holdings, prices, cash, acoes, ativos
        )
        novo_valor = calcular_valor_portfolio(novo_h, prices, novo_c)
        print(f"\n  [{nome}]  acoes={acoes}")
        print(f"  Custo transacoes: R${custo:.2f}")
        print(f"  Novo portfólio:")
        for ativo in ativos:
            print(f"    {ativo}: {novo_h.get(ativo,0):.1f} acoes")
        print(f"  Caixa: R${novo_c:.2f}")
        print(f"  Valor total: R${novo_valor:.2f}  "
              f"(δ = {novo_valor - valor_inicial:+.2f})")

    print()
    print("  Algoritmos para ações contínuas:")
    print("  PPO, SAC (Soft Actor-Critic), TD3 — mais eficientes que DQN")
    print("  DQN só funciona para ações discretas!")
