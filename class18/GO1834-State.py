"""
GO1834 - Estado para Gestão de Portfólio Financeiro
====================================================
Demonstra o espaço de estados para RL aplicado a finanças.
Requer apenas numpy.

RL em finanças: o agente observa o estado do mercado e do portfólio
e decide como rebalancear (comprar/vender ativos).

O estado inclui:
  - Preços atuais dos ativos
  - Posição atual (quantas ações possui)
  - Caixa disponível
  - Indicadores técnicos (RSI, MACD, médias móveis)
  - Indicadores macro (VIX, juros, PIB)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class EstadoPortfolio:
    """
    Representação do estado para gestão de portfólio via RL.
    Baseado em marcadores técnicos e fundamentais.
    """
    # Posição atual
    prices: Dict[str, float] = field(default_factory=dict)
    holdings: Dict[str, int] = field(default_factory=dict)
    cash: float = 10000.0
    portfolio_value: float = 10000.0

    # Indicadores técnicos
    rsi: Dict[str, float] = field(default_factory=dict)
    macd: Dict[str, float] = field(default_factory=dict)
    moving_avg_50d: Dict[str, float] = field(default_factory=dict)

    # Indicadores macro
    vix: float = 18.5        # Volatilidade do mercado
    interest_rate: float = 4.5  # Taxa de juros (%)
    gdp_growth: float = 2.3  # Crescimento do PIB (%)

    def to_vector(self) -> np.ndarray:
        """Converte para vetor normalizado para a rede neural."""
        ativos = sorted(self.prices.keys())
        partes = []

        # Preços normalizados
        partes.extend([self.prices[a] / 1000.0 for a in ativos])

        # Holdings normalizados
        total_acoes = sum(self.holdings.values()) + 1
        partes.extend([self.holdings.get(a, 0) / total_acoes for a in ativos])

        # Caixa e valor
        partes.append(self.cash / self.portfolio_value)

        # Indicadores técnicos
        partes.extend([self.rsi.get(a, 50) / 100.0 for a in ativos])
        partes.extend([np.tanh(self.macd.get(a, 0) / 5.0) for a in ativos])

        # Macro
        partes.extend([self.vix / 50.0, self.interest_rate / 10.0, self.gdp_growth / 5.0])

        return np.array(partes)

    def retorno_total(self, valor_inicial: float) -> float:
        return (self.portfolio_value - valor_inicial) / valor_inicial


def gerar_estado_mercado(
    ativos: List[str] = None,
    cenario: str = "bull",
    seed: int = 42,
) -> EstadoPortfolio:
    """
    Gera um estado de mercado simulado.
    cenario: "bull" (alta), "bear" (baixa), "sideways" (lateral)
    """
    np.random.seed(seed)
    if ativos is None:
        ativos = ["AAPL", "GOOGL", "AMZN", "MSFT"]

    precos_base = {"AAPL": 150, "GOOGL": 2800, "AMZN": 3200, "MSFT": 300}

    params = {
        "bull":     (1.1, 0.05, 15.0, 3.0, 2.5),
        "bear":     (0.85, 0.12, 35.0, 5.5, -0.5),
        "sideways": (1.0, 0.08, 20.0, 4.5, 1.5),
    }
    mult, vol, vix, juros, pib = params.get(cenario, params["sideways"])

    prices = {a: precos_base.get(a, 100) * mult * (1 + np.random.randn() * vol)
              for a in ativos}
    holdings = {a: int(np.random.choice([0, 50, 100, 150])) for a in ativos}
    cash = float(np.random.uniform(5000, 20000))
    port_value = cash + sum(prices[a] * holdings.get(a, 0) for a in ativos)

    return EstadoPortfolio(
        prices=prices,
        holdings=holdings,
        cash=cash,
        portfolio_value=port_value,
        rsi={a: float(np.random.uniform(30, 70)) for a in ativos},
        macd={a: float(np.random.randn() * 3) for a in ativos},
        moving_avg_50d={a: prices[a] * (1 + np.random.randn() * 0.02) for a in ativos},
        vix=vix + np.random.randn() * 2,
        interest_rate=juros,
        gdp_growth=pib,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("GO1834 - ESTADO PARA GESTAO DE PORTFOLIO (RL)")
    print("=" * 60)

    print("\nESTRUTURA DO ESTADO:")
    print()
    print("  state = {")
    print("    'prices':    {'AAPL': 150, 'GOOGL': 2800, ...}")
    print("    'holdings':  {'AAPL': 100, 'GOOGL': 50, ...}")
    print("    'cash':      10000,")
    print("    'portfolio_value': 50000,")
    print("    'market_indicators': {'VIX': 18.5, 'interest_rate': 4.5, 'GDP_growth': 2.3}")
    print("    'technical_indicators': {'RSI': [45, 67, ...], 'MACD': [...]}")
    print("  }")

    ativos = ["AAPL", "GOOGL", "AMZN"]

    print()
    print("─" * 60)
    print("ESTADOS EM DIFERENTES CENARIOS DE MERCADO:")
    print("─" * 60)

    for cenario in ["bull", "sideways", "bear"]:
        estado = gerar_estado_mercado(ativos=ativos, cenario=cenario)
        print(f"\n  [{cenario.upper()}]")
        print("  Precos:")
        for ativo, preco in estado.prices.items():
            rsi = estado.rsi[ativo]
            sinal_rsi = "sobrecomprado" if rsi > 70 else ("sobrevendido" if rsi < 30 else "neutro")
            print(f"    {ativo:6s}: R${preco:8.2f}  | RSI={rsi:.1f} ({sinal_rsi})")
        print(f"  Caixa: R${estado.cash:,.2f} | Portfólio: R${estado.portfolio_value:,.2f}")
        print(f"  Macro: VIX={estado.vix:.1f}, Juros={estado.interest_rate:.1f}%, PIB={estado.gdp_growth:.1f}%")

    # Vetor para rede neural
    estado_bull = gerar_estado_mercado(ativos=ativos, cenario="bull")
    vetor = estado_bull.to_vector()
    print()
    print("─" * 60)
    print(f"VETOR NORMALIZADO ({len(vetor)} dimensoes):")
    print("─" * 60)
    print(f"  {vetor.round(3)}")

    print()
    print("─" * 60)
    print("DESAFIOS DO RL EM FINANCAS:")
    print("─" * 60)
    print("  + Dados abundantes (histórico de preços)")
    print("  + Backtest possível antes de produção")
    print("  - Mercado é não-estacionário (distribuição muda)")
    print("  - Overfitting ao período de treino (look-ahead bias)")
    print("  - Transaction costs: cada trade tem custo real")
    print("  - Slippage: ordens grandes movem o mercado")
