"""
GO1836 - Recompensa Baseada em Índice de Sharpe (RL Financeiro)
================================================================
Demonstra a recompensa baseada em retorno ajustado ao risco (Sharpe).
Requer apenas numpy.

Conceito:
  O Índice de Sharpe mede o retorno de um investimento em relação ao risco
  assumido. Em RL financeiro, é usado como recompensa para encorajar o agente
  a buscar retornos com menor volatilidade — não apenas maximizar lucro cego.

  Retorno simples: (valor_hoje / valor_ontem) - 1
  Reward = (retorno - taxa_livre_risco) / volatilidade

  Sharpe alto = bons retornos com pouco risco
  Sharpe baixo = retornos ruins OU muito arriscados
"""

import numpy as np
from typing import List


def calcular_retorno_simples(
    valor_hoje: float,
    valor_ontem: float,
) -> float:
    """
    Calcula o retorno percentual de um período para outro.
    Ex: de R$100 para R$103 → retorno = 3%
    """
    # Retorno percentual: quanto cresceu em relação ao dia anterior
    return (valor_hoje / valor_ontem) - 1.0


def calcular_reward_sharpe(
    retorno: float,
    taxa_livre_risco: float,
    volatilidade: float,
) -> float:
    """
    Calcula a recompensa baseada no Índice de Sharpe.

    Quanto maior o retorno além da taxa livre de risco,
    e menor a volatilidade, maior a recompensa.

    retorno: retorno do período (ex: 0.02 = 2%)
    taxa_livre_risco: benchmark (ex: Selic diária ≈ 0.0004)
    volatilidade: desvio padrão dos retornos recentes
    """
    if volatilidade < 1e-8:
        # Evitar divisão por zero: se não há volatilidade, não há risco
        return retorno - taxa_livre_risco

    # Retorno excedente = quanto rendeu ALÉM do que renderia sem risco
    retorno_excedente = retorno - taxa_livre_risco

    # Normalizar pelo risco: mesmo retorno com menor risco = melhor
    return retorno_excedente / volatilidade


def calcular_sharpe_janela(
    historico_valores: List[float],
    taxa_livre_risco_diaria: float = 0.0004,  # Selic ~10% a.a. / 252 dias
    janela: int = 20,
) -> dict:
    """
    Calcula métricas Sharpe em uma janela deslizante de preços.
    Retorna retornos, volatilidade e Sharpe de cada período.
    """
    retornos = []
    for i in range(1, len(historico_valores)):
        r = calcular_retorno_simples(historico_valores[i], historico_valores[i - 1])
        retornos.append(r)

    retornos = np.array(retornos)

    # Usar os últimos `janela` retornos para calcular volatilidade
    if len(retornos) >= janela:
        retornos_janela = retornos[-janela:]
    else:
        retornos_janela = retornos

    volatilidade = float(np.std(retornos_janela)) if len(retornos_janela) > 1 else 0.01
    retorno_medio = float(np.mean(retornos_janela))

    sharpe = calcular_reward_sharpe(retorno_medio, taxa_livre_risco_diaria, volatilidade)

    return {
        "retorno_medio": retorno_medio,
        "volatilidade": volatilidade,
        "sharpe": sharpe,
        "n_periodos": len(retornos),
        "retornos": retornos.tolist(),
    }


def simular_portfolio_rl(
    n_dias: int = 30,
    valor_inicial: float = 10000.0,
    taxa_livre_risco: float = 0.0004,
    seed: int = 42,
) -> None:
    """
    Simula um agente de RL tentando maximizar Sharpe ao longo do tempo.
    Compara duas estratégias:
      - Agressiva: busca alto retorno, mas aceita volatilidade
      - Conservadora: Sharpe-aware, balancea retorno e risco
    """
    np.random.seed(seed)

    # Estratégia AGRESSIVA: alta volatilidade, retorno médio ok
    historico_agressivo = [valor_inicial]
    for _ in range(n_dias):
        r = np.random.normal(0.002, 0.025)  # 0.2% retorno, 2.5% vol
        historico_agressivo.append(historico_agressivo[-1] * (1 + r))

    # Estratégia CONSERVADORA: menor volatilidade, retorno um pouco menor
    np.random.seed(seed + 1)
    historico_conservador = [valor_inicial]
    for _ in range(n_dias):
        r = np.random.normal(0.0015, 0.008)  # 0.15% retorno, 0.8% vol
        historico_conservador.append(historico_conservador[-1] * (1 + r))

    metr_agr = calcular_sharpe_janela(historico_agressivo, taxa_livre_risco)
    metr_con = calcular_sharpe_janela(historico_conservador, taxa_livre_risco)

    print(f"\n  {'Métrica':<22} {'Agressiva':>12} {'Conservadora':>14}")
    print("  " + "-" * 50)
    print(f"  {'Retorno médio/dia':<22} {metr_agr['retorno_medio']:>11.4%} "
          f"{metr_con['retorno_medio']:>13.4%}")
    print(f"  {'Volatilidade/dia':<22} {metr_agr['volatilidade']:>11.4%} "
          f"{metr_con['volatilidade']:>13.4%}")
    print(f"  {'Sharpe (reward RL)':<22} {metr_agr['sharpe']:>12.3f} "
          f"{metr_con['sharpe']:>14.3f}")
    print(f"  {'Valor final':<22} R${historico_agressivo[-1]:>9,.2f} "
          f"R${historico_conservador[-1]:>11,.2f}")

    vencedor = "Conservadora" if metr_con['sharpe'] > metr_agr['sharpe'] else "Agressiva"
    print(f"\n  Estratégia com maior Sharpe: {vencedor}")
    print("  O agente RL aprende a preferir a estratégia de maior Sharpe.")


if __name__ == "__main__":
    print("=" * 60)
    print("GO1836 - RECOMPENSA BASEADA EM INDICE DE SHARPE")
    print("=" * 60)

    print("\nFORMULA:")
    print()
    print("  retorno = valor_hoje / valor_ontem - 1")
    print()
    print("  reward = (retorno - taxa_livre_risco) / volatilidade")
    print()
    print("  Interpretação:")
    print("    + reward alto  → retorno bom E baixo risco")
    print("    + reward baixo → retorno fraco OU alto risco")
    print("    + taxa_livre_risco: benchmark (Selic, CDI, T-bill)")
    print("    + volatilidade: desvio padrão dos retornos recentes")

    print()
    print("─" * 60)
    print("CENARIOS NUMERICOS:")
    print("─" * 60)

    # Parâmetros comuns
    TAXA = 0.0004  # Selic diária (~10% a.a.)

    cenarios = [
        # (descricao, retorno, volatilidade)
        ("Alto retorno, baixo risco",     0.015, 0.005),
        ("Alto retorno, alto risco",      0.015, 0.030),
        ("Baixo retorno, baixo risco",    0.002, 0.003),
        ("Abaixo da taxa livre de risco", 0.0002, 0.005),
        ("Prejuízo (retorno negativo)",  -0.010, 0.015),
    ]

    print(f"\n  {'Cenário':<35} {'Retorno':>8} {'Vol':>8} {'Sharpe':>8}")
    print("  " + "-" * 62)
    for desc, retorno, vol in cenarios:
        sharpe = calcular_reward_sharpe(retorno, TAXA, vol)
        print(f"  {desc:<35} {retorno:>7.3%} {vol:>7.3%} {sharpe:>+8.3f}")

    print()
    print("─" * 60)
    print("COMPARACAO DE ESTRATEGIAS (30 DIAS):")
    print("─" * 60)
    simular_portfolio_rl(n_dias=30)

    print()
    print("─" * 60)
    print("POR QUE SHARPE COMO REWARD?")
    print("─" * 60)
    print("  + Evita que o agente assuma riscos excessivos por retorno")
    print("  + Normaliza retornos por volatilidade — mais robusto")
    print("  + Alinha com gestão profissional de portfólio")
    print("  - Sharpe assume distribuição normal (retornos têm cauda gorda)")
    print("  - Sensível ao período de lookback para calcular volatilidade")
    print("  - Alternativas: Sortino (penaliza só volatilidade negativa),")
    print("                  Calmar (usa max drawdown)")
