# GO1925B-NSGAIIPortfolioFinanceiro
# Pseudocódigo
def portfolio_optimization(stock_returns, covariance_matrix):
    """
    Otimizar alocação ativos

    x = [w1, w2, ..., wn] (pesos ativos, Σw=1)

    f1 = -E[return] = -Σ(w_i * μ_i)  (maximizar)
    f2 = risk = √(w^T * Σ * w)  (minimizar)
    """
    # NSGA-II encontra efficient frontier
    # Investidor conservador escolhe baixo risco
    # Investidor agressivo escolhe alto retorno
    pass
