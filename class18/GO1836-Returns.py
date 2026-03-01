# GO1836-Returns

if __name__ == "__main__":
    returns = portfolio_value_today / portfolio_value_yesterday - 1
    reward = (returns - risk_free_rate) / volatility
    # Maximiza retorno ajustado pelo risco
