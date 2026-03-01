# GO1818-31PortfolioManagement
import numpy as np

class PortfolioEnv:
    def __init__(self, price_history, initial_cash=100000):
        self.prices = price_history  # (T, N) - T dias, N ativos
        self.cash = initial_cash
        self.holdings = np.zeros(len(price_history[0]))
        self.current_step = 0

    def step(self, action):
        # action: [-1, 1] para cada ativo (quanto comprar/vender)
        current_prices = self.prices[self.current_step]

        # Executar trades
        for i, a in enumerate(action):
            if a > 0:  # Comprar
                shares_to_buy = (self.cash * a) / current_prices[i]
                self.holdings[i] += shares_to_buy
                self.cash -= shares_to_buy * current_prices[i]
            elif a < 0:  # Vender
                shares_to_sell = self.holdings[i] * abs(a)
                self.holdings[i] -= shares_to_sell
                self.cash += shares_to_sell * current_prices[i]

        # Avançar tempo
        self.current_step += 1
        new_prices = self.prices[self.current_step]

        # Calcular retorno
        portfolio_value = self.cash + np.sum(self.holdings * new_prices)
        previous_value = self.cash + np.sum(self.holdings * current_prices)
        returns = (portfolio_value / previous_value) - 1

        # Recompensa = Sharpe ratio simplificado
        reward = returns / 0.02  # Normalizado pela volatilidade média

        # Penalizar se ficar sem dinheiro
        if self.cash < 0:
            reward -= 10

        done = self.current_step >= len(self.prices) - 1
        state = self.get_state()

        return state, reward, done
