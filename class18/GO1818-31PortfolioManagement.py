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


if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)

    print("=== Simulação de Gestão de Portfólio com RL ===")

    # Gerar histórico de preços sintético (10 dias, 3 ativos)
    T, N = 10, 3
    price_history = np.cumprod(
        np.vstack([[100.0, 50.0, 200.0],
                   1 + np.random.randn(T - 1, N) * 0.02]),
        axis=0
    )

    env = PortfolioEnv(price_history=price_history, initial_cash=100_000)

    # Implementar get_state() ausente no código original
    env.get_state = lambda: np.append(env.holdings, env.cash)

    print(f"  Cash inicial: R${env.cash:,.2f}")
    print(f"  Ativos: {N}, Dias disponíveis: {T}")
    print()

    total_reward = 0
    for dia in range(T - 1):
        # Política ingênua: distribuir igualmente entre todos os ativos
        action = np.ones(N) * 0.1   # Comprar 10% do cash em cada ativo
        state, reward, done = env.step(action)
        total_reward += reward
        valor = env.cash + np.sum(env.holdings * price_history[env.current_step])
        print(f"  Dia {dia + 1}: valor_portfólio=R${valor:,.2f}, reward={reward:+.4f}")
        if done:
            break

    print(f"\n  Recompensa acumulada: {total_reward:.4f}")
