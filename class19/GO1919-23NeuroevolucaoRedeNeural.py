# GO1919-23NeuroevoluçãoRedeNeural
import numpy as np
import gymnasium as gym

# Rede Neural simples
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Número total de pesos
        self.n_weights = (input_size * hidden_size + hidden_size +
                          hidden_size * output_size + output_size)

    def set_weights(self, weights):
        """Decodificar cromossomo flat para pesos da rede"""
        idx = 0
        # Camada 1: input -> hidden
        self.W1 = np.array(weights[idx:idx + self.input_size * self.hidden_size])
        self.W1 = self.W1.reshape(self.input_size, self.hidden_size)
        idx += self.input_size * self.hidden_size

        self.b1 = np.array(weights[idx:idx + self.hidden_size])
        idx += self.hidden_size

        # Camada 2: hidden -> output
        self.W2 = np.array(weights[idx:idx + self.hidden_size * self.output_size])
        self.W2 = self.W2.reshape(self.hidden_size, self.output_size)
        idx += self.hidden_size * self.output_size

        self.b2 = np.array(weights[idx:idx + self.output_size])

    def forward(self, x):
        """Forward pass"""
        # Hidden layer com tanh
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        # Output layer
        out = np.dot(h, self.W2) + self.b2
        return out

    def predict(self, x):
        """Prever ação (0 ou 1)"""
        out = self.forward(x)
        return np.argmax(out)

# Fitness: avaliar rede no CartPole
def evaluate_network(weights, n_episodes=3):
    """Rodar episódios e retornar recompensa média"""
    env = gym.make('CartPole-v1')
    nn = SimpleNN(input_size=4, hidden_size=8, output_size=2)
    nn.set_weights(weights)

    total_reward = 0
    for episode in range(n_episodes):
        state = env.reset()[0]
        episode_reward = 0

        for step in range(500):
            action = nn.predict(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_reward += episode_reward

    env.close()
    return total_reward / n_episodes

# AG para neuroevolução
INPUT_SIZE = 4
HIDDEN_SIZE = 8
OUTPUT_SIZE = 2
nn_temp = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
N_WEIGHTS = nn_temp.n_weights

POP_SIZE = 50
GENERATIONS = 50
MUTATION_RATE = 0.1
MUTATION_SIGMA = 0.5

# População inicial: pesos aleatórios em [-1, 1]
population = [np.random.uniform(-1, 1, N_WEIGHTS).tolist() for _ in range(POP_SIZE)]

best_fitness_history = []

for generation in range(GENERATIONS):
    # Avaliar (paralelizar se possível)
    print(f"Geração {generation}: Avaliando população...")
    fitness_values = [evaluate_network(ind) for ind in population]

    best_fitness = max(fitness_values)
    avg_fitness = np.mean(fitness_values)
    best_fitness_history.append(best_fitness)

    print(f"  Melhor fitness: {best_fitness:.2f}, Médio: {avg_fitness:.2f}")

    # Seleção por Torneio
    selected = []
    for _ in range(POP_SIZE):
        competitors = np.random.choice(len(population), 3, replace=False)
        winner = max(competitors, key=lambda i: fitness_values[i])
        selected.append(np.array(population[winner]))

    # Crossover (blend)
    offspring = []
    for i in range(0, POP_SIZE, 2):
        if i + 1 < POP_SIZE and np.random.random() < 0.7:
            alpha = 0.5
            child1 = alpha * selected[i] + (1 - alpha) * selected[i+1]
            child2 = (1 - alpha) * selected[i] + alpha * selected[i+1]
            offspring.extend([child1, child2])
        else:
            offspring.extend([selected[i], selected[i+1] if i+1 < POP_SIZE else selected[i]])

    # Mutação gaussiana
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            if np.random.random() < MUTATION_RATE:
                offspring[i][j] += np.random.normal(0, MUTATION_SIGMA)
                offspring[i][j] = np.clip(offspring[i][j], -2, 2)  # Limitar

    # Elitismo: manter melhor
    best_idx = np.argmax(fitness_values)
    offspring[0] = np.array(population[best_idx])

    population = [ind.tolist() for ind in offspring]

# Melhor rede final
fitness_values = [evaluate_network(ind) for ind in population]
best_idx = np.argmax(fitness_values)
best_weights = population[best_idx]
best_fitness = fitness_values[best_idx]

print(f"\n🏆 MELHOR REDE NEURAL:")
print(f"Fitness médio: {best_fitness:.2f} steps")

# Testar visualmente
env = gym.make('CartPole-v1', render_mode='human')
nn = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
nn.set_weights(best_weights)

for test_episode in range(3):
    state = env.reset()[0]
    total_reward = 0

    for step in range(500):
        action = nn.predict(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()

        if terminated or truncated:
            break

    print(f"Test Episode {test_episode}: {total_reward} steps")

env.close()
