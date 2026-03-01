# GO1908-10SeleçãoTorneioERoleta
import numpy as np

# Parâmetros
POP_SIZE = 100
CHROMOSOME_LENGTH = 20  # Precisão binária
GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01
X_MIN, X_MAX = -1, 2

# Fitness
def fitness(chromosome):
    x = decode_binary(chromosome, X_MIN, X_MAX)
    return x * np.sin(10 * np.pi * x) + 1.0

def decode_binary(chromosome, min_val, max_val):
    decimal = int(''.join(map(str, chromosome)), 2)
    max_decimal = 2**len(chromosome) - 1
    value = min_val + (max_val - min_val) * decimal / max_decimal
    return value

# População inicial
def create_population():
    return [np.random.randint(0, 2, CHROMOSOME_LENGTH).tolist() 
            for _ in range(POP_SIZE)]

# Avaliação
def evaluate_population(population):
    return [fitness(ind) for ind in population]

# Seleção por Torneio
def tournament_selection(population, fitness_values, k=3):
    """Selecionar indivíduo via torneio de k competidores"""
    selected = []
    for _ in range(len(population)):
        # Escolher k indivíduos aleatórios
        competitors = np.random.choice(len(population), k, replace=False)
        # Retornar o melhor
        winner = max(competitors, key=lambda i: fitness_values[i])
        selected.append(population[winner].copy())
    return selected

# Seleção por Roleta (Roulette Wheel)
def roulette_selection(population, fitness_values):
    """Probabilidade proporcional ao fitness"""
    # Fitness deve ser > 0, se negativo fazer shift
    min_fitness = min(fitness_values)
    if min_fitness < 0:
        adjusted_fitness = [f - min_fitness + 0.1 for f in fitness_values]
    else:
        adjusted_fitness = fitness_values

    total_fitness = sum(adjusted_fitness)
    probabilities = [f / total_fitness for f in adjusted_fitness]

    # Selecionar com probabilidade
    selected_indices = np.random.choice(len(population), 
                                        size=len(population), 
                                        p=probabilities)
    return [population[i].copy() for i in selected_indices]
