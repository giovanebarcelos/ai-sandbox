# GO1921-ParticleSwarmOptimizationCompleta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PSO:
    def __init__(self, objective_func, bounds, n_particles=30, max_iter=100,
                 w=0.7, c1=1.5, c2=1.5, seed=42):
        """
        Particle Swarm Optimization

        Args:
            w: inércia (0.4-0.9) - peso velocidade anterior
            c1: coef. cognitivo - atração pbest
            c2: coef. social - atração gbest
        """
        self.objective = objective_func
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.n_dim = len(bounds)
        self.max_iter = max_iter
        self.w = w  # Inércia
        self.c1 = c1  # Cognitivo
        self.c2 = c2  # Social

        np.random.seed(seed)

        # Inicializar posições aleatórias
        self.positions = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (n_particles, self.n_dim)
        )

        # Inicializar velocidades (pequenas)
        v_max = 0.2 * (self.bounds[:, 1] - self.bounds[:, 0])
        self.velocities = np.random.uniform(-v_max, v_max, (n_particles, self.n_dim))

        # Avaliar fitness inicial
        self.fitness = np.array([objective_func(p) for p in self.positions])

        # Melhores posições pessoais
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = self.fitness.copy()

        # Melhor posição global
        self.gbest_idx = np.argmin(self.fitness)
        self.gbest_position = self.positions[self.gbest_idx].copy()
        self.gbest_fitness = self.fitness[self.gbest_idx]

        # Histórico
        self.history = []

    def optimize(self):
        """Executar PSO"""
        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                # Atualizar velocidade
                r1, r2 = np.random.rand(self.n_dim), np.random.rand(self.n_dim)

                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])

                self.velocities[i] = (
                    self.w * self.velocities[i] + cognitive + social
                )

                # Limitar velocidade (evitar explosão)
                v_max = 0.2 * (self.bounds[:, 1] - self.bounds[:, 0])
                self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)

                # Atualizar posição
                self.positions[i] += self.velocities[i]

                # Garantir limites (refletir ou clampar)
                self.positions[i] = np.clip(
                    self.positions[i], self.bounds[:, 0], self.bounds[:, 1]
                )

                # Avaliar novo fitness
                new_fitness = self.objective(self.positions[i])
                self.fitness[i] = new_fitness

                # Atualizar pbest
                if new_fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = new_fitness
                    self.pbest_positions[i] = self.positions[i].copy()

                    # Atualizar gbest
                    if new_fitness < self.gbest_fitness:
                        self.gbest_fitness = new_fitness
                        self.gbest_position = self.positions[i].copy()

            # Rastrear convergência
            self.history.append(self.gbest_fitness)

            # Decair inércia (opcional: exploração → exploitation)
            if iteration == self.max_iter // 2:
                self.w *= 0.7

            if iteration % 20 == 0:
                print(f"Iter {iteration:3d}: gbest = {self.gbest_fitness:.6f}")

        return self.gbest_position, self.gbest_fitness


# EXEMPLO: Otimizar função Ackley (multimodal)
def ackley(x):
    """f(x) = -20*exp(-0.2*√(Σx²/n)) - exp(Σcos(2πx)/n) + 20 + e
    Mínimo global: f(0,...,0) = 0
    """
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))

    return (-20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) 
            - np.exp(sum_cos / n) + 20 + np.e)

# Otimizar 5D
bounds = [(-5, 5)] * 5
pso = PSO(ackley, bounds, n_particles=40, max_iter=100, w=0.7, c1=1.5, c2=1.5)
best_pos, best_fit = pso.optimize()

print(f"\n🎯 PSO - Resultado:")
print(f"x = {best_pos}")
print(f"f(x) = {best_fit:.8f}")

# Comparar com solução aleatória
random_samples = np.random.uniform(-5, 5, (1000, 5))
random_fitness = [ackley(s) for s in random_samples]
print(f"📊 Comparação:")
print(f"  PSO: {best_fit:.6f}")
print(f"  Random best (1000 amostras): {min(random_fitness):.6f}")
print(f"  Melhoria: {(min(random_fitness) - best_fit) / min(random_fitness) * 100:.1f}%")

# Visualizar convergência
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(pso.history, linewidth=2, color='orange')
plt.axhline(0, color='green', linestyle='--', label='Ótimo global')
plt.xlabel('Iteração')
plt.ylabel('gbest Fitness')
plt.title('Convergência PSO - Ackley 5D')
plt.yscale('log')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
# Diversidade do enxame (dispersão)
distances = [np.std(pso.positions[:, 0]) for _ in range(len(pso.history))]
plt.plot(distances, linewidth=2, color='purple')
plt.xlabel('Iteração')
plt.ylabel('Dispersão Enxame (std)')
plt.title('Evolução Diversidade')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
