# GO1923B-CMAESControleRobotico
import numpy as np

def simulate_biped_robot(pid_params, duration=10.0, dt=0.01):
    """
    Simular robô bípede com controlador PID

    pid_params: [Kp, Ki, Kd] (3 gains)

    Objetivo: Manter ângulo torso vertical (0°) com distúrbios externos
    """
    Kp, Ki, Kd = pid_params

    # Estado: [θ, θ_dot] (ângulo torso, velocidade angular)
    theta = 0.1  # Desvio inicial 0.1 rad (~6°)
    theta_dot = 0.0
    integral = 0.0

    # Parâmetros físicos (simplificado)
    mass = 50  # kg
    length = 1.0  # m (centro massa)
    g = 9.81
    inertia = mass * length**2

    # Métricas
    total_error = 0.0
    total_torque = 0.0

    for t in np.arange(0, duration, dt):
        # Erro (queremos θ = 0)
        error = -theta
        integral += error * dt
        derivative = -theta_dot

        # Controle PID: torque
        torque = Kp * error + Ki * integral + Kd * derivative
        total_torque += abs(torque) * dt

        # Distúrbio externo (vento, empurrão)
        if 3.0 < t < 3.5:
            torque += 50 * np.sin(2 * np.pi * t)  # Perturbação

        # Dinâmica (equação movimento pêndulo invertido)
        theta_ddot = (torque - mass * g * length * np.sin(theta)) / inertia

        # Integrar (Euler simples)
        theta_dot += theta_ddot * dt
        theta += theta_dot * dt

        # Acumular erro
        total_error += abs(theta) * dt

        # Falha crítica: caiu (|θ| > 45°)
        if abs(theta) > 0.785:  # 45° em radianos
            total_error += 1000 * (duration - t)  # Penalidade grande
            break

    # Fitness: minimizar erro + penalizar torque excessivo (gasto energia)
    fitness = total_error + 0.01 * total_torque
    return fitness

# Otimizar PID com CMA-ES
import cma

# Espaço busca: Kp, Ki, Kd ∈ [0, 100]
x0 = [10, 1, 5]  # Chute inicial
sigma0 = 5.0

es_pid = cma.CMAEvolutionStrategy(x0, sigma0, {
    'bounds': [0, 100],
    'popsize': 20,
    'maxiter': 50,
    'verb_disp': 10
})

while not es_pid.stop():
    solutions = es_pid.ask()
    fitness_list = [simulate_biped_robot(sol) for sol in solutions]
    es_pid.tell(solutions, fitness_list)

best_pid = es_pid.result.xbest
best_fitness = es_pid.result.fbest

print(f"🤖 Controlador PID Ótimo (CMA-ES):")
print(f"  Kp = {best_pid[0]:.2f}")
print(f"  Ki = {best_pid[1]:.2f}")
print(f"  Kd = {best_pid[2]:.2f}")
print(f"  Fitness (erro acumulado): {best_fitness:.2f}")

# Testar com PID padrão (Kp=20, Ki=0, Kd=10)
baseline_fitness = simulate_biped_robot([20, 0, 10])
print(f"📊 Comparação:")
print(f"  CMA-ES: {best_fitness:.2f}")
print(f"  PID padrão: {baseline_fitness:.2f}")
print(f"  Melhoria: {(baseline_fitness - best_fitness) / baseline_fitness * 100:.1f}%")
