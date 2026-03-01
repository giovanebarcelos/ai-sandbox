# GO1818-30DataCenterRL
class DataCenterEnv:
    def __init__(self):
        self.server_temp = 25.0  # °C
        self.ambient_temp = 30.0
        self.cooling_power = 0.0

    def step(self, actions):
        # actions: [fan_speed_1, fan_speed_2, ..., valve_positions]

        # Física simplificada
        heat_generated = 100  # Servidores sempre geram calor
        heat_removed = sum(actions[:10]) * 5  # Ventiladores removem

        self.server_temp += (heat_generated - heat_removed) * 0.01

        # Energia gasta (ventiladores consomem energia!)
        energy = sum(actions[:10])**2 * 0.1

        # Recompensa: Balancear temperatura e energia
        reward = 0
        if 20 < self.server_temp < 27:  # Range ideal
            reward += 10
        else:
            reward -= abs(self.server_temp - 23.5) * 5  # Penaliza desvio

        reward -= energy  # Penaliza consumo

        # Falha se superaquecer
        done = self.server_temp > 35
        if done:
            reward -= 1000

        return self.get_state(), reward, done

# Algoritmo usado: Modelo-based RL (aprendeu física do datacenter)


if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)

    print("=== Simulação de Datacenter com RL ===")
    env = DataCenterEnv()

    # Implementar get_state() ausente no código original
    env.get_state = lambda: [env.server_temp, env.ambient_temp, env.cooling_power]
    print(f"  Temperatura inicial: {env.server_temp}°C")
    print()

    total_reward = 0
    for step in range(1, 16):
        # Política simples: ventiladores em velocidade ~média
        actions = [0.5] * 10 + [0.3] * 5  # velocidade dos ventiladores
        state, reward, done = env.step(actions)
        total_reward += reward
        print(f"  Passo {step:2d}: temp={env.server_temp:.2f}°C, "
              f"reward={reward:+.1f}, done={done}")
        if done:
            print("  ⚠️  Superaquecimento! Episódio encerrado.")
            break

    print(f"\n  Recompensa total: {total_reward:.1f}")
    print("  (Com RL treinado, o agente aprende a equilibrar temperatura e energia)")
