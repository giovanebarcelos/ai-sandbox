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
