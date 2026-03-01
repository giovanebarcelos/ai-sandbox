# GO1826-VariarParâmetrosNa
# Variar parâmetros na simulação para generalizar


if __name__ == "__main__":
    terrain_friction = np.random.uniform(0.5, 1.5)  # Variação de atrito
    terrain_height = np.random.uniform(-0.1, 0.1)   # Altura variável
    motor_noise = np.random.normal(0, 0.05)         # Ruído nos motores
