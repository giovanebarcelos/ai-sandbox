# GO1036-CrescimentoDinâmicoDe
# Configura o crescimento dinâmico de memória GPU para evitar pré-alocação total
# e, opcionalmente, limita a memória disponível para um dispositivo específico.
# Crescimento dinâmico de memória

gpus = tf.config.list_physical_devices('GPU')

**📦 Instalação necessária:**

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Limitar memória GPU
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
)
