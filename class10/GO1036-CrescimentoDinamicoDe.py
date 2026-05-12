# GO1036-CrescimentoDinâmicoDe
# Configura o crescimento dinâmico de memória GPU para evitar pré-alocação total
# e, opcionalmente, limita a memória disponível para um dispositivo específico.
# Crescimento dinâmico de memória

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)

# Limitar memória GPU
# tf.config.set_logical_device_configuration(
#     gpus[0],
#     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
# )

if __name__ == "__main__":
    import tensorflow as tf
    import matplotlib
    import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

    # Lista GPUs disponíveis e imprime informações sobre os dispositivos físicos
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    print(f"CPUs: {len(cpus)}  |  GPUs: {len(gpus)}")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Crescimento dinâmico de memória ativado para todas as GPUs.")

    # Gráfico de barras mostrando "CPU: 1, GPU: 0" (ou N se disponível)
    fig, ax = plt.subplots(figsize=(5, 4))
    dispositivos = ['CPU', 'GPU']
    quantidades  = [len(cpus), len(gpus)]
    colors = ['#5DADE2', '#F39C12']
    bars = ax.bar(dispositivos, quantidades, color=colors)
    for bar, val in zip(bars, quantidades):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(val), ha='center', va='bottom', fontsize=12)
    ax.set_title('Dispositivos Físicos Disponíveis')
    ax.set_ylabel('Quantidade')
    ax.set_ylim(0, max(quantidades) + 1)
    plt.tight_layout()
    plt.show()
