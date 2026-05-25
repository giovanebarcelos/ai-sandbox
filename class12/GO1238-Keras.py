# GO1238-Keras
from keras.models import Model
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

# Extrair saídas de cada camada


if __name__ == '__main__':
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    # Visualizar o que cada camada "vê"
    activations = activation_model.predict(img)

    # BLOCO 1 (conv1): Detecta bordas, cores básicas
    # BLOCO 2 (conv2): Texturas simples (listras, grades)
    # BLOCO 3 (conv3): Padrões complexos (olhos, rodas)
    # BLOCO 4 (conv4): Partes de objetos (faces, portas)
    # BLOCO 5 (conv5): Objetos completos (carros, cães)

    # ─── VISUALIZAÇÃO: HIERARQUIA DE ATIVAÇÕES POR BLOCO ───
    # Simular a "complexidade" das features ao longo dos blocos
    blocos = ['Bloco 1\n(conv1)', 'Bloco 2\n(conv2)', 'Bloco 3\n(conv3)',
              'Bloco 4\n(conv4)', 'Bloco 5\n(conv5)']
    n_filters    = [64, 128, 256, 512, 512]
    feature_size = [112, 56, 28, 14, 7]   # Dimensão espacial após pooling (input=224)
    complexity   = [1, 2, 4, 8, 16]        # Complexidade relativa das features
    sparsity     = [0.7, 0.75, 0.82, 0.88, 0.92]  # Esparsidade (% de ativações = 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Número de filtros por bloco
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
    bars = axes[0, 0].bar(blocos, n_filters, color=colors, edgecolor='black')
    axes[0, 0].set_ylabel('Número de Filtros')
    axes[0, 0].set_title('Número de Filtros por Bloco\n(cresce com a profundidade)', fontsize=11)
    for bar, n in zip(bars, n_filters):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                        str(n), ha='center', fontsize=10, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Dimensão espacial por bloco
    bars2 = axes[0, 1].bar(blocos, feature_size, color=plt.cm.Oranges(np.linspace(0.4, 0.9, 5)), edgecolor='black')
    axes[0, 1].set_ylabel('Dimensão Espacial (pixels)')
    axes[0, 1].set_title('Dimensão dos Feature Maps\n(reduz com pooling)', fontsize=11)
    for bar, s in zip(bars2, feature_size):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{s}×{s}', ha='center', fontsize=10, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Complexidade das features
    axes[1, 0].plot(blocos, complexity, 'o-', color='#e15759', linewidth=2.5, markersize=10)
    axes[1, 0].fill_between(range(5), complexity, alpha=0.15, color='#e15759')
    axes[1, 0].set_ylabel('Complexidade Relativa das Features')
    axes[1, 0].set_title('Complexidade das Features Detectadas\n'
                          'B1: bordas → B5: objetos completos', fontsize=11)
    features_desc = ['Bordas\nlinhas', 'Texturas\ngrad.', 'Padrões\nlocais', 'Partes de\nobjetos', 'Objetos\ncompletos']
    for i, (b, c, d) in enumerate(zip(blocos, complexity, features_desc)):
        axes[1, 0].annotate(d, xy=(i, c), xytext=(i, c + 1),
                             ha='center', fontsize=7.5, color='#e15759',
                             arrowprops=dict(arrowstyle='->', color='#e15759', lw=1))
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(range(5))
    axes[1, 0].set_xticklabels(blocos, fontsize=8)

    # Esparsidade das ativações
    axes[1, 1].bar(blocos, [s * 100 for s in sparsity],
                   color=plt.cm.Greens(np.linspace(0.4, 0.9, 5)), edgecolor='black')
    axes[1, 1].set_ylabel('Esparsidade (%)')
    axes[1, 1].set_title('Esparsidade das Ativações\n(camadas profundas são mais esparsas)', fontsize=11)
    axes[1, 1].set_ylim(0, 100)
    for i, s in enumerate(sparsity):
        axes[1, 1].text(i, s * 100 + 1, f'{s*100:.0f}%', ha='center', fontsize=10, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('VGG16 — Propriedades das Ativações por Bloco', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
