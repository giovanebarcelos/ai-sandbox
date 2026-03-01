# GO0804-Utilidade
# COMPONENT PLANES - VISUALIZAÇÃO

def plot_component_planes(som, feature_names=None):
    """
    Plota component planes para cada dimensão da entrada
    """
    n_features = som.input_dim

    # Determinar layout de subplots
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(4*n_cols, 4*n_rows))

    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for feature_idx in range(n_features):
        ax = axes[feature_idx]

        # Extrair valores da feature de todos neurônios
        component_map = som.weights[:, :, feature_idx]

        # Plotar
        im = ax.imshow(component_map, cmap='coolwarm', 
                      interpolation='nearest')

        # Título
        if feature_names:
            title = f'{feature_names[feature_idx]}'
        else:
            title = f'Feature {feature_idx + 1}'
        ax.set_title(title, fontsize=12)

        # Colorbar
        plt.colorbar(im, ax=ax)

        ax.set_xlabel('Posição X')
        ax.set_ylabel('Posição Y')

    # Esconder axes extras
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle('Component Planes - Visualização por Feature', 
                fontsize=16, y=1.02)
    plt.show()

# Exemplo: Iris dataset
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    iris = load_iris()
    X_iris = StandardScaler().fit_transform(iris.data)

    # Treinar SOM
    som_iris = SimpleSOM(map_height=15, map_width=15, input_dim=4,
                        learning_rate=0.5, n_iterations=2000)
    som_iris.fit(X_iris, verbose=False)

    # Plotar component planes
    plot_component_planes(som_iris, 
                         feature_names=['Sepal Length', 'Sepal Width',
                                       'Petal Length', 'Petal Width'])

    print("\n💡 Component Planes do Iris:")
    print("   • Petal Length e Petal Width têm padrões similares")
    print("   • → Estas features são correlacionadas!")
    print("   • → Distinguem bem as espécies")
