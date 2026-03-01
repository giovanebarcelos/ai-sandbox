# GO0929-CheckActivations
def check_activations(model, X):
    """Verificar distribuição das ativações"""
    model.forward(X)
    for i, a in enumerate(model.activations):
        mean = np.mean(a)
        std = np.std(a)
        pct_zero = np.mean(a == 0) * 100
        print(f"Layer {i}: mean={mean:.3f}, std={std:.3f}, "
              f"dead={pct_zero:.1f}%")


if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)

    print("=== Diagnóstico de Ativações ===")

    class DummyModel:
        """Modelo fictício com ativações ReLU simuladas"""
        def __init__(self):
            self.activations = []

        def forward(self, X):
            # Simular 3 camadas com diferentes problemas
            layer1 = np.maximum(0, np.random.randn(X.shape[0], 16))   # Saudável
            layer2 = np.maximum(0, np.random.randn(X.shape[0], 8) - 2) # Muitos mortos
            layer3 = np.random.randn(X.shape[0], 4) * 10               # Explodindo
            self.activations = [layer1, layer2, layer3]

    model = DummyModel()
    X = np.random.randn(50, 4)
    check_activations(model, X)
