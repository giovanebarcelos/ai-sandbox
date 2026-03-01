# GO0928-CheckGradients
def check_gradients(model):
    """Verificar se gradientes são saudáveis"""
    for i, grad in enumerate(model.gradients):
        mean = np.mean(np.abs(grad))
        std = np.std(grad)
        print(f"Layer {i}: mean={mean:.6f}, std={std:.6f}")
        if mean == 0:
            print(f"  ⚠️  WARNING: Zero gradient!")
        if mean > 100:
            print(f"  ⚠️  WARNING: Exploding gradient!")


if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)

    print("=== Diagnóstico de Gradientes ===")

    class DummyModel:
        """Modelo fictício para demonstração"""
        def __init__(self):
            self.gradients = [
                np.random.randn(10, 5) * 0.01,    # Gradiente saudável
                np.random.randn(5, 3) * 0.0001,   # Gradiente muito pequeno
                np.zeros((3, 2)),                  # Gradiente morto (zero)
                np.random.randn(2, 1) * 200,       # Gradiente explodindo
            ]

    model = DummyModel()
    check_gradients(model)
