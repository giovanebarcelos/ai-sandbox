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
