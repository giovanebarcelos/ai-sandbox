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
