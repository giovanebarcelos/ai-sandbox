# GO0925-InitWeights
def init_weights(n_in, n_out, activation='relu'):
    if activation == 'relu':
        # He initialization
        std = np.sqrt(2.0 / n_in)
    elif activation in ['sigmoid', 'tanh']:
        # Xavier initialization
        std = np.sqrt(2.0 / (n_in + n_out))
    else:
        std = 0.01

    W = np.random.randn(n_in, n_out) * std
    b = np.zeros((1, n_out))
    return W, b


if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)

    print("=== Inicialização de Pesos ===")
    configs = [
        (128, 64, 'relu'),
        (64, 32, 'sigmoid'),
        (32, 10, 'tanh'),
        (10, 5, 'linear'),
    ]
    for n_in, n_out, act in configs:
        W, b = init_weights(n_in, n_out, act)
        print(f"  {act:8s} ({n_in}→{n_out}): "
              f"std(W)={W.std():.5f}, shape W={W.shape}, b={b.shape}")
