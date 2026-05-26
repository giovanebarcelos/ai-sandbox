# GO0919-VerificarShapes
# ═══════════════════════════════════════════════════════════════════
# VERIFICAÇÃO DE SHAPES — CHECKLIST DE DEBUGGING
# Slide 25: Debugar Redes Neurais
# ═══════════════════════════════════════════════════════════════════
"""
Verificar shapes dos tensores é o primeiro passo no debugging.
Shapes errados causam erros silenciosos ou exceções confusas.
"""

import numpy as np


def verificar_shapes_mnist(X_train, y_train_oh, X_val, y_val_oh, X_test):
    """
    Verifica shapes esperados para a atividade MNIST.
    Lança AssertionError se algo estiver errado.
    """
    print("=" * 55)
    print("VERIFICAÇÃO DE SHAPES")
    print("=" * 55)

    # 1. Verificar shapes de entrada
    print(f"X_train shape: {X_train.shape}")       # Deve ser (N, 784)
    print(f"y_train_oh shape: {y_train_oh.shape}") # Deve ser (N, 10)
    print(f"X_val shape:   {X_val.shape}")
    print(f"y_val_oh shape: {y_val_oh.shape}")
    print(f"X_test shape:  {X_test.shape}")

    assert X_train.ndim == 2, "X_train deve ser 2D"
    assert X_train.shape[1] == 784, f"Esperado 784 features, got {X_train.shape[1]}"
    assert y_train_oh.ndim == 2, "y_train_oh deve ser 2D (one-hot)"
    assert y_train_oh.shape[1] == 10, "Deve ter 10 classes"
    assert X_train.shape[0] == y_train_oh.shape[0], "N amostras incompatível"

    # 2. Verificar faixa de valores
    print(f"\nX_train min/max: {X_train.min():.3f} / {X_train.max():.3f}")
    print(f"y_train_oh   soma por linha (deve ser 1): {y_train_oh.sum(axis=1)[:5]}")

    # 3. Verificar balanceamento
    labels = y_train_oh.argmax(axis=1)
    classes, counts = np.unique(labels, return_counts=True)
    print(f"\nDistribuição classes: {dict(zip(classes.tolist(), counts.tolist()))}")
    assert len(classes) == 10, "Deve ter exatamente 10 classes"

    print("\n✅ Todos os shapes estão corretos!")


def verificar_shapes_rede(Ws, bs, layer_sizes):
    """Verifica shapes dos pesos da rede."""
    print("\n" + "=" * 55)
    print("VERIFICAÇÃO DE SHAPES DA REDE")
    print("=" * 55)
    for i, (W, b) in enumerate(zip(Ws, bs)):
        esperado_W = (layer_sizes[i], layer_sizes[i+1])
        esperado_b = (1, layer_sizes[i+1])
        status_W = "✅" if W.shape == esperado_W else "❌"
        status_b = "✅" if b.shape == esperado_b else "❌"
        print(f"  Camada {i}: W{W.shape} {status_W}  b{b.shape} {status_b}"
              f"  (esperado W{esperado_W})")


if __name__ == "__main__":
    # Dados sintéticos para demonstração
    np.random.seed(42)
    N = 200
    X_train = np.random.randn(N, 784)
    y_labels = np.random.randint(0, 10, N)
    y_train_oh = np.eye(10)[y_labels]
    X_val = np.random.randn(50, 784)
    y_val_oh = np.eye(10)[np.random.randint(0, 10, 50)]
    X_test = np.random.randn(50, 784)

    verificar_shapes_mnist(X_train, y_train_oh, X_val, y_val_oh, X_test)

    # Verificar pesos de rede [784 → 128 → 64 → 10]
    layer_sizes = [784, 128, 64, 10]
    Ws = [np.random.randn(layer_sizes[i], layer_sizes[i+1])
          for i in range(len(layer_sizes)-1)]
    bs = [np.zeros((1, layer_sizes[i+1]))
          for i in range(len(layer_sizes)-1)]
    verificar_shapes_rede(Ws, bs, layer_sizes)

    # Demonstrar erro de shape proposital
    print("\n--- Exemplo de shape ERRADO ---")
    Ws_errado = Ws.copy()
    Ws_errado[0] = np.random.randn(128, 784)  # transposta errada!
    verificar_shapes_rede(Ws_errado, bs, layer_sizes)
