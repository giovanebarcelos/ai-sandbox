# GO0914-AlgoritmoTreinamentoCompleto
# IMPLEMENTAÇÃO DO ALGORITMO DE TREINAMENTO COMPLETO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def sigmoid(z):
    """Função de ativação sigmoid"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    """Derivada da sigmoid"""
    s = sigmoid(z)
    return s * (1 - s)

def initialize_weights(layer_sizes, activation='relu'):
    """
    Inicialização dos pesos (He ou Xavier)
    Args:
        layer_sizes: lista com número de neurônios por camada
        activation: 'relu' para He, 'sigmoid'/'tanh' para Xavier
    Returns:
        weights: lista de matrizes de pesos
        biases: lista de vetores de bias
    """
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i + 1]

        # He initialization para ReLU, Xavier para sigmoid/tanh
        if activation == 'relu':
            std = np.sqrt(2.0 / n_in)  # He initialization
        else:
            std = np.sqrt(2.0 / (n_in + n_out))  # Xavier

        W = np.random.randn(n_in, n_out) * std
        b = np.zeros((1, n_out))
        weights.append(W)
        biases.append(b)
    return weights, biases

def forward_propagation(X, weights, biases, activation_fn=sigmoid):
    """
    Forward propagation através da rede
    Args:
        X: dados de entrada (m, n_features)
        weights: lista de matrizes de pesos
        biases: lista de vetores de bias
        activation_fn: função de ativação
    Returns:
        activations: lista de ativações por camada
        z_values: lista de pré-ativações por camada
    """
    activations = [X]  # a[0] = input
    z_values = []

    a = X
    for W, b in zip(weights, biases):
        z = np.dot(a, W) + b       # z = W·a + b
        a = activation_fn(z)       # a = f(z)
        z_values.append(z)
        activations.append(a)

    return activations, z_values

def backward_propagation(y, activations, z_values, weights, activation_derivative=sigmoid_derivative):
    """
    Backward propagation para calcular gradientes
    Args:
        y: labels verdadeiros
        activations: ativações do forward pass
        z_values: pré-ativações do forward pass
        weights: pesos da rede
        activation_derivative: derivada da função de ativação
    Returns:
        dW: gradientes dos pesos
        db: gradientes dos bias
    """
    m = y.shape[0]
    num_layers = len(weights)

    dW = [None] * num_layers
    db = [None] * num_layers

    # Camada de saída: δ^(L) = ŷ - y (para MSE + sigmoid)
    delta = activations[-1] - y

    # Backprop através das camadas
    for l in reversed(range(num_layers)):
        # Gradientes dos parâmetros
        dW[l] = np.dot(activations[l].T, delta) / m
        db[l] = np.sum(delta, axis=0, keepdims=True) / m

        # Propagar erro para camada anterior
        if l > 0:
            delta = np.dot(delta, weights[l].T) * activation_derivative(z_values[l-1])

    return dW, db

def update_parameters(weights, biases, dW, db, learning_rate):
    """
    Atualiza parâmetros usando gradiente descendente
    Args:
        weights, biases: parâmetros atuais
        dW, db: gradientes
        learning_rate: taxa de aprendizado
    Returns:
        weights, biases: parâmetros atualizados
    """
    for l in range(len(weights)):
        weights[l] -= learning_rate * dW[l]
        biases[l] -= learning_rate * db[l]
    return weights, biases

def train_neural_network(X, y, layer_sizes, learning_rate=0.1, n_epochs=1000, batch_size=32, verbose=True):
    """
    Algoritmo completo de treinamento
    Args:
        X: features de treino (m, n_features)
        y: labels de treino (m, n_outputs)
        layer_sizes: arquitetura [n_input, n_hidden1, ..., n_output]
        learning_rate: taxa de aprendizado (η)
        n_epochs: número de épocas
        batch_size: tamanho do mini-batch
        verbose: imprimir progresso
    Returns:
        weights, biases: parâmetros otimizados
        losses: histórico de loss
    """
    # 1. INICIALIZAÇÃO
    weights, biases = initialize_weights(layer_sizes)
    losses = []
    m = X.shape[0]

    # 2. LOOP DE TREINAMENTO
    for epoch in range(n_epochs):
        # 2.1. Embaralhar dataset
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0
        n_batches = 0

        # 2.2 & 2.3. Dividir em mini-batches e processar
        for start_idx in range(0, m, batch_size):
            end_idx = min(start_idx + batch_size, m)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # FORWARD PROPAGATION
            activations, z_values = forward_propagation(X_batch, weights, biases)
            y_pred = activations[-1]

            # CALCULAR LOSS (MSE)
            batch_loss = np.mean((y_pred - y_batch)**2)
            epoch_loss += batch_loss
            n_batches += 1

            # BACKWARD PROPAGATION
            dW, db = backward_propagation(y_batch, activations, z_values, weights)

            # ATUALIZAR PARÂMETROS
            weights, biases = update_parameters(weights, biases, dW, db, learning_rate)

        # Calcular loss médio da época
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        # Log
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Época {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

    return weights, biases, losses

# EXEMPLO DE USO - XOR
if __name__ == "__main__":
    print("="*60)
    print("TREINANDO REDE NEURAL - XOR")
    print("="*60)

    # Dataset XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Treinar rede: 2 inputs → 4 hidden → 1 output
    weights, biases, losses = train_neural_network(
        X, y,
        layer_sizes=[2, 4, 1],
        learning_rate=0.5,
        n_epochs=5000,
        batch_size=4,
        verbose=True
    )

    # Testar
    activations, _ = forward_propagation(X, weights, biases)
    predictions = activations[-1]

    print("\nResultados:")
    print("x1  x2  | y_true | y_pred")
    print("-" * 35)
    for xi, yi, pred in zip(X, y, predictions):
        print(f"{xi[0]:2.0f}  {xi[1]:2.0f}  |   {yi[0]}    | {pred[0]:.4f}")

    print(f"\n✅ Treinamento concluído! Loss final: {losses[-1]:.6f}")

    # ── VISUALIZAÇÕES ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("GO0914 – Rede Neural: XOR", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # 1. CURVA DE APRENDIZADO ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(losses, color='steelblue', linewidth=1.5)
    ax1.set_title("Curva de Aprendizado")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.annotate(
        f"Loss final\n{losses[-1]:.6f}",
        xy=(len(losses) - 1, losses[-1]),
        xytext=(len(losses) * 0.6, losses[0] * 0.3),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9, color='steelblue'
    )

    # 2. FRONTEIRA DE DECISÃO ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    res = 300
    xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, res), np.linspace(-0.2, 1.2, res))
    grid = np.c_[xx.ravel(), yy.ravel()]
    activations_grid, _ = forward_propagation(grid, weights, biases)
    Z = activations_grid[-1].reshape(xx.shape)

    ax2.contourf(xx, yy, Z, levels=50, cmap='RdYlGn', alpha=0.8)
    ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5, linestyles='--')

    cores = ['tomato' if yi[0] == 0 else 'seagreen' for yi in y]
    rotulos = ['Classe 0' if yi[0] == 0 else 'Classe 1' for yi in y]
    for xi, ci, ri in zip(X, cores, rotulos):
        ax2.scatter(xi[0], xi[1], color=ci, s=150, edgecolors='black', linewidths=1.5, zorder=5)

    # Legenda manual
    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(color='seagreen', label='Classe 1 (saída=1)'),
        Patch(color='tomato',   label='Classe 0 (saída=0)'),
    ], fontsize=8, loc='upper right')

    ax2.set_title("Fronteira de Decisão – XOR")
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")

    # Anotar predições em cada ponto
    for xi, pred in zip(X, predictions):
        ax2.annotate(f"{pred[0]:.2f}", xy=(xi[0], xi[1]),
                     xytext=(xi[0] + 0.06, xi[1] + 0.06), fontsize=8)

    plt.savefig("GO0914-graficos.png", dpi=150, bbox_inches='tight')
    print("📊 Gráficos salvos em GO0914-graficos.png")
    plt.show()
