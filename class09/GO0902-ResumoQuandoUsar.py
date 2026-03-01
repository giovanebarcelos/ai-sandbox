# GO0902-ResumoQuandoUsar
# FUNÇÕES DE ATIVAÇÃO - IMPLEMENTAÇÃO E VISUALIZAÇÃO
import numpy as np
import matplotlib.pyplot as plt
# DEFINIR FUNÇÕES E SUAS DERIVADAS
def sigmoid(z):
    """Sigmoid: σ(z) = 1 / (1 + e^(-z))"""
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z):
    """Derivada: σ'(z) = σ(z) * (1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)
def tanh(z):
    """Tangente hiperbólica"""
    return np.tanh(z)
def tanh_derivative(z):
    """Derivada: tanh'(z) = 1 - tanh²(z)"""
    return 1 - np.tanh(z)**2
def relu(z):
    """ReLU: max(0, z)"""
    return np.maximum(0, z)
def relu_derivative(z):
    """Derivada: 1 se z > 0, senão 0"""
    return (z > 0).astype(float)
def leaky_relu(z, alpha=0.01):
    """Leaky ReLU: max(alpha*z, z)"""
    return np.where(z > 0, z, alpha * z)
def leaky_relu_derivative(z, alpha=0.01):
    """Derivada: 1 se z > 0, senão alpha"""
    return np.where(z > 0, 1, alpha)
# VISUALIZAR FUNÇÕES
z = np.linspace(-5, 5, 200)
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
# Sigmoid
axes[0, 0].plot(z, sigmoid(z), 'b-', linewidth=2)
axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].axvline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].set_title('Sigmoid', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylabel('f(z)')
axes[1, 0].plot(z, sigmoid_derivative(z), 'r-', linewidth=2)
axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].set_title('Sigmoid Derivative', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlabel('z')
axes[1, 0].set_ylabel("f'(z)")
# Tanh
axes[0, 1].plot(z, tanh(z), 'b-', linewidth=2)
axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].axvline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].set_title('Tanh', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)
axes[1, 1].plot(z, tanh_derivative(z), 'r-', linewidth=2)
axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_title('Tanh Derivative', fontsize=14)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlabel('z')
# ReLU
axes[0, 2].plot(z, relu(z), 'b-', linewidth=2)
axes[0, 2].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 2].axvline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 2].set_title('ReLU', fontsize=14)
axes[0, 2].grid(True, alpha=0.3)
axes[1, 2].plot(z, relu_derivative(z), 'r-', linewidth=2)
axes[1, 2].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 2].set_title('ReLU Derivative', fontsize=14)
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_xlabel('z')
# Leaky ReLU
axes[0, 3].plot(z, leaky_relu(z), 'b-', linewidth=2)
axes[0, 3].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 3].axvline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 3].set_title('Leaky ReLU', fontsize=14)
axes[0, 3].grid(True, alpha=0.3)
axes[1, 3].plot(z, leaky_relu_derivative(z), 'r-', linewidth=2)
axes[1, 3].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 3].set_title('Leaky ReLU Derivative', fontsize=14)
axes[1, 3].grid(True, alpha=0.3)
axes[1, 3].set_xlabel('z')
plt.suptitle('Funções de Ativação e suas Derivadas', fontsize=16)
plt.tight_layout()
plt.show()
# COMPARAR PROPRIEDADES
print("="*70)
print("COMPARAÇÃO DE FUNÇÕES DE ATIVAÇÃO")
print("="*70)
test_values = [-5, -1, 0, 1, 5]
print(f"\n{'z':>5} | {'Sigmoid':>8} | {'Tanh':>8} | {'ReLU':>8} | {'Leaky ReLU':>11}")
print("-" * 70)
for z_val in test_values:
    sig = sigmoid(np.array([z_val]))[0]
    tnh = tanh(np.array([z_val]))[0]
    rel = relu(np.array([z_val]))[0]
    lrel = leaky_relu(np.array([z_val]))[0]
    print(f"{z_val:5.1f} | {sig:8.4f} | {tnh:8.4f} | {rel:8.4f} | {lrel:11.4f}")
print("\n" + "="*70)
print("PROPRIEDADES")
print("="*70)
print("Sigmoid:     Range (0, 1),     Vanishing gradient")
print("Tanh:        Range (-1, 1),    Vanishing gradient, zero-centered")
print("ReLU:        Range [0, ∞),     Fast, Dying ReLU problem")
print("Leaky ReLU:  Range (-∞, ∞),    Solves dying ReLU")
print("\n✅ Funções de ativação implementadas e visualizadas!")
