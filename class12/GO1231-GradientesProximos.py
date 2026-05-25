# GO1231-GradientesPróximos
# Saturação: gradientes próximos de 0
# tanh(-5) ≈ -0.9999  → gradiente ≈ 0  (PROBLEMA: vanishing gradient)
# tanh( 5) ≈  0.9999  → gradiente ≈ 0  (PROBLEMA: vanishing gradient)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

x = np.linspace(-6, 6, 300)

# Funções de ativação
tanh_y   = np.tanh(x)                   # tanh(x) = (e^x - e^-x)/(e^x + e^-x)
sigmoid_y = 1 / (1 + np.exp(-x))       # σ(x) = 1/(1+e^{-x})

# Derivadas analíticas (usadas na backpropagation)
# d/dx tanh(x) = 1 - tanh²(x)  →  máximo 1.0 em x=0, mas → 0 quando |x| > 2
tanh_grad    = 1 - tanh_y ** 2
# d/dx σ(x) = σ(x)·(1-σ(x))    →  máximo 0.25 em x=0 (já é menor que tanh!)
# Conclusão: em redes profundas, multiplicar N gradientes <1 → gradiente desaparece
sigmoid_grad = sigmoid_y * (1 - sigmoid_y)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ativações
axes[0].plot(x, tanh_y,    linewidth=2.5, color='#4e79a7', label='tanh(x)')
axes[0].plot(x, sigmoid_y, linewidth=2.5, color='#f28e2b', label='sigmoid(x)')
axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].axhline( 0.9999, color='#4e79a7', linewidth=1, linestyle=':', alpha=0.6, label='tanh saturado ≈ ±1')
axes[0].axhline(-0.9999, color='#4e79a7', linewidth=1, linestyle=':', alpha=0.6)
axes[0].axhline(0.9999, color='#f28e2b', linewidth=1, linestyle=':', alpha=0.6)
axes[0].fill_between(x, tanh_y, where=(x < -3), alpha=0.15, color='red', label='Zona saturada')
axes[0].fill_between(x, tanh_y, where=(x > 3),  alpha=0.15, color='red')
axes[0].set_title('Funções de Ativação com Saturação', fontsize=12)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-1.2, 1.2)

# Gradientes
axes[1].plot(x, tanh_grad,    linewidth=2.5, color='#4e79a7', label="tanh'(x) = 1 - tanh²(x)")
axes[1].plot(x, sigmoid_grad, linewidth=2.5, color='#f28e2b', label="sigmoid'(x)")
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].fill_between(x, tanh_grad, where=(np.abs(x) > 3), alpha=0.2, color='red', label='Gradiente ≈ 0 (vanishing)')
axes[1].fill_between(x, sigmoid_grad, where=(np.abs(x) > 3), alpha=0.2, color='red')
axes[1].set_title('Gradientes — Problema do Vanishing Gradient', fontsize=12)
axes[1].set_xlabel('x')
axes[1].set_ylabel("f'(x)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# Anotações nas zonas saturadas
for ax_i in [axes[0], axes[1]]:
    ax_i.annotate('Saturação\n(gradiente ≈ 0)', xy=(-4.5, -0.5 if ax_i == axes[0] else 0.02),
                  fontsize=8, color='red', ha='center')
    ax_i.annotate('Saturação\n(gradiente ≈ 0)', xy=(4.5, 0.5 if ax_i == axes[0] else 0.02),
                  fontsize=8, color='red', ha='center')

plt.suptitle('tanh e Sigmoid — Saturação e Vanishing Gradient', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
