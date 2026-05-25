# GO1232-SemSaturaçãoPara
# Sem saturação para valores positivos:
# relu(x)  = max(0, x)  →  relu(-5) = 0  (gradiente=0, ok), relu(5) = 5 (gradiente=1, ótimo!)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

x = np.linspace(-4, 4, 400)

# ─── Variantes de ReLU ───
relu        = np.maximum(0, x)           # f(x) = max(0,x)  — gradiente: 0 ou 1
leaky_relu  = np.where(x >= 0, x, 0.1 * x)  # f(x<0) = 0.1x   — evita dead neurons
# ELU: f(x<0) = α·(e^x - 1)  — saídas negativas com média zero → acelera convergência
elu         = np.where(x >= 0, x, 1.0 * (np.exp(x) - 1))
# Swish (Google, 2017): f(x) = x·σ(x)  — auto-gated, supera ReLU em redes profundas
# Diferenciável em todo ponto; levemente negativo para x próximo de -1
swish       = x * (1 / (1 + np.exp(-x)))
# GELU (Gaussian Error Linear Unit): f(x) = x·Φ(x) onde Φ é CDF da normal
# Approximação: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
# Usado no BERT, GPT e outros Transformers — suave e estocástico
gelu        = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# ─── Gradientes analíticos ───
# ReLU: gradiente=1 para x>0 (sem saturação!), =0 para x<0 (dead neuron problem)
relu_grad       = np.where(x >= 0, 1.0, 0.0)
# Leaky ReLU: gradiente=0.1 para x<0 → neurônios nunca morrem completamente
leaky_relu_grad = np.where(x >= 0, 1.0, 0.1)
# ELU: gradiente=e^x para x<0 → suave e sempre positivo
elu_grad        = np.where(x >= 0, 1.0, np.exp(np.minimum(x, 0)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ativações
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
for fn, name, c in zip([relu, leaky_relu, elu, swish, gelu],
                        ['ReLU', 'Leaky ReLU (α=0.1)', 'ELU', 'Swish', 'GELU'], colors):
    axes[0].plot(x, fn, linewidth=2.5, label=name, color=c)
axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].axvline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_title('Funções de Ativação baseadas em ReLU', fontsize=12)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].legend(fontsize=9)
axes[0].set_ylim(-1.5, 4)
axes[0].grid(True, alpha=0.3)

# Gradientes
for fn, name, c in zip([relu_grad, leaky_relu_grad, elu_grad],
                        ["ReLU' (0 ou 1)", "Leaky ReLU' (0.1 ou 1)", "ELU'"], colors[:3]):
    axes[1].plot(x, fn, linewidth=2.5, label=name, color=c)
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].fill_between(x, relu_grad, where=(x < 0), alpha=0.15, color='red', label='Dead neuron (ReLU)')
axes[1].set_title('Gradientes — Sem Saturação Positiva!', fontsize=12)
axes[1].set_xlabel('x')
axes[1].set_ylabel("f'(x)")
axes[1].legend(fontsize=9)
axes[1].set_ylim(-0.2, 1.3)
axes[1].grid(True, alpha=0.3)
axes[1].annotate('Dead neuron:\ngrad=0 para x<0', xy=(-2, 0.05), fontsize=8, color='red')

plt.suptitle('ReLU e Variantes — Sem Saturação para Valores Positivos', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
