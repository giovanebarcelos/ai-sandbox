# GO1235-NormalizaRespostaBaseado
# Local Response Normalization (LRN): normaliza resposta baseado em canais vizinhos
# (Não mais usado - substituído por Batch Normalization)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

# ─── VISUALIZAÇÃO: LRN vs Batch Normalization ───
np.random.seed(0)
canais = 8
ativacoes_brutas = np.abs(np.random.randn(canais) * 3) + 0.5  # Ativações brutas (positivas)

# LRN: normaliza canal i pelos k vizinhos: a_i / (k + alpha * sum(a_{j}^2 for j in vizinhos))^beta
# Fórmula completa de Krizhevsky (AlexNet 2012):
#   b_i = a_i / (k + alpha * Σ_{j=max(0,i-n/2)}^{min(N-1,i+n/2)} a_j²) ^ beta
# Intuito: neurônio com forte resposta suprime os vizinhos (competição lateral)
def lrn(activations, k=2, alpha=0.0001, beta=0.75, n=3):
    result = np.zeros_like(activations)
    for i in range(len(activations)):
        # Seleciona n vizinhos ao redor do canal i (janela de normalização)
        neighbors = activations[max(0, i-n//2):min(len(activations), i+n//2+1)]
        # Normaliza: divide pela soma quadrática dos vizinhos elevada a beta
        result[i] = activations[i] / (k + alpha * np.sum(neighbors**2)) ** beta
    return result

# Batch Normalization: normaliza para media=0, std=1, depois aplica gamma e beta
# Passos: (1) μ = média do batch, (2) σ = desvio padrão, (3) x̂ = (x-μ)/σ
# (4) aplica parâmetros treináveis: y = gamma * x̂ + beta (aprende a escala ideal)
def batch_norm(activations, gamma=1.0, beta_param=0.0):
    mean = np.mean(activations)  # média de todas as ativações no batch
    std  = np.std(activations) + 1e-8  # +1e-8 evita divisão por zero
    # Normaliza e reescala com parâmetros treináveis gamma e beta
    return gamma * (activations - mean) / std + beta_param

lrn_out = lrn(ativacoes_brutas)
bn_out  = batch_norm(ativacoes_brutas)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
x_pos = np.arange(canais)

axes[0].bar(x_pos, ativacoes_brutas, color='#76b7b2', edgecolor='black')
axes[0].set_title('Ativações Brutas\n(após ReLU)', fontsize=12)
axes[0].set_xlabel('Canal')
axes[0].set_ylabel('Valor')
for i, v in enumerate(ativacoes_brutas):
    axes[0].text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=8)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(x_pos, lrn_out, color='#f28e2b', edgecolor='black')
axes[1].set_title('Após LRN\n(normaliza pelos vizinhos, n=3)', fontsize=12)
axes[1].set_xlabel('Canal')
axes[1].set_ylabel('Valor')
for i, v in enumerate(lrn_out):
    axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, 0.7)

axes[2].bar(x_pos, bn_out, color='#4e79a7', edgecolor='black')
axes[2].set_title('Após Batch Normalization\n(μ=0, σ=1, depois escala e deslocamento)',
                   fontsize=12)
axes[2].set_xlabel('Canal')
axes[2].set_ylabel('Valor')
axes[2].axhline(0, color='red', linestyle='--', linewidth=1, label='μ=0')
for i, v in enumerate(bn_out):
    axes[2].text(i, v + (0.05 if v >= 0 else -0.15), f'{v:.2f}', ha='center', fontsize=8)
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3, axis='y')

plt.suptitle('LRN (AlexNet, 2012) vs Batch Normalization (VGG/ResNet, 2015)\n'
             'BN substituiu LRN pois é mais eficaz e generalizável',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
