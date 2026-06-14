# GO2117-ExplicarPor
# ═══════════════════════════════════════════════════════════════════
# XAI APLICADO: "Explicar POR QUE o crédito foi negado"
# Visualização tipo SHAP — cada feature empurra a decisão para mais
# risco (vermelho) ou menos risco (verde) em relação ao risco médio.
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

# ─── 1. PONTO-CHAVE: contribuições SHAP de cada fator ───
# Valor positivo = AUMENTA o risco de inadimplência (empurra para NEGAR)
# Valor negativo = DIMINUI o risco (empurra para APROVAR)
risco_base = 0.20  # risco médio do modelo (expected value) = 20%

contribuicoes = {
    "Idade baixa (23 anos)":   +0.15,   # +15% risco
    "Dívidas altas (R$ 18k)":  +0.28,   # +28% risco
    "Renda baixa (R$ 1.8k)":   +0.12,   # +12% risco
    "Histórico sem atrasos":   -0.05,   # -5% risco (fator a favor)
}

# ─── 2. PONTO-CHAVE: risco final = base + soma das contribuições ───
risco_final = risco_base + sum(contribuicoes.values())
decisao = "❌ CRÉDITO NEGADO" if risco_final > 0.5 else "✅ CRÉDITO APROVADO"

print("=== Por que o crédito foi negado? (explicação SHAP) ===")
print(f"Risco base (médio):  {risco_base:.0%}")
for fator, valor in contribuicoes.items():
    seta = "↑" if valor > 0 else "↓"
    print(f"  {seta} {fator}: {valor:+.0%}")
print(f"Risco final:         {risco_final:.0%}  →  {decisao}")

# ─── 3. GRÁFICO: barras horizontais ordenadas por impacto ───
fatores = list(contribuicoes.keys())
valores = np.array(list(contribuicoes.values()))

ordem = np.argsort(np.abs(valores))           # menor → maior impacto
fatores = [fatores[i] for i in ordem]
valores = valores[ordem]
cores = ['#d62728' if v > 0 else '#2ca02c' for v in valores]  # vermelho/verde

plt.figure(figsize=(10, 5))
plt.barh(fatores, valores * 100, color=cores)
plt.axvline(0, color='black', linewidth=0.8)
for i, v in enumerate(valores):
    plt.text(v * 100 + (1 if v > 0 else -1), i, f"{v:+.0%}",
             va='center', ha='left' if v > 0 else 'right', fontsize=10)
plt.xlabel('Contribuição para o risco de inadimplência (%)')
plt.title(f'Explicação da Decisão de Crédito  —  Risco final: {risco_final:.0%}',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print("✅ Gráfico de explicação gerado.")
