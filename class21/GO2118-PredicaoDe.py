# GO2118-PrediçãoDe
# ═══════════════════════════════════════════════════════════════════
# XAI APLICADO: "Por que o modelo prevê falha na turbina de avião?"
# Manutenção preditiva — SHAP mostra quanto cada sensor contribui para
# o risco de falha, tornando a decisão auditável para a engenharia.
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

# ─── 1. PONTO-CHAVE: contribuição SHAP de cada sensor ───
# Cada leitura empurra o risco de falha para cima (+) ou para baixo (-)
risco_base = 0.10  # risco médio de falha de uma turbina saudável = 10%

contribuicoes = {
    "Temperatura óleo (142°C)": +0.35,   # +35% risco
    "Vibração anormal (0.8mm/s)": +0.28, # +28% risco
    "Horas de voo (12.340h)":   +0.15,   # +15% risco
    "Pressão dentro do normal": -0.08,   # -8% risco (fator a favor)
}

# ─── 2. PONTO-CHAVE: risco final acumulado ───
risco_final = risco_base + sum(contribuicoes.values())
acao = "🛑 MANUTENÇÃO IMEDIATA" if risco_final > 0.5 else "✅ OPERAÇÃO NORMAL"

print("=== Predição de falha em turbina (explicação SHAP) ===")
print(f"Risco base:   {risco_base:.0%}")
for sensor, valor in contribuicoes.items():
    seta = "↑" if valor > 0 else "↓"
    print(f"  {seta} {sensor}: {valor:+.0%}")
print(f"Risco final:  {risco_final:.0%}  →  {acao}")

# ─── 3. GRÁFICO: contribuição de cada sensor ───
sensores = list(contribuicoes.keys())
valores = np.array(list(contribuicoes.values()))

ordem = np.argsort(np.abs(valores))
sensores = [sensores[i] for i in ordem]
valores = valores[ordem]
cores = ['#d62728' if v > 0 else '#2ca02c' for v in valores]

plt.figure(figsize=(10, 5))
plt.barh(sensores, valores * 100, color=cores)
plt.axvline(0, color='black', linewidth=0.8)
for i, v in enumerate(valores):
    plt.text(v * 100 + (1 if v > 0 else -1), i, f"{v:+.0%}",
             va='center', ha='left' if v > 0 else 'right', fontsize=10)
plt.xlabel('Contribuição para o risco de falha (%)')
plt.title(f'Manutenção Preditiva — Risco de falha: {risco_final:.0%}',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print("✅ Gráfico de explicação gerado.")
