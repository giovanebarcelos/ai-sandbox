# GO0102-AnaĺiseDeViésEmSistemasDeIA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# ═══════════════════════════════════════════════════════════════════
# ANÁLISE DE VIÉS EM SISTEMAS DE IA
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("DETECÇÃO E ANÁLISE DE VIÉS EM MODELOS DE IA")
print("="*70)
print("\nProblema: Algoritmos treinados com dados históricos")
print("podem perpetuar ou amplificar discriminações existentes.")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# CENÁRIO: SISTEMA DE APROVAÇÃO DE CRÉDITO
# ═══════════════════════════════════════════════════════════════════

print("\n🏦 CENÁRIO: Sistema de Aprovação de Crédito Bancário")
print("="*70)

# Simular dados de aprovação de crédito com viés
np.random.seed(42)

n_samples = 1000

# Gerar dados
generos = np.random.choice(['Masculino', 'Feminino'], n_samples, p=[0.6, 0.4])
idades = np.random.randint(18, 70, n_samples)
rendas = np.random.normal(5000, 2000, n_samples).clip(1000, 15000)
scores_credito = np.random.randint(300, 850, n_samples)

# VIÉS INTRODUZIDO: Mulheres precisam de score 50 pontos maior
aprovacao = []
for genero, idade, renda, score in zip(generos, idades, rendas, scores_credito):
    threshold = 650 if genero == 'Masculino' else 700  # VIÉS!

    if score >= threshold and renda >= 3000:
        aprovacao.append(1)  # Aprovado
    else:
        aprovacao.append(0)  # Negado

# Criar DataFrame
df = pd.DataFrame({
    'genero': generos,
    'idade': idades,
    'renda': rendas,
    'score_credito': scores_credito,
    'aprovado': aprovacao
})

# Converter gênero para numérico
df['genero_num'] = df['genero'].map({'Masculino': 1, 'Feminino': 0})

print("\n📊 ESTATÍSTICAS DESCRITIVAS:")
print(df.groupby('genero')['aprovado'].agg(['count', 'sum', 'mean']))

# Taxa de aprovação por gênero
taxa_homens = df[df['genero'] == 'Masculino']['aprovado'].mean()
taxa_mulheres = df[df['genero'] == 'Feminino']['aprovado'].mean()

print(f"\n📈 TAXAS DE APROVAÇÃO:")
print(f"   Homens: {taxa_homens*100:.1f}%")
print(f"   Mulheres: {taxa_mulheres*100:.1f}%")
print(f"   Disparidade: {(taxa_homens - taxa_mulheres)*100:.1f} pontos percentuais")

# ═══════════════════════════════════════════════════════════════════
# TREINAR MODELO DE ML (que aprenderá o viés)
# ═══════════════════════════════════════════════════════════════════

print("\n🤖 TREINANDO MODELO DE MACHINE LEARNING...")
print("="*70)

# Features e target
X = df[['genero_num', 'idade', 'renda', 'score_credito']]
y = df['aprovado']

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Treinar modelo
modelo = LogisticRegression(random_state=42, max_iter=1000)
modelo.fit(X_train, y_train)

# Predições
y_pred = modelo.predict(X_test)

print("\n📊 ACURÁCIA GERAL DO MODELO:")
print(f"   {modelo.score(X_test, y_test)*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════
# ANÁLISE DE VIÉS: MÉTRICAS POR GRUPO
# ═══════════════════════════════════════════════════════════════════

print("\n⚖️ ANÁLISE DE FAIRNESS (JUSTIÇA) POR GÊNERO:")
print("="*70)

# Separar por gênero
df_test = X_test.copy()
df_test['genero'] = df_test['genero_num'].map({1: 'Masculino', 0: 'Feminino'})
df_test['y_true'] = y_test.values
df_test['y_pred'] = y_pred

# Métricas por gênero
for genero_str in ['Masculino', 'Feminino']:
    genero_value = 1 if genero_str == 'Masculino' else 0
    mask = df_test['genero_num'] == genero_value

    y_true_grupo = df_test[mask]['y_true']
    y_pred_grupo = df_test[mask]['y_pred']

    taxa_predicao_positiva = y_pred_grupo.mean()
    taxa_real_positiva = y_true_grupo.mean()
    acuracia_grupo = (y_true_grupo == y_pred_grupo).mean()

    # True Positive Rate (Recall para aprovados)
    tp = ((y_true_grupo == 1) & (y_pred_grupo == 1)).sum()
    fn = ((y_true_grupo == 1) & (y_pred_grupo == 0)).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

    # False Positive Rate (aprovado indevidamente)
    fp = ((y_true_grupo == 0) & (y_pred_grupo == 1)).sum()
    tn = ((y_true_grupo == 0) & (y_pred_grupo == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n{genero_str.upper()}:")
    print(f"   Taxa de Predição Positiva: {taxa_predicao_positiva*100:.1f}%")
    print(f"   Acurácia: {acuracia_grupo*100:.1f}%")
    print(f"   True Positive Rate (Sensibilidade): {tpr*100:.1f}%")
    print(f"   False Positive Rate: {fpr*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════
# MÉTRICAS DE FAIRNESS
# ═══════════════════════════════════════════════════════════════════

print("\n📐 MÉTRICAS DE FAIRNESS (JUSTIÇA):")
print("="*70)

# 1. Statistical Parity (Demographic Parity)
taxa_pred_homens = df_test[df_test['genero'] == 'Masculino']['y_pred'].mean()
taxa_pred_mulheres = df_test[df_test['genero'] == 'Feminino']['y_pred'].mean()
disparate_impact = taxa_pred_mulheres / taxa_pred_homens if taxa_pred_homens > 0 else 0

print(f"\n1. STATISTICAL PARITY (Demographic Parity)")
print(f"   Taxa Homens: {taxa_pred_homens*100:.1f}%")
print(f"   Taxa Mulheres: {taxa_pred_mulheres*100:.1f}%")
print(f"   Disparate Impact Ratio: {disparate_impact:.2f}")
print(f"   Interpretação: {disparate_impact:.2f} < 0.8 → VIÉS DETECTADO! ❌" if disparate_impact < 0.8 else "   OK ✅")

# 2. Equalized Odds (TPR e FPR iguais entre grupos)
print(f"\n2. EQUALIZED ODDS")
print(f"   Verifica se TPR e FPR são similares entre grupos")
print(f"   (Implementação: calcular diferença de TPR/FPR acima)")

# 3. Análise dos coeficientes do modelo
print(f"\n3. IMPORTÂNCIA DAS FEATURES (Coeficientes do Modelo):")
feature_names = ['genero_num', 'idade', 'renda', 'score_credito']
coeficientes = modelo.coef_[0]

for nome, coef in zip(feature_names, coeficientes):
    print(f"   {nome}: {coef:.4f}")

print(f"\n   ⚠️ Coeficiente de 'genero_num' = {coeficientes[0]:.4f}")
if abs(coeficientes[0]) > 0.1:
    print(f"   Gênero está INFLUENCIANDO decisão! Viés confirmado. ❌")
else:
    print(f"   Gênero tem baixa influência. ✅")

# ═══════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════

print("\n📊 GERANDO VISUALIZAÇÕES...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Distribuição de Score por Gênero e Aprovação
ax1 = axes[0, 0]
for genero_val in ['Masculino', 'Feminino']:
    for aprovado_val in [0, 1]:
        data = df[(df['genero'] == genero_val) & (df['aprovado'] == aprovado_val)]['score_credito']
        label = f"{genero_val} - {'Aprovado' if aprovado_val else 'Negado'}"
        ax1.hist(data, alpha=0.5, bins=20, label=label)

ax1.set_xlabel("Score de Crédito", fontsize=11)
ax1.set_ylabel("Frequência", fontsize=11)
ax1.set_title("Distribuição de Score por Gênero e Decisão", fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Taxa de Aprovação por Gênero
ax2 = axes[0, 1]
taxas = [taxa_homens * 100, taxa_mulheres * 100]
generos_plot = ['Homens', 'Mulheres']
cores_bar = ['steelblue', 'coral']

bars = ax2.bar(generos_plot, taxas, color=cores_bar, alpha=0.7, edgecolor='black')
ax2.set_ylabel("Taxa de Aprovação (%)", fontsize=11)
ax2.set_title("Taxa de Aprovação por Gênero", fontsize=12, fontweight='bold')
ax2.set_ylim(0, 100)

# Adicionar valores nas barras
for bar, taxa in zip(bars, taxas):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{taxa:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Paridade (50%)')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()

# 3. Matriz de Confusão por Gênero
ax3 = axes[1, 0]
ax3.axis('off')

# Calcular confusion matrices
cm_homens = confusion_matrix(
    df_test[df_test['genero'] == 'Masculino']['y_true'],
    df_test[df_test['genero'] == 'Masculino']['y_pred']
)
cm_mulheres = confusion_matrix(
    df_test[df_test['genero'] == 'Feminino']['y_true'],
    df_test[df_test['genero'] == 'Feminino']['y_pred']
)

texto_cm = f"""MATRIZES DE CONFUSÃO POR GÊNERO

HOMENS:
            Predito Neg  Predito Pos
Real Neg    {cm_homens[0,0]:4d}         {cm_homens[0,1]:4d}
Real Pos    {cm_homens[1,0]:4d}         {cm_homens[1,1]:4d}

MULHERES:
            Predito Neg  Predito Pos
Real Neg    {cm_mulheres[0,0]:4d}         {cm_mulheres[0,1]:4d}
Real Pos    {cm_mulheres[1,0]:4d}         {cm_mulheres[1,1]:4d}

INTERPRETAÇÃO:
• FP (False Pos): Aprovado indevidamente
• FN (False Neg): Negado injustamente
"""

ax3.text(0.1, 0.5, texto_cm, fontsize=10, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax3.set_title("Matrizes de Confusão", fontsize=12, fontweight='bold')

# 4. Fairness Metrics
ax4 = axes[1, 1]
metricas_nomes = ['Disparate\nImpact', 'Ideal\n(1.0)', 'Threshold\n(0.8)']
metricas_valores = [disparate_impact, 1.0, 0.8]
cores_metricas = ['red' if disparate_impact < 0.8 else 'green', 'blue', 'orange']

ax4.bar(metricas_nomes, metricas_valores, color=cores_metricas, alpha=0.7, edgecolor='black')
ax4.set_ylabel("Ratio", fontsize=11)
ax4.set_title("Fairness: Disparate Impact Ratio", fontsize=12, fontweight='bold')
ax4.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5, label='Paridade Perfeita')
ax4.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Threshold Legal (80%)')
ax4.set_ylim(0, 1.2)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle("Análise de Viés em Sistema de Aprovação de Crédito", 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# ESTRATÉGIAS DE MITIGAÇÃO
# ═══════════════════════════════════════════════════════════════════

print("\n💡 ESTRATÉGIAS DE MITIGAÇÃO DE VIÉS:")
print("="*70)

print("\n1️⃣ PRÉ-PROCESSAMENTO (Antes do Treinamento):")
print("   ✓ Resampling: Balancear dados de grupos protegidos")
print("   ✓ Reweighting: Dar pesos diferentes aos exemplos")
print("   ✓ Remoção de features sensíveis (gênero, raça, etc.)")
print("   ⚠️ Atenção: Features proxy podem manter viés (CEP → raça)")

print("\n2️⃣ IN-PROCESSING (Durante o Treinamento):")
print("   ✓ Fairness Constraints: Adicionar restrições ao modelo")
print("   ✓ Adversarial Debiasing: Rede adversária remove viés")
print("   ✓ Regularização de fairness")

print("\n3️⃣ PÓS-PROCESSAMENTO (Após o Treinamento):")
print("   ✓ Ajustar thresholds por grupo")
print("   ✓ Calibração de probabilidades")
print("   ✓ Equalized Odds post-processing")

print("\n4️⃣ AUDITORIA CONTÍNUA:")
print("   ✓ Monitorar métricas de fairness em produção")
print("   ✓ Testes A/B com grupos protegidos")
print("   ✓ Relatórios de impacto algorítmico")

print("\n⚖️ LEGISLAÇÃO RELEVANTE:")
print("   • GDPR (Europa): Direito à explicação")
print("   • Fair Credit Reporting Act (EUA)")
print("   • LGPD (Brasil): Art. 20 - Decisões automatizadas")

print("\n🎯 CONCLUSÃO:")
print("Viés algorítmico não é problema técnico apenas, mas ÉTICO e SOCIAL.")
print("Desenvolvedores têm responsabilidade de auditar e mitigar vieses.")
print("Fairness é um espaço de tradeoffs: não há solução perfeita única.")

print("\n✅ ANÁLISE DE VIÉS COMPLETA!")
