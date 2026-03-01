# GO2109-FairnessEMitigaçãoDeViés
# ═══════════════════════════════════════════════════════════════════
# FAIRNESS E MITIGAÇÃO DE VIÉS
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
import warnings
warnings.filterwarnings('ignore')

# ─── 1. CRIAR DATASET SINTÉTICO COM VIÉS ───
print("📦 Criando dataset sintético com viés racial...")

np.random.seed(42)
n_samples = 2000

# Gerar features
age = np.random.randint(18, 70, n_samples)
income = np.random.normal(50000, 20000, n_samples)
education = np.random.randint(1, 5, n_samples)  # 1=HS, 2=Associate, 3=Bachelor, 4=Master+

# Atributo protegido: Raça (0=Minoria, 1=Maioria)
# 30% minoria, 70% maioria
race = np.random.choice([0, 1], n_samples, p=[0.3, 0.4])

# Target: Aprovação de crédito (1=Aprovado, 0=Negado)
# Introduzir viés: minoritários têm menor probabilidade de aprovação mesmo com mesmas qualificações

# Score base (sem viés)
credit_score = (
    0.02 * age +
    0.00003 * income +
    10 * education +
    np.random.normal(0, 10, n_samples)
)

# Adicionar viés racial
racial_penalty = -15 * (race == 0)  # Penalizar minorias
credit_score_biased = credit_score + racial_penalty

# Decisão binária (threshold=50)
approved = (credit_score_biased > 50).astype(int)

# Criar DataFrame
df = pd.DataFrame({
    'age': age,
    'income': income,
    'education': education,
    'race': race,
    'approved': approved
})

print(f"  Total de amostras: {len(df)}")
print(f"  Distribuição racial: Minoria={sum(race==0)}, Maioria={sum(race==1)}")
print(f"  Taxa de aprovação geral: {approved.mean():.2%}")

# ─── 2. ANÁLISE DE VIÉS (PRÉ-MITIGAÇÃO) ───
print("\n📊 Analisando viés no dataset...")

# Taxa de aprovação por grupo
approval_by_race = df.groupby('race')['approved'].agg(['mean', 'count'])
approval_by_race.index = ['Minoria', 'Maioria']
print("\n  Taxa de aprovação por grupo:")
print(approval_by_race)

# Disparate Impact Ratio
minority_approval = df[df['race'] == 0]['approved'].mean()
majority_approval = df[df['race'] == 1]['approved'].mean()
disparate_impact = minority_approval / majority_approval

print(f"\n  ⚠️ Disparate Impact Ratio: {disparate_impact:.3f}")
print(f"     (< 0.8 indica discriminação - Regra 80%)")

# Visualizar
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Taxa de aprovação por raça
approval_by_race['mean'].plot(kind='bar', ax=axes[0], color=['coral', 'skyblue'])
axes[0].set_title('Taxa de Aprovação por Grupo', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Taxa de Aprovação')
axes[0].set_xlabel('Grupo')
axes[0].set_ylim([0, 1])
axes[0].axhline(y=0.8 * majority_approval, color='red', linestyle='--', label='80% Rule')
axes[0].legend()
axes[0].set_xticklabels(['Minoria', 'Maioria'], rotation=0)

# 2. Distribuição de income por raça
df.boxplot(column='income', by='race', ax=axes[1])
axes[1].set_title('Distribuição de Renda por Grupo', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Renda (USD)')
axes[1].set_xlabel('Grupo')
axes[1].set_xticklabels(['Minoria', 'Maioria'])
plt.sca(axes[1])
plt.xticks([1, 2], ['Minoria', 'Maioria'])

# 3. Correlação aprovação vs features
corr = df.corr()['approved'].drop('approved').sort_values()
corr.plot(kind='barh', ax=axes[2], color='teal')
axes[2].set_title('Correlação com Aprovação', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Correlação')
axes[2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('bias_analysis_pre.png', dpi=150)
print("  ✓ Gráfico salvo: bias_analysis_pre.png")

# ─── 3. TREINAR MODELO SEM MITIGAÇÃO ───
print("\n🔨 Treinando modelo SEM mitigação de viés...")

# Split
X = df[['age', 'income', 'education', 'race']]
y = df['approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Treinar
model_biased = RandomForestClassifier(n_estimators=100, random_state=42)
model_biased.fit(X_train, y_train)

# Predições
y_pred_biased = model_biased.predict(X_test)
acc_biased = accuracy_score(y_test, y_pred_biased)

print(f"  Accuracy: {acc_biased:.4f}")

# Métricas de fairness por grupo
test_df = X_test.copy()
test_df['approved'] = y_test
test_df['predicted'] = y_pred_biased

fairness_metrics = {}
for race_val, race_name in [(0, 'Minoria'), (1, 'Maioria')]:
    mask = test_df['race'] == race_val
    acc = accuracy_score(test_df[mask]['approved'], test_df[mask]['predicted'])
    approval_rate = test_df[mask]['predicted'].mean()

    fairness_metrics[race_name] = {
        'Accuracy': acc,
        'Approval Rate': approval_rate
    }

fairness_df = pd.DataFrame(fairness_metrics).T
print("\n  Métricas por grupo:")
print(fairness_df)

# Equal Opportunity Difference (TPR difference)
minority_tpr = ((test_df['race'] == 0) & (test_df['approved'] == 1) & (test_df['predicted'] == 1)).sum() / \
               ((test_df['race'] == 0) & (test_df['approved'] == 1)).sum()
majority_tpr = ((test_df['race'] == 1) & (test_df['approved'] == 1) & (test_df['predicted'] == 1)).sum() / \
               ((test_df['race'] == 1) & (test_df['approved'] == 1)).sum()
eod = abs(minority_tpr - majority_tpr)

print(f"\n  ⚠️ Equal Opportunity Difference: {eod:.3f}")
print(f"     (> 0.1 indica discriminação)")

# ─── 4. MITIGAÇÃO - PRÉ-PROCESSAMENTO (Reweighing) ───
print("\n🔧 Aplicando mitigação: Reweighing (Pré-processamento)...")

# Preparar para AIF360
df_train = X_train.copy()
df_train['approved'] = y_train.values

# Converter para BinaryLabelDataset (AIF360)
dataset_train = BinaryLabelDataset(
    df=df_train,
    label_names=['approved'],
    protected_attribute_names=['race'],
    favorable_label=1,
    unfavorable_label=0
)

# Aplicar Reweighing
RW = Reweighing(unprivileged_groups=[{'race': 0}], 
                privileged_groups=[{'race': 1}])
dataset_train_rw = RW.fit_transform(dataset_train)

# Treinar modelo com pesos
weights = dataset_train_rw.instance_weights
model_fair = RandomForestClassifier(n_estimators=100, random_state=42)
model_fair.fit(X_train, y_train, sample_weight=weights)

# Predições
y_pred_fair = model_fair.predict(X_test)
acc_fair = accuracy_score(y_test, y_pred_fair)

print(f"  Accuracy (com mitigação): {acc_fair:.4f}")

# Métricas de fairness após mitigação
test_df_fair = X_test.copy()
test_df_fair['approved'] = y_test
test_df_fair['predicted'] = y_pred_fair

fairness_metrics_fair = {}
for race_val, race_name in [(0, 'Minoria'), (1, 'Maioria')]:
    mask = test_df_fair['race'] == race_val
    acc = accuracy_score(test_df_fair[mask]['approved'], test_df_fair[mask]['predicted'])
    approval_rate = test_df_fair[mask]['predicted'].mean()

    fairness_metrics_fair[race_name] = {
        'Accuracy': acc,
        'Approval Rate': approval_rate
    }

fairness_df_fair = pd.DataFrame(fairness_metrics_fair).T
print("\n  Métricas por grupo (após mitigação):")
print(fairness_df_fair)

# EOD após mitigação
minority_tpr_fair = ((test_df_fair['race'] == 0) & (test_df_fair['approved'] == 1) & (test_df_fair['predicted'] == 1)).sum() / \
                    ((test_df_fair['race'] == 0) & (test_df_fair['approved'] == 1)).sum()
majority_tpr_fair = ((test_df_fair['race'] == 1) & (test_df_fair['approved'] == 1) & (test_df_fair['predicted'] == 1)).sum() / \
                    ((test_df_fair['race'] == 1) & (test_df_fair['approved'] == 1)).sum()
eod_fair = abs(minority_tpr_fair - majority_tpr_fair)

print(f"\n  ✅ Equal Opportunity Difference (mitigado): {eod_fair:.3f}")

# ─── 5. COMPARAÇÃO ANTES vs DEPOIS ───
print("\n📊 Comparando modelos...")

comparison = pd.DataFrame({
    'SEM Mitigação': [acc_biased, eod, fairness_df.loc['Minoria', 'Approval Rate'], 
                       fairness_df.loc['Maioria', 'Approval Rate']],
    'COM Mitigação': [acc_fair, eod_fair, fairness_df_fair.loc['Minoria', 'Approval Rate'], 
                       fairness_df_fair.loc['Maioria', 'Approval Rate']]
}, index=['Accuracy', 'EOD', 'Approval Rate (Minoria)', 'Approval Rate (Maioria)'])

print("\n  Comparação:")
print(comparison)

# Visualizar comparação
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Approval rates
approval_comparison = pd.DataFrame({
    'SEM Mitigação': [fairness_df.loc['Minoria', 'Approval Rate'], 
                       fairness_df.loc['Maioria', 'Approval Rate']],
    'COM Mitigação': [fairness_df_fair.loc['Minoria', 'Approval Rate'], 
                       fairness_df_fair.loc['Maioria', 'Approval Rate']]
}, index=['Minoria', 'Maioria'])

approval_comparison.plot(kind='bar', ax=axes[0], color=['coral', 'skyblue'])
axes[0].set_title('Taxa de Aprovação: Antes vs Depois', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Taxa de Aprovação')
axes[0].set_xlabel('Grupo')
axes[0].legend(title='Modelo')
axes[0].set_xticklabels(['Minoria', 'Maioria'], rotation=0)
axes[0].set_ylim([0, 1])

# EOD comparison
eod_comparison = pd.Series([eod, eod_fair], index=['SEM Mitigação', 'COM Mitigação'])
eod_comparison.plot(kind='bar', ax=axes[1], color=['red', 'green'])
axes[1].set_title('Equal Opportunity Difference', fontsize=12, fontweight='bold')
axes[1].set_ylabel('EOD')
axes[1].set_xlabel('Modelo')
axes[1].axhline(y=0.1, color='orange', linestyle='--', label='Threshold (0.1)')
axes[1].legend()
axes[1].set_xticklabels(['SEM\nMitigação', 'COM\nMitigação'], rotation=0)

plt.tight_layout()
plt.savefig('fairness_comparison.png', dpi=150)
print("\n  ✓ Comparação salva: fairness_comparison.png")

# ─── 6. RELATÓRIO FINAL ───
print("\n" + "="*70)
print("✅ ANÁLISE DE FAIRNESS CONCLUÍDA!")
print("="*70)

print("\n📊 Resumo:")
print(f"  - Disparate Impact (dataset): {disparate_impact:.3f}")
print(f"  - EOD sem mitigação: {eod:.3f}")
print(f"  - EOD com mitigação: {eod_fair:.3f}")
print(f"  - Melhoria: {((eod - eod_fair) / eod * 100):.1f}%")
print(f"  - Trade-off accuracy: {acc_biased - acc_fair:.4f}")

print("\n📁 Arquivos gerados:")
print("  - bias_analysis_pre.png")
print("  - fairness_comparison.png")

print("\n💡 Conclusões:")
if eod_fair < 0.1:
    print("  ✅ Modelo mitigado atende critérios de fairness (EOD < 0.1)")
else:
    print("  ⚠️ Modelo ainda apresenta disparidade (EOD > 0.1)")
    print("  → Considere técnicas adicionais: threshold optimization, adversarial debiasing")
