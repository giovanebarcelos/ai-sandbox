# GO0413-Exercicio2AnaliseExploratoriaAvancada
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Carregar o dataset Titanic
print("Carregando dataset Titanic...")
df = sns.load_dataset('titanic')

print("\n" + "=" * 60)
print("INFORMAÇÕES GERAIS DO DATASET")
print("=" * 60)
print(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
print(f"\nPrimeiras linhas:")
print(df.head())

# 2. Verificar valores faltantes e tipos de dados
print("\n" + "=" * 60)
print("TIPOS DE DADOS")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("VALORES FALTANTES")
print("=" * 60)
missing_data = pd.DataFrame({
    'Coluna': df.columns,
    'Total Faltantes': df.isnull().sum(),
    'Percentual (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Total Faltantes'] > 0].sort_values('Total Faltantes', ascending=False)
print(missing_data.to_string(index=False))

# Estatísticas descritivas
print("\n" + "=" * 60)
print("ESTATÍSTICAS DESCRITIVAS")
print("=" * 60)
print(df.describe())

# 3. Criar visualizações
fig = plt.figure(figsize=(16, 12))

# 3.1 Distribuição de sobreviventes por classe
plt.subplot(2, 3, 1)
survival_by_class = df.groupby(['pclass', 'survived']).size().unstack()
survival_by_class.plot(kind='bar', stacked=False, ax=plt.gca(), color=['#E74C3C', '#2ECC71'])
plt.title('Sobreviventes por Classe', fontsize=14, fontweight='bold')
plt.xlabel('Classe')
plt.ylabel('Quantidade')
plt.legend(['Não Sobreviveu', 'Sobreviveu'], loc='upper right')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)

# Taxa de sobrevivência por classe
plt.subplot(2, 3, 2)
survival_rate = df.groupby('pclass')['survived'].mean()
bars = plt.bar(survival_rate.index, survival_rate.values, color=['#3498DB', '#9B59B6', '#E67E22'])
plt.title('Taxa de Sobrevivência por Classe', fontsize=14, fontweight='bold')
plt.xlabel('Classe')
plt.ylabel('Taxa de Sobrevivência')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis='y')
# Adicionar valores nas barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2%}', ha='center', va='bottom', fontweight='bold')

# 3.2 Histograma de idades
plt.subplot(2, 3, 3)
df['age'].hist(bins=30, edgecolor='black', alpha=0.7, color='#3498DB')
plt.axvline(df['age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {df["age"].mean():.1f}')
plt.axvline(df['age'].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {df["age"].median():.1f}')
plt.title('Distribuição de Idades', fontsize=14, fontweight='bold')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Sobrevivência por sexo
plt.subplot(2, 3, 4)
survival_by_sex = df.groupby(['sex', 'survived']).size().unstack()
survival_by_sex.plot(kind='bar', stacked=False, ax=plt.gca(), color=['#E74C3C', '#2ECC71'])
plt.title('Sobreviventes por Sexo', fontsize=14, fontweight='bold')
plt.xlabel('Sexo')
plt.ylabel('Quantidade')
plt.legend(['Não Sobreviveu', 'Sobreviveu'])
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)

# 3.3 Correlação entre variáveis numéricas
plt.subplot(2, 3, 5)
numeric_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação', fontsize=14, fontweight='bold')

# Sobrevivência por faixa etária
plt.subplot(2, 3, 6)
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100], 
                         labels=['Criança (0-12)', 'Adolescente (13-18)', 
                                'Adulto (19-35)', 'Meia-idade (36-60)', 'Idoso (60+)'])
survival_by_age = df.groupby('age_group')['survived'].mean()
bars = plt.bar(range(len(survival_by_age)), survival_by_age.values, 
              color='#1ABC9C', edgecolor='black')
plt.title('Taxa de Sobrevivência por Faixa Etária', fontsize=14, fontweight='bold')
plt.xlabel('Faixa Etária')
plt.ylabel('Taxa de Sobrevivência')
plt.xticks(range(len(survival_by_age)), survival_by_age.index, rotation=45, ha='right')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis='y')
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('titanic_eda.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualizações salvas: titanic_eda.png")

# 4. Identificar 3 insights importantes
print("\n" + "=" * 60)
print("📊 INSIGHTS IMPORTANTES")
print("=" * 60)

# Insight 1: Taxa de sobrevivência por classe
survival_by_class_pct = df.groupby('pclass')['survived'].mean()
print("\n1️⃣ SOBREVIVÊNCIA POR CLASSE:")
for pclass, rate in survival_by_class_pct.items():
    print(f"   Classe {pclass}: {rate:.2%}")
print(f"   → Passageiros da 1ª classe tiveram {survival_by_class_pct[1]/survival_by_class_pct[3]:.2f}x mais chance de sobreviver que a 3ª classe")

# Insight 2: Taxa de sobrevivência por sexo
survival_by_sex_pct = df.groupby('sex')['survived'].mean()
print("\n2️⃣ SOBREVIVÊNCIA POR SEXO:")
for sex, rate in survival_by_sex_pct.items():
    print(f"   {sex.capitalize()}: {rate:.2%}")
print(f"   → Mulheres tiveram {survival_by_sex_pct['female']/survival_by_sex_pct['male']:.2f}x mais chance de sobreviver que homens")

# Insight 3: Idade média dos sobreviventes
age_by_survival = df.groupby('survived')['age'].mean()
print("\n3️⃣ IDADE MÉDIA:")
print(f"   Não sobreviventes: {age_by_survival[0]:.1f} anos")
print(f"   Sobreviventes: {age_by_survival[1]:.1f} anos")
print(f"   → Sobreviventes eram em média {abs(age_by_survival[1] - age_by_survival[0]):.1f} anos mais jovens")

# 5. Features mais úteis para predição
print("\n" + "=" * 60)
print("🎯 FEATURES MAIS ÚTEIS PARA PREDIÇÃO")
print("=" * 60)

# Correlação com sobrevivência
correlation_with_survived = correlation['survived'].drop('survived').abs().sort_values(ascending=False)
print("\nCorrelação com 'survived' (valores absolutos):")
for feature, corr_value in correlation_with_survived.items():
    print(f"   {feature}: {corr_value:.3f}")

print("\n📌 RECOMENDAÇÃO DE FEATURES:")
print("   1. pclass (classe) - Forte correlação negativa (-0.34)")
print("   2. fare (tarifa) - Correlação positiva (0.26)")
print("   3. sex (sexo) - Dado categórico crucial (74% mulheres sobreviveram)")
print("   4. age (idade) - Importância moderada, crianças tinham prioridade")
print("   5. parch (pais/filhos) - Famílias pequenas tiveram melhor taxa")
print("\n   Features a considerar:")
print("   • Criar feature 'family_size' = sibsp + parch + 1")
print("   • Criar feature 'is_alone' (booleano)")
print("   • Criar feature 'age_group' (faixas etárias)")
print("   • Criar feature 'title' extraído de 'name' (Mr., Mrs., Miss., etc.)")
