# GO0706-SegmentaçãoDeClientesParaMarketing
# ETAPA 1: CARREGAR E EXPLORAR DADOS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# CRIAR DATASET SINTÉTICO


if __name__ == "__main__":
    np.random.seed(42)

    # Segmento 1: Jovens com baixa renda e baixo gasto
    seg1_age = np.random.normal(25, 5, 50)
    seg1_income = np.random.normal(30, 10, 50)
    seg1_spending = np.random.normal(30, 10, 50)

    # Segmento 2: Meia-idade com alta renda e alto gasto
    seg2_age = np.random.normal(45, 8, 50)
    seg2_income = np.random.normal(80, 15, 50)
    seg2_spending = np.random.normal(75, 10, 50)

    # Segmento 3: Idosos com alta renda mas baixo gasto
    seg3_age = np.random.normal(60, 7, 50)
    seg3_income = np.random.normal(75, 12, 50)
    seg3_spending = np.random.normal(25, 8, 50)

    # Segmento 4: Jovens com renda média e alto gasto
    seg4_age = np.random.normal(28, 6, 50)
    seg4_income = np.random.normal(50, 12, 50)
    seg4_spending = np.random.normal(80, 10, 50)

    # Combinar todos
    df = pd.DataFrame({
        'CustomerID': range(1, 201),
        'Age': np.concatenate([seg1_age, seg2_age, seg3_age, seg4_age]),
        'Annual_Income': np.concatenate([seg1_income, seg2_income, 
                                         seg3_income, seg4_income]),
        'Spending_Score': np.concatenate([seg1_spending, seg2_spending, 
                                          seg3_spending, seg4_spending])
    })

    # Garantir valores dentro dos limites
    df['Age'] = df['Age'].clip(18, 70).astype(int)
    df['Annual_Income'] = df['Annual_Income'].clip(15, 150).round(1)
    df['Spending_Score'] = df['Spending_Score'].clip(1, 100).round(0).astype(int)

    # EXPLORAÇÃO INICIAL

    print("=" * 60)
    print("ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"\nPrimeiras linhas:")
    print(df.head(10))

    print(f"\nEstatísticas descritivas:")
    print(df.describe())

    print(f"\nValores faltantes:")
    print(df.isnull().sum())

    # ───────────────────────────────────────────────────────────────────
    # VISUALIZAÇÕES
    # ───────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Histogramas
    axes[0, 0].hist(df['Age'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribuição de Idade')
    axes[0, 0].set_xlabel('Idade')
    axes[0, 0].set_ylabel('Frequência')

    axes[0, 1].hist(df['Annual_Income'], bins=20, edgecolor='black', 
                   alpha=0.7, color='green')
    axes[0, 1].set_title('Distribuição de Renda Anual')
    axes[0, 1].set_xlabel('Renda (mil)')

    axes[0, 2].hist(df['Spending_Score'], bins=20, edgecolor='black', 
                   alpha=0.7, color='orange')
    axes[0, 2].set_title('Distribuição de Score de Gastos')
    axes[0, 2].set_xlabel('Score')

    # Scatter plots
    axes[1, 0].scatter(df['Annual_Income'], df['Spending_Score'], alpha=0.5)
    axes[1, 0].set_xlabel('Renda Anual (mil)')
    axes[1, 0].set_ylabel('Spending Score')
    axes[1, 0].set_title('Renda vs Spending Score')

    axes[1, 1].scatter(df['Age'], df['Spending_Score'], alpha=0.5, color='red')
    axes[1, 1].set_xlabel('Idade')
    axes[1, 1].set_ylabel('Spending Score')
    axes[1, 1].set_title('Idade vs Spending Score')

    axes[1, 2].scatter(df['Age'], df['Annual_Income'], alpha=0.5, color='purple')
    axes[1, 2].set_xlabel('Idade')
    axes[1, 2].set_ylabel('Renda Anual (mil)')
    axes[1, 2].set_title('Idade vs Renda')

    plt.tight_layout()
    plt.show()

    # Matriz de correlação
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[['Age', 'Annual_Income', 'Spending_Score']].corr(), 
               annot=True, cmap='coolwarm', center=0, 
               square=True, linewidths=1)
    plt.title('Matriz de Correlação')
    plt.show()

    print("\n✅ ETAPA 1 COMPLETA!")
    print("💡 Observações: Parece haver grupos distintos nos scatter plots!")
