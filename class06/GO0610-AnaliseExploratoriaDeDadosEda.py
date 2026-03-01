# GO0610-AnáliseExploratóriaDeDadosEda
# ═══════════════════════════════════════════════════════════════════
# ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# ═══════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────
# MATRIZ DE CORRELAÇÃO
# ───────────────────────────────────────────────────────────────────

plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação', fontsize=16)
plt.tight_layout()
plt.show()

# Features mais correlacionadas com o target
print("="*60)
print("CORRELAÇÕES COM O PREÇO")
print("="*60)
correlations = corr_matrix['MedHouseVal'].sort_values(ascending=False)
print(correlations)

# ───────────────────────────────────────────────────────────────────
# SCATTER PLOTS DAS FEATURES PRINCIPAIS
# ───────────────────────────────────────────────────────────────────

# Top 4 features mais correlacionadas (excluindo o próprio target)
top_features = correlations.index[1:5]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for i, feature in enumerate(top_features):
    axes[i].scatter(df[feature], df['MedHouseVal'], alpha=0.3)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('MedHouseVal')
    axes[i].set_title(f'{feature} vs Preço (corr={correlations[feature]:.3f})')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────────
# DETECTAR OUTLIERS
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("DETECÇÃO DE OUTLIERS (IQR)")
print("="*60)

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

for col in df.columns:
    outliers, lower, upper = detect_outliers_iqr(df, col)
    pct = (len(outliers) / len(df)) * 100
    print(f"{col:15s}: {len(outliers):5d} outliers ({pct:.1f}%)")

# ───────────────────────────────────────────────────────────────────
# DISTRIBUIÇÕES
# ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=50, edgecolor='black', alpha=0.7)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequência')
    axes[i].set_title(f'Distribuição de {col}')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ EDA completa!")
