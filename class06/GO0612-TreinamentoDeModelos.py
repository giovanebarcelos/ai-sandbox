# GO0612-TreinamentoDeModelos
# ═══════════════════════════════════════════════════════════════════
# TREINAMENTO DE MODELOS
# ═══════════════════════════════════════════════════════════════════

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ───────────────────────────────────────────────────────────────────
# PREPARAR DADOS
# ───────────────────────────────────────────────────────────────────

# Remover target e features categóricas não encodadas
X = df_eng.drop(['MedHouseVal', 'HouseAgeCategory'], axis=1, errors='ignore')
y = df_eng['MedHouseVal']

# Dividir
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("="*60)
print("TREINAMENTO DE MODELOS")
print("="*60)
print(f"Treino: {X_train.shape}")
print(f"Teste: {X_test.shape}")

# ───────────────────────────────────────────────────────────────────
# TREINAR MÚLTIPLOS MODELOS
# ───────────────────────────────────────────────────────────────────

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Ridge (α=10.0)': Ridge(alpha=10.0),
    'Lasso (α=0.1)': Lasso(alpha=0.1, max_iter=10000),
    'Lasso (α=1.0)': Lasso(alpha=1.0, max_iter=10000),
    'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
}

results = []

for name, model in models.items():
    # Treinar
    model.fit(X_train_scaled, y_train)

    # Predições
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Métricas
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Contar features não-zero
    if hasattr(model, 'coef_'):
        n_features = np.sum(np.abs(model.coef_) > 1e-5)
    else:
        n_features = X_train.shape[1]

    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Test MAE': test_mae,
        '# Features': n_features
    })

# ───────────────────────────────────────────────────────────────────
# COMPARAÇÃO
# ───────────────────────────────────────────────────────────────────

results_df = pd.DataFrame(results)

print("\n" + "="*60)
print("RESULTADOS")
print("="*60)
print(results_df.to_string(index=False))

# Melhor modelo
best_idx = results_df['Test R²'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_r2 = results_df.loc[best_idx, 'Test R²']

print(f"\n🏆 Melhor modelo: {best_model_name}")
print(f"   R² (teste): {best_r2:.4f}")

# ───────────────────────────────────────────────────────────────────
# VISUALIZAR COMPARAÇÃO
# ───────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# R² scores
x_pos = np.arange(len(results_df))
ax1.barh(x_pos, results_df['Train R²'], alpha=0.6, label='Treino')
ax1.barh(x_pos, results_df['Test R²'], alpha=0.6, label='Teste')
ax1.set_yticks(x_pos)
ax1.set_yticklabels(results_df['Model'])
ax1.set_xlabel('R²')
ax1.set_title('R² Score por Modelo')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='x')

# RMSE
ax2.barh(x_pos, results_df['Train RMSE'], alpha=0.6, label='Treino')
ax2.barh(x_pos, results_df['Test RMSE'], alpha=0.6, label='Teste')
ax2.set_yticks(x_pos)
ax2.set_yticklabels(results_df['Model'])
ax2.set_xlabel('RMSE')
ax2.set_title('RMSE por Modelo')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
