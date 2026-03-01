# GO0614-ValidaçãoCruzadaComMúltiplasMétricas
# ═══════════════════════════════════════════════════════════════════
# VALIDAÇÃO CRUZADA COMPLETA COM MÚLTIPLAS MÉTRICAS
# ═══════════════════════════════════════════════════════════════════

from sklearn.model_selection import cross_validate, KFold

# Usar todos os dados (sem test set, CV faz isso)
X_all = df_eng.drop(['MedHouseVal', 'HouseAgeCategory'], axis=1, errors='ignore')
y_all = df_eng['MedHouseVal']

# Normalizar
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# ───────────────────────────────────────────────────────────────────
# CROSS-VALIDATION COM MÚLTIPLOS MODELOS
# ───────────────────────────────────────────────────────────────────

models_cv = {
    'Linear Regression': LinearRegression(),
    'Ridge (CV)': RidgeCV(alphas=np.logspace(-3, 3, 30)),
    'Lasso (CV)': LassoCV(alphas=np.logspace(-3, 1, 30), max_iter=10000),
    'Elastic Net (CV)': ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                                      alphas=np.logspace(-3, 1, 30), max_iter=10000)
}

scoring = {
    'r2': 'r2',
    'neg_rmse': 'neg_root_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error'
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

print("="*60)
print("10-FOLD CROSS-VALIDATION")
print("="*60)

cv_results_summary = []

for name, model in models_cv.items():
    print(f"\nAvaliando {name}...")

    cv_results = cross_validate(model, X_all_scaled, y_all, 
                                 cv=kf, scoring=scoring,
                                 return_train_score=True)

    cv_results_summary.append({
        'Model': name,
        'Train R²': cv_results['train_r2'].mean(),
        'Test R²': cv_results['test_r2'].mean(),
        'Test R² Std': cv_results['test_r2'].std(),
        'Test RMSE': -cv_results['test_neg_rmse'].mean(),
        'Test MAE': -cv_results['test_neg_mae'].mean()
    })

# ───────────────────────────────────────────────────────────────────
# RESULTADOS
# ───────────────────────────────────────────────────────────────────

cv_df = pd.DataFrame(cv_results_summary)

print("\n" + "="*60)
print("RESULTADOS CROSS-VALIDATION")
print("="*60)
print(cv_df.to_string(index=False))

# Melhor modelo
best_idx = cv_df['Test R²'].idxmax()
print(f"\n🏆 Melhor modelo (10-fold CV): {cv_df.loc[best_idx, 'Model']}")
print(f"   R² = {cv_df.loc[best_idx, 'Test R²']:.4f} ± {cv_df.loc[best_idx, 'Test R² Std']:.4f}")
