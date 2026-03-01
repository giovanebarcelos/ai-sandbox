# GO0611-FeatureEngineering
# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

# Criar cópia para não modificar original
df_eng = df.copy()

print("="*60)
print("FEATURE ENGINEERING")
print("="*60)

# ───────────────────────────────────────────────────────────────────
# 1. FEATURES DE INTERAÇÃO
# ───────────────────────────────────────────────────────────────────

df_eng['RoomsPerHousehold'] = df_eng['AveRooms'] / df_eng['AveOccup']
df_eng['BedroomsPerRoom'] = df_eng['AveBedrms'] / df_eng['AveRooms']
df_eng['PopulationPerHousehold'] = df_eng['Population'] / df_eng['HouseAge']

print("\n1. Features de Interação criadas:")
print("  - RoomsPerHousehold")
print("  - BedroomsPerRoom")
print("  - PopulationPerHousehold")

# ───────────────────────────────────────────────────────────────────
# 2. TRANSFORMAÇÕES LOGARÍTMICAS
# ───────────────────────────────────────────────────────────────────

# Para features com distribuição assimétrica
skewed_features = ['Population', 'AveOccup']
for feature in skewed_features:
    df_eng[f'log_{feature}'] = np.log1p(df_eng[feature])

print("\n2. Transformações logarítmicas:")
for f in skewed_features:
    print(f"  - log_{f}")

# ───────────────────────────────────────────────────────────────────
# 3. BINNING
# ───────────────────────────────────────────────────────────────────

df_eng['HouseAgeCategory'] = pd.cut(df_eng['HouseAge'], 
                                     bins=[0, 10, 25, 50, 100],
                                     labels=['Novo', 'Recente', 'Médio', 'Antigo'])

# One-hot encode
age_dummies = pd.get_dummies(df_eng['HouseAgeCategory'], prefix='Age')
df_eng = pd.concat([df_eng, age_dummies], axis=1)

print("\n3. Binning (HouseAge → Categorias):")
print(df_eng['HouseAgeCategory'].value_counts())

# ───────────────────────────────────────────────────────────────────
# 4. FEATURES POLINOMIAIS (selecionadas)
# ───────────────────────────────────────────────────────────────────

from sklearn.preprocessing import PolynomialFeatures

# Apenas para features principais (para não explodir dimensionalidade)
selected_features = ['MedInc', 'AveRooms']
X_selected = df_eng[selected_features].values

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_selected)

poly_feature_names = poly.get_feature_names_out(selected_features)
for i, name in enumerate(poly_feature_names):
    if name not in selected_features:  # Apenas novas features
        df_eng[name] = X_poly[:, i]

print(f"\n4. Features Polinomiais (grau 2):")
print(f"  Features originais: {len(selected_features)}")
print(f"  Features após poly: {X_poly.shape[1]}")

# ───────────────────────────────────────────────────────────────────
# RESUMO
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RESUMO")
print("="*60)
print(f"Features originais: {df.shape[1]}")
print(f"Features após engineering: {df_eng.shape[1]}")
print(f"Novas features criadas: {df_eng.shape[1] - df.shape[1]}")

print("\n✅ Feature Engineering completa!")
