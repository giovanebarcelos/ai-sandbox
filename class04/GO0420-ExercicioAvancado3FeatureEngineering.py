# GO0420-ExercicioAvancado3FeatureEngineering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 70)
print("FEATURE ENGINEERING AVANÇADO")
print("=" * 70)

# 1. DATASET SINTÉTICO
np.random.seed(42)
n = 500

df = pd.DataFrame({
    'X1': np.random.randn(n),
    'X2': np.random.randn(n),
    'X3': np.random.randn(n),
    'X4': np.random.exponential(1, n),
    'X5': np.random.uniform(0, 10, n)
})

# Target com relação complexa
df['y'] = (
    2 * df['X1']**2 + 
    3 * df['X2'] * df['X3'] -
    np.log1p(df['X4']) +
    np.random.normal(0, 0.5, n)
)

X_orig = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X_orig, y, test_size=0.2, random_state=42
)

# 2. BASELINE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_base = Ridge(alpha=1.0)
model_base.fit(X_train_scaled, y_train)

y_pred_base = model_base.predict(X_test_scaled)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
r2_base = r2_score(y_test, y_pred_base)

print(f"\n📊 BASELINE:")
print(f"   RMSE: {rmse_base:.3f}")
print(f"   R²: {r2_base:.3f}")

# 3. FEATURE ENGINEERING
print("\n" + "=" * 70)
print("CRIANDO FEATURES")
print("=" * 70)

X_train_fe = X_train.copy()
X_test_fe = X_test.copy()

# Quadráticas
print("\n1️⃣  Features Quadráticas:")
for col in ['X1', 'X2']:
    X_train_fe[f'{col}_sq'] = X_train_fe[col] ** 2
    X_test_fe[f'{col}_sq'] = X_test_fe[col] ** 2
    print(f"   ✓ {col}_sq")

# Interações
print("\n2️⃣  Interações:")
X_train_fe['X2_x_X3'] = X_train_fe['X2'] * X_train_fe['X3']
X_test_fe['X2_x_X3'] = X_test_fe['X2'] * X_test_fe['X3']
print("   ✓ X2_x_X3")

# Log
print("\n3️⃣  Transformações:")
X_train_fe['X4_log'] = np.log1p(X_train_fe['X4'])
X_test_fe['X4_log'] = np.log1p(X_test_fe['X4'])
print("   ✓ X4_log")

print(f"\n📊 Features: {X_orig.shape[1]} → {X_train_fe.shape[1]}")

# 4. MODELO COM FE
scaler_fe = StandardScaler()
X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
X_test_fe_scaled = scaler_fe.transform(X_test_fe)

model_fe = Ridge(alpha=1.0)
model_fe.fit(X_train_fe_scaled, y_train)

y_pred_fe = model_fe.predict(X_test_fe_scaled)
rmse_fe = np.sqrt(mean_squared_error(y_test, y_pred_fe))
r2_fe = r2_score(y_test, y_pred_fe)

print(f"\n📊 COM FEATURE ENGINEERING:")
print(f"   RMSE: {rmse_fe:.3f}")
print(f"   R²: {r2_fe:.3f}")

# 5. FEATURE SELECTION
print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

selector = SelectKBest(f_regression, k=6)
X_train_sel = selector.fit_transform(X_train_fe_scaled, y_train)
X_test_sel = selector.transform(X_test_fe_scaled)

model_sel = Ridge(alpha=1.0)
model_sel.fit(X_train_sel, y_train)

y_pred_sel = model_sel.predict(X_test_sel)
rmse_sel = np.sqrt(mean_squared_error(y_test, y_pred_sel))
r2_sel = r2_score(y_test, y_pred_sel)

print(f"\n📊 COM FE + SELECTION:")
print(f"   RMSE: {rmse_sel:.3f}")
print(f"   R²: {r2_sel:.3f}")

# 6. COMPARAÇÃO
improvement = ((rmse_base - rmse_sel) / rmse_base) * 100

print("\n" + "=" * 70)
print("RESULTADOS")
print("=" * 70)
print(f"\nBaseline RMSE: {rmse_base:.3f}")
print(f"FE + Selection RMSE: {rmse_sel:.3f}")
print(f"Melhoria: {improvement:.1f}%")

print("\n✅ Feature Engineering Avançado concluído!")
