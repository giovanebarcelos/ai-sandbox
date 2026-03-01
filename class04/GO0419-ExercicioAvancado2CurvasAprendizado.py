# GO0419-ExercicioAvancado2CurvasAprendizado
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("ANÁLISE DE CURVAS DE APRENDIZADO")
print("=" * 70)

# 1. CARREGAR DATASET
data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 2. MODELOS
models = {
    'Ridge (alpha=1)': Ridge(alpha=1.0),
    'Ridge (alpha=100)': Ridge(alpha=100.0),
    'Tree (depth=3)': DecisionTreeRegressor(max_depth=3, random_state=42),
    'Tree (depth=10)': DecisionTreeRegressor(max_depth=10, random_state=42)
}

# 3. LEARNING CURVES
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, (name, model) in enumerate(models.items()):
    print(f"\n⚙️  {name}")

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_scaled, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )

    train_mean = np.sqrt(-train_scores.mean(axis=1))
    val_mean = np.sqrt(-val_scores.mean(axis=1))

    ax = axes[idx]
    ax.plot(train_sizes, train_mean, 'o-', label='Treino', linewidth=2)
    ax.plot(train_sizes, val_mean, 'o-', label='Validação', linewidth=2)
    ax.set_xlabel('Tamanho Treino')
    ax.set_ylabel('RMSE')
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)

    gap = val_mean[-1] - train_mean[-1]
    diagnosis = "OVERFITTING" if gap > 15 else "BOM FIT"
    print(f"   Gap: {gap:.1f} → {diagnosis}")

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=100)
print("\n✓ learning_curves.png")
print("\n✅ Análise de Curvas concluída!")
