# GO1924-GeneticProgrammingSymbolicRegression
from gplearn.genetic import SymbolicRegressor
import numpy as np
import matplotlib.pyplot as plt

# Gerar dados de função desconhecida
np.random.seed(42)
X_train = np.random.uniform(-10, 10, 100).reshape(-1, 1)
y_train = 2 * X_train**2 + 3 * X_train + 5 + np.random.randn(100, 1) * 10

X_test = np.linspace(-10, 10, 200).reshape(-1, 1)
y_test_true = 2 * X_test**2 + 3 * X_test + 5

# Treinar GP para descobrir equação
gp = SymbolicRegressor(
    population_size=2000,
    generations=20,
    tournament_size=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    random_state=42
)

gp.fit(X_train, y_train.ravel())

# Melhor programa descoberto
print(f"📜 Equação Descoberta pelo GP:")
print(f"  {gp._program}")

# Prever
y_pred = gp.predict(X_test)

# Visualizar
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.6, label='Dados treino (com ruído)')
plt.plot(X_test, y_test_true, 'g-', linewidth=3, label='Função real: 2x²+3x+5')
plt.plot(X_test, y_pred, 'r--', linewidth=2, label=f'GP: {gp._program}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Genetic Programming - Symbolic Regression')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Comparar com polinômio sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

from sklearn.metrics import r2_score
print(f"📊 R² Score:")
print(f"  GP: {r2_score(y_test_true, y_pred):.4f}")
print(f"  Polynomial Regression: {r2_score(y_test_true, y_pred_poly.ravel()):.4f}")
