# GO0621-ParaGerarLearningCurves
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, X, y, cv=5, scoring='r2'):
    """
    Plota learning curve para diagnosticar bias/variance
    """
    train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    # Calcular médias e desvios
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    # Plotar
    plt.figure(figsize=(10, 6))

    plt.fill_between(train_sizes_abs, 
                     train_mean - train_std,
                     train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.fill_between(train_sizes_abs, 
                     test_mean - test_std,
                     test_mean + test_std, 
                     alpha=0.1, color='orange')

    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue',
             label='Treino', linewidth=2, markersize=6)
    plt.plot(train_sizes_abs, test_mean, 'o-', color='orange',
             label='Validação', linewidth=2, markersize=6)

    plt.xlabel('Número de Amostras de Treino', fontsize=12)
    plt.ylabel(f'Score ({scoring})', fontsize=12)
    plt.title('Learning Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)

    # Diagnóstico automático
    final_gap = train_mean[-1] - test_mean[-1]
    final_test_score = test_mean[-1]

    diagnosis = ""
    if final_gap > 0.15:
        diagnosis = "⚠️ HIGH VARIANCE (Overfitting)\n→ Reduzir complexidade ou regularizar"
    elif final_test_score < 0.6:
        diagnosis = "⚠️ HIGH BIAS (Underfitting)\n→ Aumentar complexidade do modelo"
    elif test_mean[-1] - test_mean[-2] > 0.02:
        diagnosis = "📈 Mais dados podem ajudar\n→ Curva ainda não estabilizou"
    else:
        diagnosis = "✅ Modelo balanceado!\n→ Performance adequada"

    plt.text(0.02, 0.02, diagnosis, 
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

    print("="*60)
    print("DIAGNÓSTICO")
    print("="*60)
    print(f"Score final (treino):     {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
    print(f"Score final (validação):  {test_mean[-1]:.4f} ± {test_std[-1]:.4f}")
    print(f"Gap (treino - val):       {final_gap:.4f}")
    print(f"\n{diagnosis}")

# Uso:
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    plot_learning_curve(model, X_scaled, y, cv=5, scoring='r2')
