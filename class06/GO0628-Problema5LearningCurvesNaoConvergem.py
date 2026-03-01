# GO0628-Problema5LearningCurvesNãoConvergem
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


if __name__ == "__main__":
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )

    train_mean = -train_scores.mean(axis=1)
    val_mean = -val_scores.mean(axis=1)

    plt.plot(train_sizes, train_mean, label='Treino')
    plt.plot(train_sizes, val_mean, label='Validação')
    plt.xlabel('Tamanho do Treino')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

    # Interpretação:
    # - Curvas paralelas distantes: UNDERFITTING → modelo mais complexo
    # - Treino baixo, val alto: OVERFITTING → regularização ou mais dados
    # - Convergindo: BOM! → coletar mais dados pode ajudar um pouco
