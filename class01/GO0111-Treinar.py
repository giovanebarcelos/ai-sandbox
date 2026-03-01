# GO0111-Treinar
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Preparar dados


if __name__ == "__main__":
    X = iris.data
    y = iris.target

    # Dividir
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

    # Treinar
    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo.fit(X_train, y_train)

    # Avaliar
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    print(f"\n📊 Acurácia: {acuracia*100:.2f}%")

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.ylabel('Real')
    plt.xlabel('Predição')
    plt.title('Matriz de Confusão')
    plt.show()

    # Relatório
    print("\nRelatório:\n")
    print(classification_report(y_test, y_pred, 
                              target_names=iris.target_names))
