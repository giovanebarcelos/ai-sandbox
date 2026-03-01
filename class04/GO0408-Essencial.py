# GO0408-Essencial
# Pipeline básico
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Dividir dados


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinar
    model = DecisionTreeClassifier()
    model.fit(X_train_scaled, y_train)

    # Avaliar
    score = model.score(X_test_scaled, y_test)
