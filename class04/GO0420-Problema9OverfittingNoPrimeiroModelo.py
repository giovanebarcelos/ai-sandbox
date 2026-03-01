# GO0420-Problema9OverfittingNoPrimeiroModelo
# 1. Verificar complexidade do modelo
from sklearn.tree import DecisionTreeClassifier

# ❌ Modelo muito complexo (sem restrições):


if __name__ == "__main__":
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # ✅ Modelo regularizado:
    model = DecisionTreeClassifier(
        max_depth=5,           # Limitar profundidade
        min_samples_split=20,  # Mínimo de amostras para dividir
        min_samples_leaf=10    # Mínimo de amostras por folha
    )
    model.fit(X_train, y_train)

    # 2. Usar validação cruzada
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # 3. Coletar mais dados ou usar data augmentation

    # 4. Simplificar features (remover irrelevantes)
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
