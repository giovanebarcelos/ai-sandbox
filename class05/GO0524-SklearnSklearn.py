# GO0524-SklearnSklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import time

# Carregar dados
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos
modelos = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False)
}

# Comparar
for nome, modelo in modelos.items():
    # Tempo de treino
    start = time.time()
    modelo.fit(X_train, y_train)
    tempo_treino = time.time() - start

    # Acurácia
    train_acc = modelo.score(X_train, y_train)
    test_acc = modelo.score(X_test, y_test)

    # Cross-validation
    cv_scores = cross_val_score(modelo, X, y, cv=5)

    print(f"\n{nome}:")
    print(f"  Tempo treino: {tempo_treino:.2f}s")
    print(f"  Treino:       {train_acc:.3f}")
    print(f"  Teste:        {test_acc:.3f}")
    print(f"  CV (média):   {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
