# GO0530-SklearnSklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ================================
# 1. RANDOM FOREST COM CLASS WEIGHTS
# ================================


if __name__ == "__main__":
    rf_balanced = RandomForestClassifier(
        class_weight='balanced',  # Calcula pesos automaticamente
        random_state=42
    )
    rf_balanced.fit(X_train, y_train)
    y_pred_rf = rf_balanced.predict(X_test)

    print("RANDOM FOREST COM CLASS WEIGHTS:")
    print(classification_report(y_test, y_pred_rf))

    # ================================
    # 2. LOGISTIC REGRESSION COM PESOS
    # ================================
    lr_balanced = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    lr_balanced.fit(X_train, y_train)
    y_pred_lr = lr_balanced.predict(X_test)

    print("\nLOGISTIC REGRESSION COM CLASS WEIGHTS:")
    print(classification_report(y_test, y_pred_lr))

    # ================================
    # 3. SVM COM PESOS
    # ================================
    svm_balanced = SVC(
        class_weight='balanced',
        random_state=42
    )
    svm_balanced.fit(X_train, y_train)
    y_pred_svm = svm_balanced.predict(X_test)

    print("\nSVM COM CLASS WEIGHTS:")
    print(classification_report(y_test, y_pred_svm))

    # ================================
    # 4. PESOS CUSTOMIZADOS
    # ================================
    # Você pode definir pesos manualmente:
    custom_weights = {
        0: 1,    # Classe majoritária: peso normal
        1: 20    # Classe minoritária: 20x mais importante
    }

    rf_custom = RandomForestClassifier(
        class_weight=custom_weights,
        random_state=42
    )
    rf_custom.fit(X_train, y_train)
