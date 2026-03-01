# GO0534-SmoteClass
# SMOTE + Class Weights


if __name__ == "__main__":
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        class_weight='balanced',  # Pesos adicionais
        random_state=42
    )
    model.fit(X_smote, y_smote)
