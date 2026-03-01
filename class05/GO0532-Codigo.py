# GO0532-Codigo


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3,
        stratify=y,  # ← ESSENCIAL!
        random_state=42
    )
