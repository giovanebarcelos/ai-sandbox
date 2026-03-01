# GO0913-Go0912templates


if __name__ == "__main__":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
