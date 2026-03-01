# GO0412-Problema3AcuráciaMuitoAltaSuspeita
# Verificar correlação das features com o target


if __name__ == "__main__":
    correlation = df.corr()['target'].sort_values(ascending=False)
    print(correlation)

    # Correlação > 0.95 é suspeita!
    suspicious_features = correlation[correlation > 0.95].index.tolist()
    print(f"Features suspeitas: {suspicious_features}")
