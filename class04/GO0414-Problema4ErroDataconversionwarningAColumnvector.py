# GO0414-Problema4ErroDataconversionwarningAColumnvector
# Problema:


if __name__ == "__main__":
    y = df[['target']]  # 2D

    # Solução 1: Usar .values.ravel()
    y = df['target'].values  # 1D

    # Solução 2: Usar squeeze
    y = df[['target']].values.ravel()

    # Solução 3: Indexação simples
    y = df['target']  # Pandas Series (aceito)
