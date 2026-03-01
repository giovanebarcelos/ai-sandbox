# GO0411-Problema2ErroValueerrorCouldNot
# Opção 1: Label Encoding (variáveis ordinais)
from sklearn.preprocessing import LabelEncoder


if __name__ == "__main__":
    le = LabelEncoder()
    df['categoria_encoded'] = le.fit_transform(df['categoria'])

    # Opção 2: One-Hot Encoding (variáveis nominais)
    df_encoded = pd.get_dummies(df, columns=['categoria'])

    # Opção 3: Usando pandas
    df['categoria_encoded'] = df['categoria'].astype('category').cat.codes
