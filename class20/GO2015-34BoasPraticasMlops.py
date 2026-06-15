# GO2015-34BoasPraticasMlops
# Código de referência (Feast - requer repositório Feast configurado, não roda no Colab):
#
# from feast import FeatureStore
#
# store = FeatureStore(repo_path=".")
# features = store.get_online_features(
#     entity_rows=[{"iris_id": 123}],
#     features=["iris:sepal_length", "iris:petal_length"]
# ).to_dict()

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


class FeatureStoreSimulado:
    """Simula um Feature Store online (ex: Feast) usando um dicionário em memória."""

    def __init__(self, dados, nomes_features):
        self.dados = dados
        self.nomes_features = nomes_features

    def get_online_features(self, entity_rows, features):
        resultado = {"iris_id": []}
        for f in features:
            resultado[f] = []

        for row in entity_rows:
            entity_id = row["iris_id"]
            resultado["iris_id"].append(entity_id)
            for f in features:
                nome_feature = f.split(":")[1]
                idx = self.nomes_features.index(nome_feature)
                resultado[f].append(float(self.dados[entity_id][idx]))

        return resultado


if __name__ == "__main__":
    print("=== Feature Store (demonstração local com Feast simulado) ===")
    print()

    iris = load_iris()
    nomes_features = [n.replace(" (cm)", "").replace(" ", "_") for n in iris.feature_names]

    store = FeatureStoreSimulado(dados=iris.data, nomes_features=nomes_features)

    features_solicitadas = [f"iris:{nomes_features[0]}", f"iris:{nomes_features[2]}"]
    resultado = store.get_online_features(
        entity_rows=[{"iris_id": 123}],
        features=features_solicitadas
    )

    print(f"  Features solicitadas: {features_solicitadas}")
    print(f"  Resultado: {resultado}")

    # Gráfico: valores das features retornadas pelo feature store
    nomes = list(resultado.keys())[1:]
    valores = [resultado[n][0] for n in nomes]

    plt.figure(figsize=(6, 4))
    plt.bar(nomes, valores, color="mediumpurple")
    plt.ylabel("Valor (cm)")
    plt.title(f"Features online para iris_id=123")
    plt.tight_layout()
    plt.show()
