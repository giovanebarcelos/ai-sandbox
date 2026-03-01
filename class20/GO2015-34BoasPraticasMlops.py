# GO2015-34BoasPráticasMlops
# Feast (open-source feature store)
from feast import FeatureStore


if __name__ == "__main__":
    store = FeatureStore(repo_path=".")
    features = store.get_online_features(
        entity_rows=[{"iris_id": 123}],
        features=["iris:sepal_length", "iris:petal_length"]
    ).to_dict()
