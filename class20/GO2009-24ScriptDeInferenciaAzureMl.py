# GO2009-24ScriptDeInferênciaAzureMl
import json
import numpy as np
import joblib

def init():
    global model
    model_path = Model.get_model_path('iris-rf-model')
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)['data']
    predictions = model.predict(data)
    return predictions.tolist()


if __name__ == '__main__':
    import json
    import numpy as np

    print("=== Script de Inferência Azure ML (demo local) ===")
    print()
    print("  Em produção, este script é executado pelo Azure ML como endpoint.")
    print("  As funções init() e run() são chamadas automaticamente.")
    print()

    # Simular treinamento e serialização local para demo
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    import joblib, tempfile, os

    # Treinar modelo simples
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'model.pkl')
        joblib.dump(clf, model_path)

        # Simular init() e run() localmente
        model_local = joblib.load(model_path)

        amostra = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]
        raw_data = json.dumps({"data": amostra})

        data = json.loads(raw_data)['data']
        preds = model_local.predict(data).tolist()

        iris = load_iris()
        print(f"  Entrada: {amostra}")
        print(f"  Predições (índices): {preds}")
        print(f"  Classes: {[iris.target_names[p] for p in preds]}")
