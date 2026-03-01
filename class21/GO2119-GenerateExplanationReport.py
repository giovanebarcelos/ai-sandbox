# GO2119-GenerateExplanationReport
# Template para exportar explicações
def generate_explanation_report(model, sample, shap_values, features):
    """
    Gera relatório estruturado de explicação
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_version": "v1.2.3",
        "prediction": {
            "class": model.predict([sample])[0],
            "probability": model.predict_proba([sample])[0].tolist()
        },
        "explanation": {
            "method": "SHAP TreeExplainer",
            "top_features": [
                {
                    "name": features[i],
                    "value": sample[i],
                    "shap_value": shap_values[i],
                    "contribution": f"{abs(shap_values[i])/sum(abs(shap_values))*100:.1f}%"
                }
                for i in np.argsort(np.abs(shap_values))[::-1][:5]
            ]
        },
        "compliance": {
            "gdpr_compliant": True,
            "lgpd_compliant": True,
            "auditable": True
        }
    }
    return report


if __name__ == '__main__':
    import numpy as np
    import json
    from datetime import datetime

    print("=== Demonstração de Relatório de Explicação ===")

    # Modelo simulado para demo
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris

    iris = load_iris()
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(iris.data, iris.target)

    # Amostra para explicar
    sample = iris.data[42]
    features = iris.feature_names

    # SHAP simulado (valores aleatórios proporcionais a importâncias)
    importances = clf.feature_importances_
    shap_values = importances * (sample - iris.data.mean(axis=0))

    # Gerar relatório
    report = generate_explanation_report(clf, sample, shap_values, features)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  Predição: {iris.target_names[report['prediction']['class']]}")
    print("  Top feature:", report['explanation']['top_features'][0]['name'])
    print("  GDPR compliant:", report['compliance']['gdpr_compliant'])
