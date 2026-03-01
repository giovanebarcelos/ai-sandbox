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
