# GO2020-29ModelMonitoringDriftDetection
# Código de referência (Evidently - requer pip install evidently e arquivos CSV próprios):
#
# from evidently import ColumnMapping
# from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
# import pandas as pd
#
# reference_data = pd.read_csv('train_data.csv')
# current_data = pd.read_csv('production_data_week1.csv')
#
# column_mapping = ColumnMapping(
#     target='target',
#     numerical_features=['feature1', 'feature2', 'feature3'],
#     categorical_features=['category1', 'category2']
# )
#
# report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
# report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
# report.save_html('drift_report.html')
#
# drift_results = report.as_dict()
# if drift_results['metrics'][0]['result']['dataset_drift']:
#     print("⚠️ DATA DRIFT DETECTADO!")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris


if __name__ == "__main__":
    iris = load_iris(as_frame=True)
    reference_data = iris.frame.drop(columns=['target'])

    # Simular "dados de produção" com drift em duas features
    rng = np.random.default_rng(42)
    current_data = reference_data.copy()
    current_data['sepal length (cm)'] += rng.normal(1.0, 0.3, size=len(current_data))  # drift forte
    current_data['petal width (cm)'] += rng.normal(0.0, 0.05, size=len(current_data))  # drift fraco/quase nulo

    print("=== Model Monitoring - Data Drift Detection (KS-test) ===\n")

    drift_detectado_geral = False
    resultados = []

    for col in reference_data.columns:
        stat, p_value = ks_2samp(reference_data[col], current_data[col])
        drift = p_value < 0.05
        drift_detectado_geral = drift_detectado_geral or drift
        resultados.append({"feature": col, "ks_stat": stat, "p_value": p_value, "drift": drift})
        status = "⚠️ DRIFT" if drift else "OK"
        print(f"  {col:25s} KS={stat:.4f}  p-value={p_value:.4f}  -> {status}")

    if drift_detectado_geral:
        print("\n⚠️ DATA DRIFT DETECTADO!")
        print("  - Retreinar modelo urgente")
        print("  - Investigar features com maior drift")
    else:
        print("\n✅ Nenhum drift significativo detectado.")

    # Gráfico: distribuição de referência vs atual para cada feature
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flat, reference_data.columns):
        ax.hist(reference_data[col], bins=20, alpha=0.5, label="Referência (treino)", color="steelblue")
        ax.hist(current_data[col], bins=20, alpha=0.5, label="Atual (produção)", color="orange")
        r = next(r for r in resultados if r["feature"] == col)
        titulo = f"{col}\n{'⚠️ DRIFT' if r['drift'] else 'OK'} (p={r['p_value']:.4f})"
        ax.set_title(titulo)
        ax.legend(fontsize=8)

    fig.suptitle("Distribuição de features: referência vs. produção")
    plt.tight_layout()
    plt.show()
