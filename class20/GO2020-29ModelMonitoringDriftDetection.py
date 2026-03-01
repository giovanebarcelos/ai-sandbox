# GO2020-29ModelMonitoringDriftDetection
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import pandas as pd

# Dados de referência (treino) vs dados atuais (produção)
reference_data = pd.read_csv('train_data.csv')
current_data = pd.read_csv('production_data_week1.csv')

# Configurar colunas
column_mapping = ColumnMapping(
    target='target',
    numerical_features=['feature1', 'feature2', 'feature3'],
    categorical_features=['category1', 'category2']
)

# Gerar relatório de drift
report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset()
])

report.run(reference_data=reference_data, 
           current_data=current_data,
           column_mapping=column_mapping)

# Salvar relatório HTML
report.save_html('drift_report.html')

# Alertas automáticos
drift_results = report.as_dict()
if drift_results['metrics'][0]['result']['dataset_drift']:
    print("⚠️ DATA DRIFT DETECTADO!")
    print("  - Retreinar modelo urgente")
    print("  - Investigar features com maior drift")
