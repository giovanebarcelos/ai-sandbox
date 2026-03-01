# GO2116-35ePipelineMlopsCompleto
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import json
from datetime import datetime

print("="*70)
print("PIPELINE MLOps COMPLETO")
print("="*70)

# 1. CONFIGURAR MLFLOW
mlflow.set_tracking_uri("file:./mlruns")  # Local (pode ser servidor remoto)
mlflow.set_experiment("diabetes-prediction")

print("\n🔧 MLflow configurado")
print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
print(f"   Experiment: {mlflow.get_experiment_by_name('diabetes-prediction')}")

# 2. CARREGAR E PREPARAR DADOS
print("\n📊 Carregando dados...")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# 3. DEFINIR MODELOS E HIPERPARÂMETROS
models_config = [
    {
        'name': 'Random Forest',
        'model': RandomForestRegressor,
        'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
    },
    {
        'name': 'Gradient Boosting',
        'model': GradientBoostingRegressor,
        'params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42}
    },
    {
        'name': 'Ridge Regression',
        'model': Ridge,
        'params': {'alpha': 1.0, 'random_state': 42}
    }
]

# 4. TREINAR E LOGAR MODELOS
print("\n🏋️ Treinando modelos e logando no MLflow...\n")

results = []

for config in models_config:
    with mlflow.start_run(run_name=config['name']):
        print(f"📦 Treinando: {config['name']}")

        # Instanciar modelo
        model = config['model'](**config['params'])

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                    scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())

        # Treinar no conjunto completo
        model.fit(X_train, y_train)

        # Predições
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Métricas
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'cv_rmse': cv_rmse
        }

        # Logar parâmetros
        mlflow.log_params(config['params'])

        # Logar métricas
        mlflow.log_metrics(metrics)

        # Logar modelo
        mlflow.sklearn.log_model(model, "model")

        # Salvar gráfico
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Predição Perfeita')
        plt.xlabel('Real')
        plt.ylabel('Predito')
        plt.title(f'{config["name"]} - Predições vs Real')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = f"{config['name'].replace(' ', '_')}_predictions.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        # Salvar resultado
        results.append({
            'model': config['name'],
            'test_rmse': metrics['test_rmse'],
            'test_r2': metrics['test_r2'],
            'cv_rmse': cv_rmse
        })

        print(f"  ✅ Test RMSE: {metrics['test_rmse']:.2f}")
        print(f"  ✅ Test R²: {metrics['test_r2']:.3f}")
        print(f"  ✅ CV RMSE: {cv_rmse:.2f}\n")

# 5. COMPARAR MODELOS
print("="*70)
print("📊 COMPARAÇÃO DE MODELOS")
print("="*70)

df_results = pd.DataFrame(results)
print("\n", df_results.to_string(index=False))

# Melhor modelo
best_model_name = df_results.loc[df_results['test_rmse'].idxmin(), 'model']
print(f"\n🏆 MELHOR MODELO: {best_model_name}")

# Plotar comparação
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(df_results['model'], df_results['test_rmse'])
axes[0].set_title('RMSE (menor é melhor)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('RMSE')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(df_results['model'], df_results['test_r2'])
axes[1].set_title('R² Score (maior é melhor)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('R²')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('models_comparison.png', dpi=150)
mlflow.log_artifact('models_comparison.png')
print("✅ Comparação salva e logada no MLflow")

# 6. REGISTRAR MELHOR MODELO
print("\n📦 Registrando melhor modelo no Model Registry...")

# Buscar run do melhor modelo
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("diabetes-prediction")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{best_model_name}'",
    order_by=["metrics.test_rmse ASC"],
    max_results=1
)

if runs:
    best_run = runs[0]

    # Registrar modelo
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_version = mlflow.register_model(model_uri, "diabetes-model")

    print(f"  ✅ Modelo registrado: diabetes-model")
    print(f"  ✅ Versão: {model_version.version}")
    print(f"  ✅ Run ID: {best_run.info.run_id}")

    # Transicionar para Production
    client.transition_model_version_stage(
        name="diabetes-model",
        version=model_version.version,
        stage="Production"
    )
    print(f"  ✅ Modelo movido para PRODUCTION")

# 7. CARREGAR MODELO DE PRODUCTION E PREVER
print("\n🚀 Testando modelo em Production...")

model_production = mlflow.pyfunc.load_model("models:/diabetes-model/Production")

# Fazer predições
test_samples = X_test[:5]
predictions = model_production.predict(test_samples)

print("\n📊 Predições de teste:")
for i, (pred, real) in enumerate(zip(predictions, y_test[:5]), 1):
    print(f"  Amostra {i}: Predito={pred:.2f}, Real={real:.2f}")

# 8. SIMULAR MONITORAMENTO
print("\n📈 Simulando monitoramento em produção...")

# Criar dataframe de monitoramento
monitoring_data = []

for i in range(100):
    # Simular drift (dados ligeiramente diferentes)
    noise = np.random.randn(1, X.shape[1]) * 0.1
    sample = X_test[i % len(X_test)] + noise

    prediction = model_production.predict(sample.reshape(1, -1))[0]

    monitoring_data.append({
        'timestamp': datetime.now().isoformat(),
        'prediction': prediction,
        'sample_id': i
    })

df_monitoring = pd.DataFrame(monitoring_data)

# Análise de drift
plt.figure(figsize=(12, 6))
plt.plot(df_monitoring['sample_id'], df_monitoring['prediction'], 'o-', alpha=0.6)
plt.axhline(y=y.mean(), color='r', linestyle='--', label='Média Original')
plt.xlabel('Sample ID')
plt.ylabel('Predição')
plt.title('Monitoramento de Predições em Produção', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('monitoring_predictions.png', dpi=150)
print("✅ Gráfico de monitoramento salvo")

print("\n" + "="*70)
print("✅ PIPELINE MLOPS COMPLETO EXECUTADO!")
print("="*70)

print("\n📋 Componentes do Pipeline:")
print("  1. ✅ Treinamento de múltiplos modelos")
print("  2. ✅ Logging de parâmetros e métricas (MLflow)")
print("  3. ✅ Comparação de modelos")
print("  4. ✅ Seleção do melhor modelo")
print("  5. ✅ Registro no Model Registry")
print("  6. ✅ Deploy para Production")
print("  7. ✅ Inferência com modelo em produção")
print("  8. ✅ Monitoramento de predições")

print("\n🚀 Próximos Passos:")
print("  - Implementar API REST (FastAPI/Flask)")
print("  - Containerizar com Docker")
print("  - Deploy em Kubernetes")
print("  - Monitoramento de drift (Evidently AI)")
print("  - CI/CD com GitHub Actions")
print("  - A/B Testing")

print("\n💻 Acessar MLflow UI:")
print("  $ mlflow ui")
print("  Acesse: http://localhost:5000")
