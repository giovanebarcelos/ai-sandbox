# GO2008-23AzureMachineLearning
# Código de referência (requer workspace Azure ML configurado - não roda no Colab):
#
# from azureml.core import Workspace, Model, Environment
# from azureml.core.webservice import AciWebservice, Webservice
# from azureml.core.model import InferenceConfig
#
# ws = Workspace.from_config()
#
# model = Model.register(workspace=ws,
#                        model_path='model.pkl',
#                        model_name='iris-rf-model',
#                        description='Iris classifier')
#
# env = Environment.from_conda_specification(name='iris-env',
#                                            file_path='conda.yml')
#
# inference_config = InferenceConfig(entry_script='score.py',
#                                    environment=env)
#
# aci_config = AciWebservice.deploy_configuration(cpu_cores=1,
#                                                  memory_gb=1,
#                                                  auth_enabled=True)
#
# service = Model.deploy(workspace=ws,
#                        name='iris-api-aml',
#                        models=[model],
#                        inference_config=inference_config,
#                        deployment_config=aci_config)
#
# service.wait_for_deployment(show_output=True)
# print(f"Scoring URI: {service.scoring_uri}")

import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("=== Deploy no Azure Machine Learning (demonstração conceitual) ===")
    print()
    print("  Este fluxo requer um workspace Azure ML configurado e o SDK azureml-core.")
    print("  Simulando localmente as etapas do pipeline de deploy...")
    print()

    etapas = [
        "Conectar ao Workspace",
        "Registrar modelo (model.pkl)",
        "Criar/validar ambiente Conda",
        "Configurar InferenceConfig (score.py)",
        "Configurar deploy ACI (1 CPU, 1GB RAM)",
        "Deploy do endpoint"
    ]

    duracoes = [0.5, 0.8, 0.6, 0.4, 0.5, 1.2]  # tempos simulados (s)

    for etapa, dur in zip(etapas, duracoes):
        print(f"  -> {etapa}...")
        time.sleep(0.05)  # apenas para fins didáticos

    print()
    print("  Scoring URI (simulado): https://iris-api-aml.azurewebsites.net/score")

    # Gráfico: tempo simulado de cada etapa do pipeline de deploy
    plt.figure(figsize=(9, 5))
    plt.barh(etapas, duracoes, color="cornflowerblue")
    plt.xlabel("Tempo simulado (s)")
    plt.title("Pipeline de deploy no Azure ML (simulação)")
    plt.tight_layout()
    plt.show()
