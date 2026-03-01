# GO2008-23AzureMachineLearning
from azureml.core import Workspace, Model, Environment
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig

# Conectar ao workspace


if __name__ == "__main__":
    ws = Workspace.from_config()

    # Registrar modelo
    model = Model.register(workspace=ws,
                           model_path='model.pkl',
                           model_name='iris-rf-model',
                           description='Iris classifier')

    # Ambiente
    env = Environment.from_conda_specification(name='iris-env',
                                               file_path='conda.yml')

    # Configuração de inferência
    inference_config = InferenceConfig(entry_script='score.py',
                                       environment=env)

    # Configuração de deploy (ACI)
    aci_config = AciWebservice.deploy_configuration(cpu_cores=1,
                                                     memory_gb=1,
                                                     auth_enabled=True)

    # Deploy
    service = Model.deploy(workspace=ws,
                           name='iris-api-aml',
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=aci_config)

    service.wait_for_deployment(show_output=True)
    print(f"Scoring URI: {service.scoring_uri}")
