# ml/train_model.py

import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, JobResourceConfiguration, UserIdentityConfiguration


credential = DefaultAzureCredential()
# Autenticación con Azure
ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_WORKSPACE_NAME"]
)

# Definir entorno de ejecución con tus requirements.txt
env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",  # imagen base
    conda_file="backend/ml/training/conda.yaml",
    name="cv-adaptation-env"
)

# Crear el job de entrenamiento serverless
job = command(
    code="backend/ml/training",      # ✅ sube la carpeta que contiene requirements.txt
    command="python train_finetune.py",
    environment=env,
    identity=UserIdentityConfiguration(),
    experiment_name="cv-adaptation",
    display_name="fine-tune-cv-model-cheap",
    resources=JobResourceConfiguration(
        instance_type="Standard_DS1_v2",  # 🔸 CPU barata
        instance_count=1
    ),
    queue_settings={"job_tier": "spot"}  # 🔸 Spot = más barato
)
job.compute = "serverless"
ml_client.jobs.create_or_update(job)
