# ml/deployment/register_model.py

import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

# Conectar a Azure ML
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group=os.environ["AZURE_RESOURCE_GROUP"],
    workspace=os.environ["AZURE_WORKSPACE_NAME"]
)

# Buscar el último job del experimento
experiment_name = "cv-adaptation"
jobs = ml_client.jobs.list(experiment_name=experiment_name, order_by="created_at desc")
last_job = next(jobs)

# Ruta del output del job
model_path = last_job.outputs["output"].uri

# Registrar el modelo
registered_model = ml_client.models.create_or_update(
    Model(
        name="adapted-cv-model",
        path=model_path,
        type=AssetTypes.CUSTOM_MODEL,
        description="Modelo fine-tuned para generar CVs adaptados",
    )
)

print(f"✅ Modelo registrado: {registered_model.name}, versión {registered_model.version}")
