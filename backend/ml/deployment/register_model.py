# register_model.py
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")
MODEL_NAME = "genesis-model"
MODEL_PATH = "./model"

ml_client = MLClient(DefaultAzureCredential(), AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME)

model = ml_client.models.create_or_update(
    Model(
        path=MODEL_PATH,
        name=MODEL_NAME,
        description="Modelo fine-tuneado desde Hugging Face",
        type="custom_model",
    )
)

print(f"Modelo registrado: {model.name}, versión: {model.version}")
