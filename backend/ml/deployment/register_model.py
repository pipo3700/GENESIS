import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from pathlib import Path

# Configuración desde entorno
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")
MODEL_NAME = os.getenv("MODEL_NAME", "genesis-model")
MODEL_PATH = "./model"

# Verificación de la carpeta del modelo
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model directory not found: {MODEL_PATH}")

# Asegurar que contiene los archivos necesarios
required_files = ["config.json", "pytorch_model.bin"]
missing_files = [f for f in required_files if not os.path.exists(os.path.join(MODEL_PATH, f))]

if missing_files:
    print(f"⚠️ Warning: missing model files: {missing_files}")
    print("📁 Contents:")
    for f in os.listdir(MODEL_PATH):
        print("   -", f)
else:
    print(f"✅ Model directory is complete: {MODEL_PATH}")

# Inicializar cliente de Azure ML
credential = DefaultAzureCredential(exclude_environment_credential=True)
ml_client = MLClient(
    credential,
    subscription_id=AZURE_SUBSCRIPTION_ID,
    resource_group_name=AZURE_RESOURCE_GROUP,
    workspace_name=AZURE_WORKSPACE_NAME
)

# Registrar como modelo Hugging Face (custom_model, pero con estructura HF)
model = Model(
    path=MODEL_PATH,
    name=MODEL_NAME,
    description="Modelo fine-tuneado Hugging Face compatible",
    type="custom_model",  # ¡Correcto para HF!
    tags={
        "framework": "transformers",
        "hf_compatible": "true"
    }
)

# Registro en Azure ML
try:
    registered_model = ml_client.models.create_or_update(model)
    print(f"✅ Modelo registrado exitosamente:")
    print(f"   - Nombre: {registered_model.name}")
    print(f"   - Versión: {registered_model.version}")
    print(f"   - ID: {registered_model.id}")
except Exception as e:
    print(f"❌ Error al registrar el modelo: {e}")
    raise
