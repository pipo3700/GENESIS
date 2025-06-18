# register_model.py
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model

# Configuración desde entorno
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP") 
AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")
MODEL_NAME = "genesis-model"
MODEL_PATH = "./model"

# ✅ Verificar que el directorio del modelo existe
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model directory not found: {MODEL_PATH}")

# ✅ Verificar que contiene archivos del modelo
required_files = ["config.json", "pytorch_model.bin"]
missing_files = [f for f in required_files if not os.path.exists(os.path.join(MODEL_PATH, f))]

if missing_files:
    print(f"⚠️  Warning: Some model files might be missing: {missing_files}")
    print("📁 Available files in model directory:")
    for file in os.listdir(MODEL_PATH):
        print(f"   - {file}")

print(f"✅ Model directory found: {MODEL_PATH}")

# Cliente Azure ML
ml_client = MLClient(
    DefaultAzureCredential(), 
    AZURE_SUBSCRIPTION_ID, 
    AZURE_RESOURCE_GROUP, 
    AZURE_WORKSPACE_NAME
)

try:
    # Registrar el modelo
    model = ml_client.models.create_or_update(
        Model(
            path=MODEL_PATH,
            name=MODEL_NAME,
            description="Modelo fine-tuneado desde Hugging Face",
            type="custom_model",
        )
    )
    
    print(f"✅ Modelo registrado exitosamente:")
    print(f"   - Nombre: {model.name}")
    print(f"   - Versión: {model.version}")
    print(f"   - ID: {model.id}")
    
except Exception as e:
    print(f"❌ Error al registrar el modelo: {str(e)}")
    raise