import os
import mlflow
import mlflow.pytorch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from pathlib import Path

# Cargar configuración desde variables de entorno
MODEL_NAME = os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-base")
DATASET_PATH = os.getenv("DATASET_PATH")
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")

# Inicializar cliente de Azure ML usando federated identity (GitHub Actions)
try:
    credential = DefaultAzureCredential(exclude_environment_credential=True)
    ml_client = MLClient(
        credential=credential,
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME
    )
except Exception as e:
    raise RuntimeError(f"❌ Error autenticando con Azure ML: {e}")

# Intentar cargar el último modelo desde Azure ML
download_path = Path("downloaded_model")
try:
    models = list(ml_client.models.list(name="genesis-model"))
    if models:
        latest_model = sorted(models, key=lambda m: m.version, reverse=True)[0]
        ml_client.models.download(
            name=latest_model.name, 
            version=latest_model.version, 
            download_path=download_path
        )
        print(f"✅ Modelo descargado desde Azure ML en: {download_path}")
        model_dir = download_path / "model" if (download_path / "model").exists() else download_path
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    else:
        raise Exception("No se encontró ninguna versión registrada")
except Exception as e:
    print(f"⚠️ No se pudo recuperar un modelo registrado. Motivo: {e}")
    print("➡️ Se utilizará el modelo base de Hugging Face.")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Tokenizador
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Preprocesar dataset
def preprocess(example):
    inputs = tokenizer(example["input"], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(example["target"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(preprocess, remove_columns=["input", "target"])

# Configurar entrenamiento
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=500,
    save_total_limit=1,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
)

# Tracking con MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("FineTuningGenesis")

with mlflow.start_run():
    mlflow.log_param("model", MODEL_NAME)
    mlflow.log_param("epochs", training_args.num_train_epochs)
    
    trainer.train()
    metrics = trainer.evaluate()
    mlflow.log_metrics(metrics)
    
    # Guardar y exportar modelo
    trainer.save_model("./model")
    mlflow.pytorch.log_model(trainer.model, artifact_path="model")

print("✅ Fine-tuning y guardado completado correctamente.")
