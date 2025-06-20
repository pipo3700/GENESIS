import os
import mlflow
import mlflow.transformers
import mlflow.pytorch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model

# 💡 Limpiar variables si existen para forzar el uso del token federado
for var in ["AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_CLIENT_SECRET"]:
    os.environ.pop(var, None)

# Configuración desde entorno
MODEL_NAME = os.getenv("HUGGINGFACE_MODEL")
DATASET_PATH = os.getenv("DATASET_PATH")
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")

# Cliente Azure ML con token federado (gracias a azure/login@v2)
ml_client = MLClient(DefaultAzureCredential(), AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME)

# Intentar cargar el último modelo registrado
try:
    model_list = list(ml_client.models.list(name="genesis-model"))
    model_list = sorted(model_list, key=lambda x: x.version, reverse=True)
    latest_model_path = model_list[0].path
    print(f"Cargando modelo más reciente desde Azure ML: {latest_model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(latest_model_path)
except Exception as e:
    print("No hay modelo previo. Usando modelo base de Hugging Face.")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Tokenizador
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(example):
    inputs = tokenizer(example["input"], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(example["target"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(preprocess, remove_columns=["input", "target"])

# Entrenamiento
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

# MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("FineTuningGenesis")

with mlflow.start_run():
    mlflow.log_param("model", MODEL_NAME)
    mlflow.log_param("epochs", training_args.num_train_epochs)
    trainer.train()
    metrics = trainer.evaluate()
    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(trainer.model, artifact_path="model")
    trainer.save_model("./model")
