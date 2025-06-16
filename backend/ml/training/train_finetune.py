# ml/training/train_finetune.py

import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

MODEL_NAME = os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-base")
DATASET_URL = os.getenv("BLOB_DATASET_URL")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(example):
    inputs = tokenizer(example["input"], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(example["target"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Dataset desde Blob Storage
dataset = load_dataset('json', data_files=DATASET_URL, split='train')
dataset = dataset.map(preprocess, remove_columns=["input", "target"])

# Modelo base
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Entrenamiento
training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    num_train_epochs=1,            # 🔸 Solo 1 época para pruebas
    logging_steps=10,
    save_strategy="no",
    max_steps=100,                 # 🔸 Límite absoluto de pasos
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# Guardar modelo y tokenizer para el registro posterior
model.save_pretrained("./outputs")
tokenizer.save_pretrained("./outputs")
