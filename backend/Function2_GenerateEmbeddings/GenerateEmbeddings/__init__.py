import logging
import os
import json
from openai import AzureOpenAI  
import requests
import fitz 
from azure.storage.blob import BlobClient
from azure.core.credentials import AzureKeyCredential
from azure.cosmos import CosmosClient, PartitionKey
import azure.functions as func

# Config vars
AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
cosmos_url = os.environ["COSMOS_URL"]
cosmos_key = os.environ["COSMOS_KEY"]
cosmos_db = os.environ["COSMOS_DB"]
cosmos_container = os.environ["COSMOS_CONTAINER"]
STORAGE_ACCOUNT_NAME = os.environ["STORAGE_ACCOUNT_NAME"]
STORAGE_SAS_TOKEN = os.environ["BLOB_SAS_TOKEN"]

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

cosmos_client = CosmosClient(cosmos_url, credential=cosmos_key)
db = cosmos_client.get_database_client(cosmos_db)
container = db.get_container_client(cosmos_container)

def extract_text_from_pdf_bytes(pdf_bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def download_blob_text(blob_url):
    full_url = f"{blob_url}?{STORAGE_SAS_TOKEN}"
    response = requests.get(full_url)
    response.raise_for_status()
    return response.text

def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model=OPENAI_DEPLOYMENT
    )
    return response.data[0].embedding

def insert_into_cosmos(doc_id, text, embedding, doc_type):
    document = {
        "id": f"{doc_id}-{doc_type}",
        "text": text,
        "embedding": embedding,
        "type": doc_type
    }
    container.upsert_item(document)
    logging.info(f"Documento subido a Cosmos DB")

def main(event: func.EventGridEvent):
    logging.info("Evento recibido")
    
    try:
        # Obtener datos del evento
        event_data = event.get_json()
        blob_url = event_data["url"]
        
        filename = blob_url.split("/")[-1]
        job_id = filename.split("-")[1]

        job_offer_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/upload/joboffer/jobOffer-{job_id}.txt"

        # Procesar archivos
        cv_text = extract_text_from_pdf_bytes(requests.get(blob_url + "?" + STORAGE_SAS_TOKEN).content)
        job_offer_text = download_blob_text(job_offer_url)

        # Generar embeddings
        cv_embedding = generate_embedding(cv_text)
        job_embedding = generate_embedding(job_offer_text)

        # Insertar en Cosmos DB
        insert_into_cosmos(job_id, cv_text, cv_embedding, "cv")
        insert_into_cosmos(job_id, job_offer_text, job_embedding, "joboffer")
        
        logging.info(f"Procesamiento completado para job_id: {job_id}")

    except Exception as e:
        logging.error(f"Error procesando los blobs: {e}")
        raise e