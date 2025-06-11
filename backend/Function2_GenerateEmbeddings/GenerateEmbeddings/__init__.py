import logging
import os
import json
import openai
import requests
import fitz 
from azure.storage.blob import BlobClient
from azure.core.credentials import AzureKeyCredential
from azure.cosmos import CosmosClient, PartitionKey
import azure.functions as func

app = func.FunctionApp()

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
    openai.api_key = AZURE_OPENAI_KEY
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_type = "azure"
    openai.api_version = "2023-05-15"

    embedding = openai.Embedding.create(
        input=text,
        engine=OPENAI_DEPLOYMENT
    )["data"][0]["embedding"]
    return embedding

def insert_into_cosmos(doc_id, text, embedding, doc_type):
    document = {
        "id": f"{doc_id}-{doc_type}",
        "text": text,
        "embedding": embedding,
        "type": doc_type
    }
    container.upsert_item(document)
    logging.info(f"Documento subido a Cosmos DB")

@app.event_grid_trigger(arg_name="azeventgrid")
def GenerateEmbeddings(azeventgrid: func.EventGridEvent):
    logging.info("Evento recibido")
    event_data = json.loads(azeventgrid.get_body())
    blob_url = event_data["url"]
    
    filename = blob_url.split("/")[-1]
    job_id = filename.split("-")[1]

    job_offer_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/upload/joboffer/jobOffer-{job_id}.txt"

    try:
        cv_text = extract_text_from_pdf_bytes(requests.get(blob_url + "?" + STORAGE_SAS_TOKEN).content)
        job_offer_text = download_blob_text(job_offer_url)

        cv_embedding = generate_embedding(cv_text)
        job_embedding = generate_embedding(job_offer_text)

        insert_into_cosmos(job_id, cv_text, cv_embedding, "cv")
        insert_into_cosmos(job_id, job_offer_text, job_embedding, "joboffer")

    except Exception as e:
        logging.error(f"Error procesando los blobs: {e}")
