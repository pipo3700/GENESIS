import logging, os, json, time, requests, fitz
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
import azure.functions as func
from io import BytesIO
from numpy import dot
from numpy.linalg import norm
from reportlab.pdfgen import canvas

# Configuración
client = AzureOpenAI(
    api_version="2024-12-01",  
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"]
)
cosmos = CosmosClient(os.environ["COSMOS_URL"], credential=os.environ["COSMOS_KEY"])
db = cosmos.get_database_client(os.environ["COSMOS_DB"])
container = db.get_container_client(os.environ["COSMOS_CONTAINER"])
blob_service = BlobServiceClient.from_connection_string(os.environ["STORAGE_CONNECTION_STRING"])

ACCOUNT = os.environ["STORAGE_ACCOUNT_NAME"]
SAS = os.environ["BLOB_SAS_TOKEN"]

def cosine_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def wait_for_embeddings(job_id, retries=10, delay=3):
    for _ in range(retries):
        try:
            cv = container.read_item(f"{job_id}-cv", partition_key=f"{job_id}-cv")
            job = container.read_item(f"{job_id}-joboffer", partition_key=f"{job_id}-joboffer")
            return cv["text"], job["text"], cv["embedding"], job["embedding"]
        except:
            time.sleep(delay)
    raise Exception("Embeddings no disponibles aún")

def generate_pdf(text):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica", 10)
    y = 800
    for line in text.split("\n"):
        if y < 40: p.showPage(); y = 800
        p.drawString(50, y, line[:1000])
        y -= 15
    p.save()
    buffer.seek(0)
    return buffer

def upload_pdf(pdf_stream, job_id):
    blob = blob_service.get_blob_client("upload", f"generated/{job_id}.pdf")
    blob.upload_blob(pdf_stream, overwrite=True, content_type="application/pdf")
    return f"https://{ACCOUNT}.blob.core.windows.net/upload/generated/{job_id}.pdf"

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Function 3 — GenerateAdaptedCV ejecutándose...")
    
    # Headers CORS
    cors_headers = {
        "Access-Control-Allow-Origin": "https://red-sand-04619bc10.6.azurestaticapps.net",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Ocp-Apim-Subscription-Key",
        "Access-Control-Max-Age": "86400"
    }
    
    # Manejar preflight request (OPTIONS)
    if req.method == "OPTIONS":
        return func.HttpResponse(
            "",
            status_code=200,
            headers=cors_headers
        )
    
    try:
        job_id = req.get_json().get("jobId")
        if not job_id: 
            return func.HttpResponse(
                "Falta jobId", 
                status_code=400,
                headers=cors_headers
            )

        cv_text, job_text, cv_embed, job_embed = wait_for_embeddings(job_id)
        sim = cosine_sim(cv_embed, job_embed)

        prompt = f"""
Eres un asistente experto en RRHH. Adapta el CV original a la oferta de trabajo resaltando los puntos relevantes.

Similitud cosenoidal: {sim:.2f}

--- CV ORIGINAL ---
{cv_text}

--- OFERTA ---
{job_text}

Genera el CV adaptado:
"""

        response = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        new_cv = response.choices[0].message.content
        pdf = generate_pdf(new_cv)
        url = upload_pdf(pdf, job_id)

        return func.HttpResponse(
            json.dumps({"generatedCvUrl": url}), 
            mimetype="application/json", 
            status_code=200,
            headers=cors_headers
        )

    except Exception as e:
        logging.error(f"Error en Function 3: {e}")
        return func.HttpResponse(
            f"Error interno: {str(e)}", 
            status_code=500,
            headers=cors_headers
        )