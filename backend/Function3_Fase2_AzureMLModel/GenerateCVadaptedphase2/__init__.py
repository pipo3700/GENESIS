import tempfile
import logging, os, json, time, base64, fitz
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from io import BytesIO
from numpy import dot
from numpy.linalg import norm
from reportlab.pdfgen import canvas
import azure.functions as func

# Variables globales
_cosmos = None
_container = None
_blob_service = None
_pipe = None
_ml_client = None

def fix_cosmos_key(key):
    key = key.strip()
    missing_padding = len(key) % 4
    if missing_padding:
        key += '=' * (4 - missing_padding)
    base64.b64decode(key)
    return key

def get_cosmos_client():
    global _cosmos, _container
    if _cosmos is None:
        cosmos_key = fix_cosmos_key(os.environ["COSMOS_KEY"])
        _cosmos = CosmosClient(os.environ["COSMOS_URL"], credential=cosmos_key)
        db = _cosmos.get_database_client(os.environ["COSMOS_DB"])
        _container = db.get_container_client(os.environ["COSMOS_CONTAINER"])
    return _cosmos, _container

def get_blob_service():
    global _blob_service
    if _blob_service is None:
        _blob_service = BlobServiceClient.from_connection_string(os.environ["STORAGE_CONNECTION_STRING"])
    return _blob_service

def get_ml_client():
    global _ml_client
    if _ml_client is None:
        try:
            # Try Managed Identity first
            credential = ManagedIdentityCredential()
            _ml_client = MLClient(
                credential=credential,
                subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
                resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
                workspace_name=os.environ["AZURE_WORKSPACE_NAME"]
            )
        except Exception as e:
            logging.warning(f"Managed Identity failed: {e}")
            # Fallback to service principal if environment variables are set
            try:
                from azure.identity import ClientSecretCredential
                credential = ClientSecretCredential(
                    tenant_id=os.environ["AZURE_TENANT_ID"],
                    client_id=os.environ["AZURE_CLIENT_ID"],
                    client_secret=os.environ["AZURE_CLIENT_SECRET"]
                )
                _ml_client = MLClient(
                    credential=credential,
                    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
                    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
                    workspace_name=os.environ["AZURE_WORKSPACE_NAME"]
                )
            except Exception as e2:
                logging.error(f"Service Principal auth also failed: {e2}")
                raise Exception("Unable to authenticate with Azure ML")
    return _ml_client

def cosine_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def wait_for_embeddings(job_id, retries=10, delay=3):
    _, container = get_cosmos_client()
    for attempt in range(retries):
        try:
            cv = container.read_item(f"{job_id}-cv", partition_key=f"{job_id}-cv")
            job = container.read_item(f"{job_id}-joboffer", partition_key=f"{job_id}-joboffer")
            return cv["text"], job["text"], cv["embedding"], job["embedding"]
        except:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise Exception("Embeddings no disponibles")

def generate_pdf(text):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica", 10)
    y = 800
    for line in text.split("\n"):
        while len(line) > 100:
            if y < 40:
                p.showPage()
                y = 800
            p.drawString(50, y, line[:100])
            y -= 15
            line = line[100:]
        if line.strip():
            if y < 40:
                p.showPage()
                y = 800
            p.drawString(50, y, line)
            y -= 15
    p.save()
    buffer.seek(0)
    return buffer

def upload_pdf(pdf_stream, job_id):
    blob_service = get_blob_service()
    blob_name = f"generated/{job_id}.pdf"
    blob = blob_service.get_blob_client("upload", blob_name)
    blob.upload_blob(pdf_stream, overwrite=True, content_type="application/pdf")
    account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    sas_token = os.environ["BLOB_SAS_TOKEN"]
    return f"https://{account_name}.blob.core.windows.net/upload/{blob_name}?{sas_token}"

def get_latest_registered_model():
    try:
        ml_client = get_ml_client()
        models = ml_client.models.list(name="genesis-model")
        latest_model = max(models, key=lambda m: m.version)
        logging.info(f"✅ Usando modelo registrado: {latest_model.name} v{latest_model.version}")

        # Descargar modelo a una carpeta temporal
        temp_dir = tempfile.mkdtemp()
        ml_client.models.download(name=latest_model.name, version=latest_model.version, download_path=temp_dir)
        logging.info(f"📦 Modelo descargado en: {temp_dir}")
        return temp_dir
    except Exception as e:
        logging.error(f"Failed to get ML model: {e}")
        fallback_path = os.environ.get("MODEL_PATH", "google/flan-t5-small")
        logging.warning(f"Using fallback model: {fallback_path}")
        return fallback_path


def get_model_pipeline():
    global _pipe
    if _pipe is None:
        try:
            model_path = get_latest_registered_model()
            model_dir = os.path.join(model_path, "model")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

            _pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        except Exception as e:
            logging.error(f"Failed to load custom model: {e}")
            logging.warning("Using fallback public model")
            _pipe = pipeline("text2text-generation", model="google/flan-t5-small")
    return _pipe


def main(req: func.HttpRequest) -> func.HttpResponse:
    cors_headers = {
        "Access-Control-Allow-Origin": "https://red-sand-04619bc10.6.azurestaticapps.net",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Ocp-Apim-Subscription-Key",
        "Access-Control-Max-Age": "86400"
    }

    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=200, headers=cors_headers)

    try:
        request_json = req.get_json()
        job_id = request_json.get("jobId")
        if not job_id:
            raise ValueError("Falta jobId")

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

        pipe = get_model_pipeline()
        logging.info(" Pipeline cargado, iniciando inferencia...")
        result = pipe(prompt, max_length=1024, do_sample=False)
        logging.info(f" Resultado del modelo: {result}")
        new_cv = result[0]["generated_text"]

        pdf = generate_pdf(new_cv)
        url = upload_pdf(pdf, job_id)
        

        return func.HttpResponse(
            json.dumps({"generatedCvUrl": url}),
            mimetype="application/json",
            status_code=200,
            headers=cors_headers
        )

    except Exception as e:
        logging.error(f"Error Function 3: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            headers=cors_headers,
            mimetype="application/json"
        )