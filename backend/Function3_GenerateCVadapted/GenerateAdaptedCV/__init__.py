import logging, os, json, time, requests, fitz, base64
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
import azure.functions as func
from io import BytesIO
from numpy import dot
from numpy.linalg import norm
from reportlab.pdfgen import canvas

# Variables globales para inicialización lazy
_client = None
_cosmos = None
_container = None
_blob_service = None

def fix_cosmos_key(key):
    """Corrige el padding de la clave Base64 de Cosmos DB"""
    try:
        key = key.strip()
        missing_padding = len(key) % 4
        if missing_padding:
            key += '=' * (4 - missing_padding)
        base64.b64decode(key)  # Verificar que es válida
        return key
    except Exception as e:
        logging.error(f"Error corrigiendo clave Cosmos: {e}")
        raise Exception(f"Clave Cosmos DB inválida: {str(e)}")

def get_openai_client():
    """Inicializa el cliente OpenAI de forma lazy"""
    global _client
    if _client is None:
        try:
            _client = AzureOpenAI(
                api_version="2024-02-01",
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_KEY"]
            )
            logging.info("Cliente OpenAI inicializado correctamente")
        except Exception as e:
            logging.error(f"Error inicializando OpenAI: {e}")
            raise Exception(f"Error inicializando OpenAI: {str(e)}")
    return _client

def get_cosmos_client():
    """Inicializa el cliente Cosmos de forma lazy"""
    global _cosmos, _container
    if _cosmos is None:
        try:
            cosmos_key = fix_cosmos_key(os.environ["COSMOS_KEY"])
            _cosmos = CosmosClient(os.environ["COSMOS_URL"], credential=cosmos_key)
            db = _cosmos.get_database_client(os.environ["COSMOS_DB"])
            _container = db.get_container_client(os.environ["COSMOS_CONTAINER"])
            logging.info("Cliente Cosmos inicializado correctamente")
        except Exception as e:
            logging.error(f"Error inicializando Cosmos: {e}")
            raise Exception(f"Error inicializando Cosmos: {str(e)}")
    return _cosmos, _container

def get_blob_service():
    """Inicializa el servicio de blob de forma lazy"""
    global _blob_service
    if _blob_service is None:
        try:
            _blob_service = BlobServiceClient.from_connection_string(
                os.environ["STORAGE_CONNECTION_STRING"]
            )
            logging.info("Servicio Blob inicializado correctamente")
        except Exception as e:
            logging.error(f"Error inicializando Blob Service: {e}")
            raise Exception(f"Error inicializando Blob Service: {str(e)}")
    return _blob_service

def cosine_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def wait_for_embeddings(job_id, retries=10, delay=3):
    _, container = get_cosmos_client()
    for attempt in range(retries):
        try:
            cv = container.read_item(f"{job_id}-cv", partition_key=f"{job_id}-cv")
            job = container.read_item(f"{job_id}-joboffer", partition_key=f"{job_id}-joboffer")
            return cv["text"], job["text"], cv["embedding"], job["embedding"]
        except Exception as e:
            logging.warning(f"Intento {attempt + 1}/{retries} fallido: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise Exception(f"Embeddings no disponibles después de {retries} intentos")

def generate_pdf(text):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica", 10)
    y = 800
    
    # Mejorar la generación del PDF
    lines = text.split("\n")
    for line in lines:
        # Dividir líneas muy largas
        while len(line) > 100:
            if y < 40:
                p.showPage()
                y = 800
            p.drawString(50, y, line[:100])
            y -= 15
            line = line[100:]
        
        if line.strip():  # Solo si la línea no está vacía
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
    try:
        blob_name = f"generated/{job_id}.pdf"
        blob = blob_service.get_blob_client("upload", blob_name)
        blob.upload_blob(pdf_stream, overwrite=True, content_type="application/pdf")

        account_name = os.environ["STORAGE_ACCOUNT_NAME"]
        sas_token = os.environ["BLOB_SAS_TOKEN"] 
        url = f"https://{account_name}.blob.core.windows.net/upload/{blob_name}?{sas_token}"
        return url
    except Exception as e:
        logging.error(f"Error subiendo PDF: {e}")
        raise Exception(f"Error subiendo PDF: {str(e)}")

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
        # Verificar que el request tiene datos
        try:
            request_json = req.get_json()
            if not request_json:
                raise ValueError("Request body vacío")
            job_id = request_json.get("jobId")
        except Exception as e:
            logging.error(f"Error procesando request: {e}")
            return func.HttpResponse(
                json.dumps({"error": "Request inválido"}),
                status_code=400,
                headers=cors_headers,
                mimetype="application/json"
            )
        
        if not job_id:
            return func.HttpResponse(
                json.dumps({"error": "Falta jobId"}),
                status_code=400,
                headers=cors_headers,
                mimetype="application/json"
            )
        
        logging.info(f"Procesando job_id: {job_id}")
        
        # Obtener embeddings
        try:
            cv_text, job_text, cv_embed, job_embed = wait_for_embeddings(job_id)
            sim = cosine_sim(cv_embed, job_embed)
            logging.info(f"Similaridad calculada: {sim:.2f}")
        except Exception as e:
            logging.error(f"Error obteniendo embeddings: {e}")
            return func.HttpResponse(
                json.dumps({"error": f"Error obteniendo datos: {str(e)}"}),
                status_code=500,
                headers=cors_headers,
                mimetype="application/json"
            )
        
        # Generar prompt
        prompt = f"""
Eres un asistente experto en RRHH. Adapta el CV original a la oferta de trabajo resaltando los puntos relevantes, sin inventarte nada.

Similitud cosenoidal: {sim:.2f}

--- CV ORIGINAL ---
{cv_text}

--- OFERTA ---
{job_text}

Genera el CV adaptado:
"""
        
        # Llamar a OpenAI
        try:
            client = get_openai_client()
            response = client.chat.completions.create(
                model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            new_cv = response.choices[0].message.content
            logging.info("CV generado correctamente")
        except Exception as e:
            logging.error(f"Error en OpenAI: {e}")
            return func.HttpResponse(
                json.dumps({"error": f"Error generando CV: {str(e)}"}),
                status_code=500,
                headers=cors_headers,
                mimetype="application/json"
            )
        
        # Generar y subir PDF
        try:
            pdf = generate_pdf(new_cv)
            url = upload_pdf(pdf, job_id)
            logging.info(f"PDF generado y subido: {url}")
        except Exception as e:
            logging.error(f"Error generando/subiendo PDF: {e}")
            return func.HttpResponse(
                json.dumps({"error": f"Error creando PDF: {str(e)}"}),
                status_code=500,
                headers=cors_headers,
                mimetype="application/json"
            )
        
        return func.HttpResponse(
            json.dumps({"generatedCvUrl": url}),
            mimetype="application/json",
            status_code=200,
            headers=cors_headers
        )
    
    except Exception as e:
        logging.error(f"Error general en Function 3: {e}")
        return func.HttpResponse(
            json.dumps({"error": f"Error interno: {str(e)}"}),
            status_code=500,
            headers=cors_headers,
            mimetype="application/json"
        )