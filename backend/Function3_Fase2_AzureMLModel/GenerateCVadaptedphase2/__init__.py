import tempfile
import logging, os, json, time, base64, fitz
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from io import BytesIO
from numpy import dot
from numpy.linalg import norm
from reportlab.pdfgen import canvas
import azure.functions as func
import torch

_cosmos = None
_container = None
_blob_service = None
_model = None
_tokenizer = None
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
            credential = ManagedIdentityCredential()
            _ml_client = MLClient(
                credential=credential,
                subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
                resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
                workspace_name=os.environ["AZURE_WORKSPACE_NAME"]
            )
        except Exception as e:
            logging.warning(f"Managed Identity failed: {e}")
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

def load_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        model_path = get_latest_registered_model()
        model_dir = os.path.join(model_path, "genesis-model", "model")
        logging.info(f"📁 Contenido de {model_dir}: {os.listdir(model_dir)}")
        _tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
    return _model, _tokenizer

def get_latest_registered_model():
    try:
        ml_client = get_ml_client()
        models = ml_client.models.list(name="genesis-model")
        latest_model = max(models, key=lambda m: m.version)
        logging.info(f"✅ Usando modelo registrado: {latest_model.name} v{latest_model.version}")
        temp_dir = tempfile.mkdtemp()
        ml_client.models.download(name=latest_model.name, version=latest_model.version, download_path=temp_dir)
        logging.info(f"📦 Modelo descargado en: {temp_dir}")
        return temp_dir
    except Exception as e:
        logging.error(f"Failed to get ML model: {e}")
        fallback_path = os.environ.get("MODEL_PATH", "google/flan-t5-small")
        logging.warning(f"Using fallback model: {fallback_path}")
        return fallback_path

def generate_adapted_cv_prompt(cv_text, job_text, sim):
    """Genera un prompt más directo para adaptar el CV"""
    prompt = f"""Reescribe este currículum adaptándolo para la siguiente oferta de trabajo. Solo reorganiza y resalta la información más relevante sin inventar nada nuevo.

OFERTA DE TRABAJO:
{job_text}

CURRÍCULUM ORIGINAL:
{cv_text}

CURRÍCULUM ADAPTADO:"""
    return prompt

def extract_adapted_cv(generated_text, prompt):
    """Extrae el CV adaptado de la respuesta del modelo"""
    
    # Método 1: Buscar después del último marcador
    markers = ["CURRÍCULUM ADAPTADO:", "CV ADAPTADO:", "--- CV ADAPTADO ---"]
    for marker in markers:
        if marker in generated_text:
            parts = generated_text.split(marker)
            if len(parts) > 1:
                result = parts[-1].strip()
                if result and len(result) > 50:  # Validar que no esté vacío
                    return result
    
    # Método 2: Remover el prompt original si está presente
    lines = generated_text.split('\n')
    prompt_lines = set(prompt.split('\n'))
    
    adapted_lines = []
    found_content = False
    
    for line in lines:
        # Saltar líneas que son parte del prompt
        if line.strip() in prompt_lines or line.strip() in ['OFERTA DE TRABAJO:', 'CURRÍCULUM ORIGINAL:', 'CURRÍCULUM ADAPTADO:']:
            found_content = True
            continue
        
        # Si encontramos contenido después del prompt, agregarlo
        if found_content and line.strip():
            adapted_lines.append(line)
    
    if adapted_lines:
        return '\n'.join(adapted_lines)
    
    # Método 3: Fallback - buscar contenido que no sea del prompt original
    clean_lines = []
    for line in lines:
        if line.strip() and not any(keyword in line.lower() for keyword in ['oferta de trabajo', 'currículum original', 'reescribe']):
            clean_lines.append(line)
    
    if clean_lines:
        return '\n'.join(clean_lines)
    
    # Método 4: Último recurso - devolver todo
    return generated_text.strip()

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

        # Usar el nuevo prompt más directo
        prompt = generate_adapted_cv_prompt(cv_text, job_text, sim)
        
        model, tokenizer = load_model_and_tokenizer()

        logging.info("🧠 Ejecutando inferencia con generate()...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Parámetros mejorados para evitar repetición del prompt
        outputs = model.generate(
            **inputs, 
            max_length=1024, 
            do_sample=True,           # Cambiar a True para más variabilidad
            temperature=0.7,          # Añadir temperatura
            top_p=0.9,               # Añadir nucleus sampling
            repetition_penalty=1.2,   # Penalizar repeticiones
            no_repeat_ngram_size=3,   # Evitar repetir n-gramas
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
        )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Usar la función mejorada para extraer el CV
        new_cv = extract_adapted_cv(generated, prompt)
        
        logging.info(f"✅ Prompt usado:\n{prompt[:200]}...")
        logging.info(f"✅ Texto generado completo:\n{generated[:500]}...")
        logging.info(f"✅ CV adaptado extraído:\n{new_cv[:300]}...")
        
        # Validar que el resultado sea válido
        if len(new_cv) < 100:
            logging.warning("CV adaptado muy corto, usando fallback")
            # Crear un CV básico adaptado manualmente
            new_cv = f"""Nombre: {cv_text.split('Nombre:')[1].split('Email:')[0].strip() if 'Nombre:' in cv_text else 'Candidato'}
Email: {cv_text.split('Email:')[1].split('Teléfono:')[0].strip() if 'Email:' in cv_text else 'email@ejemplo.com'}

PERFIL PROFESIONAL:
Candidato con experiencia relevante para la posición ofertada.

EXPERIENCIA DESTACADA:
{cv_text.split('Experiencia:')[1].split('Habilidades:')[0].strip() if 'Experiencia:' in cv_text else 'Experiencia profesional relevante'}

HABILIDADES CLAVE:
{cv_text.split('Habilidades:')[1].split('Educación:')[0].strip() if 'Habilidades:' in cv_text else 'Habilidades técnicas'}

FORMACIÓN:
{cv_text.split('Educación:')[1].strip() if 'Educación:' in cv_text else 'Formación académica'}"""
        
        # Verificar que no contenga el prompt original
        if any(keyword in new_cv.lower() for keyword in ['oferta de trabajo:', 'currículum original:', 'reescribe']):
            logging.warning("El CV contiene partes del prompt, limpiando...")
            lines = new_cv.split('\n')
            clean_lines = [line for line in lines if not any(keyword in line.lower() for keyword in ['oferta de trabajo', 'currículum original', 'reescribe'])]
            new_cv = '\n'.join(clean_lines)
        
        pdf = generate_pdf(new_cv)
        url = upload_pdf(pdf, job_id)

        return func.HttpResponse(
            body=json.dumps({"generatedCvUrl": url}),
            status_code=200,
            mimetype="application/json",
            headers=cors_headers
        )

    except Exception as e:
        logging.error(f"Error Function 3: {e}")
        return func.HttpResponse(
            body=json.dumps({"error": str(e)}),
            status_code=500,
            headers=cors_headers,
            mimetype="application/json"
        )