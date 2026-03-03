import os
import html
from fastapi import FastAPI, Form
from fastapi.responses import Response
from openai import OpenAI, AuthenticationError, RateLimitError, OpenAIError
from twilio.twiml.messaging_response import MessagingResponse

app = FastAPI()

# ====== CONFIG (cámbialo aquí) ======
# 1) MODELO GPT: cambia este valor para usar otro modelo
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# 2) Vector Store (RAG). Si no lo tienes o falla, puedes desactivar RAG con USE_RAG=0
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID", "")
USE_RAG = os.getenv("USE_RAG", "1") == "1"

SYSTEM_PROMPT = """
Eres el asistente técnico oficial de GMI Dental Implantology.
Responde siempre basándote en los documentos cargados (implantes, líneas, indicaciones, superficies, protocolos).
Si una información no está en los documentos, dilo claramente.
Habla siempre en español.
Responde de manera clara, profesional y en no más de 4 frases.
Tu audiencia son dentistas, clínicas y técnicos.
""".strip()
# ====================================

# Healthcheck
@app.get("/")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "use_rag": USE_RAG}

# Leer API Key desde variable de entorno
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Falta la variable OPENAI_API_KEY en Railway.")

client = OpenAI(api_key=api_key)

@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(...)
):
    user_text = (Body or "").strip()

    try:
        # Construye la llamada a OpenAI
        kwargs = dict(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            max_output_tokens=220,
        )

        # Activa RAG solo si está habilitado y hay VECTOR_STORE_ID
        if USE_RAG and VECTOR_STORE_ID:
            kwargs["tools"] = [{
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID],
                "max_num_results": 5
            }]

        response = client.responses.create(**kwargs)

        reply_text = (response.output_text or "").strip()
        if not reply_text:
            reply_text = "No he podido generar respuesta. ¿Puedes reformular la pregunta?"

    except RateLimitError as e:
        print("RateLimitError:", repr(e))
        reply_text = "Estoy procesando muchas consultas en este momento. Intenta de nuevo en unos minutos."

    except AuthenticationError as e:
        print("AuthenticationError:", repr(e))
        reply_text = "Hay un problema interno con la clave de IA. Por favor, contacta con soporte técnico."

    except OpenAIError as e:
        # Este log es CLAVE para diagnosticar en Railway
        print("OpenAIError:", repr(e))
        reply_text = "He tenido un problema procesando tu mensaje. Inténtalo nuevamente."

    except Exception as e:
        # Por si hay cualquier otro fallo (Twilio / FastAPI / etc.)
        print("UnhandledError:", repr(e))
        reply_text = "Error interno inesperado. Inténtalo nuevamente."

    # Generar TwiML de forma robusta
    twiml = MessagingResponse()
    twiml.message(reply_text)

    return Response(content=str(twiml), media_type="text/xml")
