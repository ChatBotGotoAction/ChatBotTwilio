import os
from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from openai import OpenAI, AuthenticationError, RateLimitError, OpenAIError

app = FastAPI()

# Leer API Key desde variable de entorno
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Falta la variable OPENAI_API_KEY en Railway.")

client = OpenAI(api_key=api_key)

# ID del vector store generado por carga_docs.py
VECTOR_STORE_ID = "vs_694774fa6ead48191a261aa04e11b7868"

SYSTEM_PROMPT = """
Eres el asistente técnico oficial de GMI Dental Implantology.
Responde siempre basándote en los documentos cargados (implantes, líneas, indicaciones, superficies, protocolos).
Si una información no está en los documentos, dilo claramente.
Habla siempre en español.
Responde de manera clara, profesional y en no más de 4 frases.
Tu audiencia son dentistas, clínicas y técnicos.
"""

@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(...)
):

    try:
        response = client.responses.create(
            model="gpt-5.1",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": Body},
            ],
            tools=[{
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID],
                "max_num_results": 5
            }],
            max_output_tokens=220
        )

        reply_text = response.output_text.strip()

    except RateLimitError:
        reply_text = "Estoy procesando muchas consultas en este momento. Intenta de nuevo en unos minutos."

    except AuthenticationError:
        reply_text = "Hay un problema interno con la clave de IA. Por favor, contacta con soporte técnico."

    except OpenAIError:
        reply_text = "He tenido un problema procesando tu mensaje. Inténtalo nuevamente."

    return PlainTextResponse(
        content=f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{reply_text}</Message>
</Response>""",
        media_type="application/xml"
    )
