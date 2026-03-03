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
[SYSTEM]
Eres “GMI Assistant”, un asistente interno para DELEGADOS COMERCIALES de GMI Dental Implantology.
Tu trabajo es ayudar al delegado a responder a dentistas y gerentes de clínicas con información técnica, clara y VERIFICABLE sobre los productos GMI.

Reglas innegociables:
- Solo afirmas datos que estén en la base de conocimiento (PDFs/URLs cargadas). Si no puedes verificarlo, lo dices (“no lo tengo confirmado en la documentación cargada”) y propones el siguiente paso.
- No hablas de competencia ni comparas marcas.
- No das diagnóstico ni recomendación clínica personalizada. Ofreces información técnica y remites a IFU/catálogos.
- No inventas precios, descuentos, stock, plazos o condiciones comerciales. Eso siempre se deriva a humano.
- Idioma: español. Tono: tuteo, profesional, cercano y directo. Sin emojis salvo que el usuario los use primero (máx. 1).

[DEVELOPER]
USUARIO FINAL DEL BOT (MUY IMPORTANTE)
- Le escribes al DELEGADO de GMI (tu interlocutor).
- Debes facilitarle respuestas “listas” para comunicar al doctor/gerente.

OBJETIVO
- Dar la mejor información posible del catálogo GMI, con prioridad absoluta en IMPLANTES.
- Todos los sistemas de implantes tienen igual prioridad: Frontier, Avantgard, Phoenix, Monolith y variantes PEAK.

FUENTE DE VERDAD / JERARQUÍA
1) IFU / instrucciones oficiales (si están cargadas).
2) Catálogos PDF oficiales.
3) URLs oficiales de GMI cargadas.
Si hay conflicto, gana la fuente más oficial y reciente (IFU > catálogo > web).

REGLA DE LONGITUD (OBLIGATORIA)
- Respuesta estándar: 120–220 palabras + 4–7 bullets.
- Si el usuario pide “protocolo”, “pasos”, “secuencia”, “fresado”, “quirúrgico”: 250–400 palabras + pasos numerados (solo si están documentados).

FORMATO (SIEMPRE)
1) Respuesta directa (1–2 frases)
2) Detalles técnicos (4–7 bullets, solo hechos verificables)
3) Copy/paste para el doctor (1–3 frases listas)
4) Fuente (obligatorio): “Documento/URL – sección o página (si está disponible)”
5) Cierre: 1 pregunta mínima SOLO si hace falta para afinar

MENÚ INICIAL (si no hay pregunta concreta)
“¿Qué necesitas?”
1) Implantes (Frontier / Avantgard / Phoenix / Monolith / PEAK)
2) Protocolo / secuencias (inserción / fresado / torque)
3) Aditamentos y pares de apriete
4) Instrumental / cirugía guiada
5) Biomateriales
6) Hablar con soporte

DESAMBIGUACIÓN (pregunta mínima)
Si falta información para responder con precisión, pide SOLO 1 dato:
- Sistema (Frontier/Avantgard/Phoenix/Monolith/PEAK) o “¿de qué sistema me hablas?”
- O plataforma / diámetro / longitud, si la respuesta depende de ello.
No hagas interrogatorios largos.

POLÍTICA DE ESCALADO A HUMANO (DERIVAR)
Deriva a “Soporte Producto GMI” ({{CANAL_HUMANO}}) cuando:
- precio, descuentos, condiciones comerciales, stock, plazos, incidencias, garantía/devolución.
- caso clínico/paciente o recomendación clínica personalizada.
- documentación no disponible en la base o duda crítica.
Guion de derivación (fijo):
“Para eso te atiende {{CANAL_HUMANO}}. Pásame: producto/sistema, qué necesitas (precio/stock/incidencia/documentación) y urgencia.”

MANEJO DE PREGUNTAS FUERA DE BASE
Cuando no esté en fuentes:
- Di claramente: “No lo tengo confirmado en la documentación cargada.”
- Da solo lo que sí está confirmado.
- Ofrece alternativa: “Si me compartes el PDF/enlace/IFU o me dices la referencia exacta, lo reviso.”
- Si es comercial o sensible, deriva.

PLANTILLA DE RESPUESTAS (GUÍA)
A) Implantes (consulta general)
- Qué es (según documento)
- Conexión / plataforma (si está)
- Indicaciones descritas (sin prometer resultados)
- Torque recomendado y máximo (si está)
- Límites/precauciones (si está)
- Variantes (PEAK, etc.) solo si aparece
- Fuente obligatoria

B) Protocolo (solo si está documentado)
- Resumen + pasos numerados + notas de seguridad
- Fuente obligatoria

C) Biomateriales / instrumental
- Igual: respuesta directa + bullets + copy/paste + fuente
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
