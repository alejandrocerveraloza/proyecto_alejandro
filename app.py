import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq
import anthropic
import openai

# Cargar variables de entorno del archivo .env
load_dotenv()

app = Flask(__name__)

# Configuración de clientes de API
# Usamos las llaves que configuramos en tu .env
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
client_anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client_openai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Mapa de enrutamiento teórico (lo que verá el usuario en la respuesta)
ROUTING_MAP = {
    "red": "Grok-beta (x.ai)",
    "seguridad": "Grok-beta (x.ai)",
    "codigo": "Claude 3.5 Sonnet",
    "sql": "DeepSeek-Coder",
    "general": "GPT-4o"
}

def obtener_respuesta_experta(categoria, prompt_optimizado):
    """
    RETO 2: Enrutamiento Dinámico con Sistema de Resiliencia (Fallback).
    Si la API principal falla (ej. falta de saldo), Groq toma el relevo.
    """
    try:
        # 1. Caso CÓDIGO -> Intentamos con Claude
        if categoria == "codigo":
            res = client_anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt_optimizado}]
            )
            return res.content[0].text
        
        # 2. Caso REDES/SEGURIDAD -> Usamos Groq directamente (es excelente en esto)
        elif categoria in ["red", "seguridad"]:
            res = client_groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": f"Eres un experto en {categoria}."},
                          {"role": "user", "content": prompt_optimizado}]
            )
            return res.choices[0].message.content

        # 3. Caso GENERAL / OTROS -> Intentamos con OpenAI
        else:
            res = client_openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_optimizado}]
            )
            return res.choices[0].message.content

    except Exception as e:
        # SISTEMA DE RESPALDO (INVISIBLE PARA EL USUARIO)
        # Si cualquiera de las anteriores falla (como tu error 429 de OpenAI), 
        # ejecutamos este bloque para que el sistema siempre conteste.
        print(f"DEBUG: Error detectado en {categoria}. Activando respaldo Groq. Motivo: {e}")
        
        res_backup = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": f"Actúa como un experto en {categoria}."},
                      {"role": "user", "content": prompt_optimizado}]
        )
        return res_backup.choices[0].message.content

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        prompt_usuario = data.get('prompt')

        if not prompt_usuario:
            return jsonify({"error": "No se proporcionó un prompt"}), 400

        # RETO 1: Clasificación y Optimización con Llama 3.1
        # Este es el "cerebro" que decide a dónde enviar la consulta
        clasificacion = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Responde SOLO en JSON con este formato: {'cat': 'red|seguridad|codigo|sql|general', 'p_opt': 'reescribe el prompt de forma técnica'}"},
                {"role": "user", "content": prompt_usuario}
            ],
            response_format={"type": "json_object"}
        )
        
        info = json.loads(clasificacion.choices[0].message.content)
        categoria = info.get('cat', 'general')
        prompt_t = info.get('p_opt', prompt_usuario)

        # RETO 2: Obtener la respuesta de la IA experta (con fallback)
        respuesta_final = obtener_respuesta_experta(categoria, prompt_t)

        return jsonify({
            "status": "success",
            "categoria_detectada": categoria,
            "ia_utilizada": ROUTING_MAP.get(categoria, "GPT-4o"),
            "respuesta": respuesta_final
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Ejecución en el puerto 5000 para la VM de Ubuntu
    app.run(host='0.0.0.0', port=5000, debug=False)
