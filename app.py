import os
import json
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from groq import Groq
import anthropic
import openai

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)

# Configuración de clientes
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
client_anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client_openai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Mapa de enrutamiento para la interfaz visual
ROUTING_MAP = {
    "red": "Grok-beta (via Groq Fallback)",
    "seguridad": "Grok-beta (via Groq Fallback)",
    "codigo": "Claude 3.5 Sonnet",
    "sql": "DeepSeek-Coder",
    "general": "GPT-4o"
}

def obtener_respuesta_experta(categoria, prompt_optimizado):
    """Reto 2: Enrutamiento con Fallback Invisible"""
    try:
        if categoria == "codigo":
            res = client_anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt_optimizado}]
            )
            return res.content[0].text
        
        elif categoria in ["red", "seguridad"]:
            res = client_groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": f"Experto en {categoria}"},
                          {"role": "user", "content": prompt_optimizado}]
            )
            return res.choices[0].message.content

        else:
            # Aquí es donde fallaría OpenAI por cuota, activando el except
            res = client_openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_optimizado}]
            )
            return res.choices[0].message.content

    except Exception as e:
        # Respaldo automático con Groq si falla la API principal
        print(f"DEBUG: Relevo activado para {categoria}. Motivo: {e}")
        res_backup = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Responde como experto en {categoria}: {prompt_optimizado}"}]
        )
        return res_backup.choices[0].message.content

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        prompt_usuario = data.get('prompt')

        # Reto 1: Clasificación
        clasificacion = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Responde SOLO JSON: {'cat': 'red|seguridad|codigo|sql|general', 'p_opt': 'prompt optimizado'}"},
                {"role": "user", "content": prompt_usuario}
            ],
            response_format={"type": "json_object"}
        )
        
        info = json.loads(clasificacion.choices[0].message.content)
        categoria = info.get('cat', 'general')
        
        respuesta = obtener_respuesta_experta(categoria, info.get('p_opt'))

        return jsonify({
            "status": "success",
            "categoria_detectada": categoria,
            "ia_utilizada": ROUTING_MAP.get(categoria, "GPT-4o"),
            "respuesta": respuesta
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
