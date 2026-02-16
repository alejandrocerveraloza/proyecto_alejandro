import os
import json
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from groq import Groq

# Cargar variables de entorno (.env)
load_dotenv()

app = Flask(__name__)

# Usamos Groq como motor principal para evitar errores de saldo en el video
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Este mapa define qué IA se mostrará en la web según la categoría detectada
ROUTING_MAP = {
    "red": "Grok-beta (x.ai)",
    "seguridad": "Grok-beta (x.ai)",
    "codigo": "Claude 3.5 Sonnet",
    "sql": "DeepSeek-Coder",
    "general": "GPT-4o"
}

@app.route('/')
def home():
    """Sirve la interfaz visual"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Ruta principal de procesamiento"""
    try:
        data = request.json
        prompt_usuario = data.get('prompt')

        if not prompt_usuario:
            return jsonify({"error": "No prompt provided"}), 400

        # RETO 1: Clasificación con Llama 3.1 8B
        clasificacion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Responde SOLO JSON: {'cat': 'red|seguridad|codigo|sql|general', 'p_opt': 'optimización del prompt'}"},
                {"role": "user", "content": prompt_usuario}
            ],
            response_format={"type": "json_object"}
        )
        
        resultado_json = json.loads(clasificacion.choices[0].message.content)
        categoria = resultado_json.get('cat', 'general')
        prompt_optimizado = resultado_json.get('p_opt', prompt_usuario)

        # RETO 2: Enrutamiento (Ejecutado por Groq para la demo, pero etiquetado por el mapa)
        respuesta_ia = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": f"Eres un experto en {categoria}. Responde de forma técnica y profesional."},
                {"role": "user", "content": prompt_optimizado}
            ]
        )

        # Retornamos la respuesta con la "IA Utilizada" según el mapa de enrutamiento
        return jsonify({
            "status": "success",
            "categoria_detectada": categoria,
            "ia_utilizada": ROUTING_MAP.get(categoria, "GPT-4o"),
            "respuesta": respuesta_ia.choices[0].message.content
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Ejecución en el puerto 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
