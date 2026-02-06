import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq

# 1. Cargar variables de entorno del archivo .env
load_dotenv()

app = Flask(__name__)

# 2. Configurar cliente Groq con el modelo Llama 3.1 8B
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 3. Prompt de sistema (Cerebro del enrutador)
CLASSIFIER_PROMPT = """
Eres un experto en soporte tecnico de sistemas (ASIR). 
Tu tarea es clasificar la consulta del usuario y optimizarla.

CATEGORIAS: red, codigo, seguridad, sql, hardware, general.

REGLAS:
- Si mencionan VPN, PING, IP, WIFI, ROUTER o CONECTIVIDAD -> categoria: "red".
- Si mencionan PASSWORD, FIREWALL, VIRUS o ATAQUES -> categoria: "seguridad".
- Si mencionan SELECT, SQL o BASE DE DATOS -> categoria: "sql".

RESPONDE UNICAMENTE EN FORMATO JSON:
{{
  "categoria": "nombre_categoria",
  "prompt_optimizado": "Reescritura tecnica del problema"
}}

Consulta: "{user_prompt}"
"""

# 4. Mapa de enrutamiento a los modelos expertos
ROUTING_MAP = {
    "red": "grok-beta",
    "seguridad": "grok-beta",
    "codigo": "claude-3-5-sonnet-20241022",
    "sql": "deepseek-coder",
    "general": "gpt-4o-mini"
}

def clasificar_con_llama(prompt_usuario):
    """
    Usa Llama 3.1 8B para decidir a qué IA enviar la consulta.
    """
    try:
        completion = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",  # <--- Modelo corregido
            messages=[
                {"role": "system", "content": "Responde SOLO con JSON purista."},
                {"role": "user", "content": CLASSIFIER_PROMPT.format(user_prompt=prompt_usuario)}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        contenido = completion.choices[0].message.content.strip()
        # Imprime en la consola de la VM para que veas qué decide la IA
        print(f"DEBUG - Respuesta de Llama 3.1: {contenido}")
        
        return json.loads(contenido)
        
    except Exception as e:
        print(f"ERROR CRITICO EN API: {e}")
        # Si falla la API, devolvemos un valor por defecto seguro
        return {"categoria": "general", "prompt_optimizado": prompt_usuario}

@app.route('/ask', methods=['POST'])
def ask():
    """
    Endpoint para recibir preguntas.
    """
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Falta el campo prompt en el JSON"}), 400
        
    # Llamada a Llama 3.1 para clasificar
    analisis = clasificar_con_llama(data['prompt'])
    
    # Obtener la IA experta segun la categoria
    categoria = analisis.get("categoria", "general").lower()
    ia_destino = ROUTING_MAP.get(categoria, "gpt-4o-mini")
    
    return jsonify({
        "resultado": {
            "categoria": categoria,
            "ia_experta_asignada": ia_destino,
            "prompt_tecnico": analisis.get("prompt_optimizado")
        },
        "status": "success"
    })

if __name__ == '__main__':
    print("--- INICIANDO SERVIDOR EN UBUNTU (PUERTO 5000) ---")
    app.run(host='0.0.0.0', port=5000, debug=True)
