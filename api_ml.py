"""
api_ml.py - Microservicio de Machine Learning en Producción (v2.0)
===================================================================
Expone la API predictiva y prescriptiva para el sistema educativo UPEA Psicología.
Cumplimiento normativo y de calidad:
- ISO/IEC 27001: Autenticación interna vía X-ML-API-KEY, consultas SQL parametrizadas (%s).
- ISO/IEC 25010: Eficiencia con catálogo en caché en memoria (TTL 30 min), logs rotativos.
- Enfoque metodológico: Predicción basada en TEMA y secuencia curricular con XAI.
"""

import os
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
import pymysql
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# 1. CONFIGURACIÓN Y SEGURIDAD
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo.pkl")
META_PATH = os.path.join(BASE_DIR, "metadata_modelo.json")
LOG_PATH = os.path.join(BASE_DIR, "ml_service.log")

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "psicologia_proyecto")

ML_HOST = os.getenv("ML_HOST", "127.0.0.1")
ML_PORT = int(os.getenv("ML_PORT", "5000"))
ML_API_KEY = os.getenv("ML_API_KEY", "ml_secret_token_2026_psi_upea")

# Configurar logging rotativo seguro (máx 5 MB, hasta 3 copias)
logger = logging.getLogger("ml_service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    rfh = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    rfh.setFormatter(formatter)
    logger.addHandler(rfh)
    # Consola
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

app = Flask(__name__)

# =====================================================================
# 2. CARGA DEL MODELO Y METADATOS
# =====================================================================
modelo_pipeline = None
metadata_modelo = {}


def cargar_modelo():
    global modelo_pipeline, metadata_modelo
    try:
        if os.path.exists(MODEL_PATH):
            modelo_pipeline = joblib.load(MODEL_PATH)
            logger.info(f"✅ Cerebro ML cargado exitosamente desde {MODEL_PATH}")
        else:
            logger.warning(f"⚠️ No se encontró {MODEL_PATH}. Ejecuta entrenar_modelo.py")

        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                metadata_modelo = json.load(f)
            logger.info("✅ Metadatos del modelo cargados exitosamente.")
    except Exception as e:
        logger.error(f"❌ Error al cargar modelo o metadatos: {e}")


cargar_modelo()

# =====================================================================
# 3. CACHÉ EN MEMORIA DEL CATÁLOGO DE TEMAS (TTL 30 MINUTOS)
# =====================================================================
cache_catalogo = {
    "ultimo_refresco": 0,
    "temas": [],  # lista de dicts
    "materias": {},  # materia_id -> nombre
}
TTL_CATALOGO_SEGUNDOS = 1800  # 30 minutos


def obtener_conexion_db():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=3
    )


def refrescar_catalogo_si_es_necesario(forzar=False):
    ahora = time.time()
    if not forzar and (ahora - cache_catalogo["ultimo_refresco"] < TTL_CATALOGO_SEGUNDOS) and cache_catalogo["temas"]:
        return

    try:
        with obtener_conexion_db() as conn:
            with conn.cursor() as cursor:
                # Consultar materias y temas con joins parametrizados y seguros
                cursor.execute("""
                    SELECT t.id, t.nombre, t.materia_id, m.nombre as materia_nombre, 
                           t.orden, t.paginas_libro, t.estado
                    FROM temas t
                    JOIN materias m ON t.materia_id = m.id
                    WHERE t.deleted_at IS NULL AND m.deleted_at IS NULL
                    ORDER BY t.materia_id, t.orden ASC
                """)
                temas_db = cursor.fetchall()

        if temas_db:
            cache_catalogo["temas"] = temas_db
            cache_catalogo["materias"] = {t["materia_id"]: t["materia_nombre"] for t in temas_db}
            cache_catalogo["ultimo_refresco"] = ahora
            logger.info(f"✅ Catálogo en memoria actualizado: {len(temas_db)} temas disponibles.")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo refrescar catálogo desde DB ({e}). Usando copia en memoria si existe.")
        cache_catalogo["ultimo_refresco"] = ahora  # Evitar reintentar conexión en cada petición
        if not cache_catalogo["temas"]:
            # Catálogo de respaldo para asegurar continuidad de servicio
            cache_catalogo["temas"] = [
                {"id": 1, "nombre": "Tema 1: Qué es la Psicología", "materia_id": 1, "materia_nombre": "INTRODUCCIÓN A LA PSICOLOGÍA", "orden": 1, "paginas_libro": "págs. 5-8", "estado": "visto"},
                {"id": 2, "nombre": "Tema 2: Historia y Evolución", "materia_id": 1, "materia_nombre": "INTRODUCCIÓN A LA PSICOLOGÍA", "orden": 2, "paginas_libro": "págs. 9-20", "estado": "programado"},
                {"id": 3, "nombre": "Tema 3: Procesos Psicológicos Básicos", "materia_id": 1, "materia_nombre": "INTRODUCCIÓN A LA PSICOLOGÍA", "orden": 3, "paginas_libro": "págs. 21-30", "estado": "programado"},
                {"id": 4, "nombre": "Tema 4: Motivación y Emoción", "materia_id": 1, "materia_nombre": "INTRODUCCIÓN A LA PSICOLOGÍA", "orden": 4, "paginas_libro": "págs. 31-33", "estado": "programado"},
                {"id": 5, "nombre": "Tema 1: Definición de Salud", "materia_id": 2, "materia_nombre": "REALIDAD NACIONAL Y SALUD MENTAL", "orden": 1, "paginas_libro": "págs. 43-44", "estado": "visto"},
                {"id": 6, "nombre": "Tema 2: Salud Mental en Bolivia", "materia_id": 2, "materia_nombre": "REALIDAD NACIONAL Y SALUD MENTAL", "orden": 2, "paginas_libro": "págs. 45-47", "estado": "programado"},
            ]
            cache_catalogo["materias"] = {1: "INTRODUCCIÓN A LA PSICOLOGÍA", 2: "REALIDAD NACIONAL Y SALUD MENTAL"}


# Inicializar catálogo al arrancar
refrescar_catalogo_si_es_necesario()

# =====================================================================
# 4. MIDDLEWARE DE SEGURIDAD INTERNA (X-ML-API-KEY)
# =====================================================================
@app.before_request
def verificar_autenticacion():
    # El endpoint /salud es público para healthchecks de monitoreo
    if request.path == "/salud" or request.method == "OPTIONS":
        return

    # Si ML_API_KEY está configurada, validar la cabecera
    if ML_API_KEY:
        clave_recibida = request.headers.get("X-ML-API-KEY")
        if not clave_recibida or clave_recibida != ML_API_KEY:
            logger.warning(f"🔒 Intento de acceso no autorizado desde {request.remote_addr} a {request.path}")
            return jsonify({
                "error": "No autorizado",
                "mensaje": "Se requiere una clave válida en la cabecera X-ML-API-KEY."
            }), 401


# =====================================================================
# 5. ENDPOINTS DE LA API
# =====================================================================

@app.route("/salud", methods=["GET"])
def salud():
    """Healthcheck y estado del modelo."""
    return jsonify({
        "estado": "ok",
        "servicio": "Microservicio Machine Learning UPEA Psicología",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modelo_cargado": modelo_pipeline is not None,
        "metadata": {
            "version": metadata_modelo.get("version", "2.0.0"),
            "origen_modelo": metadata_modelo.get("origen_modelo", "desconocido"),
            "mae": metadata_modelo.get("mae"),
            "r2": metadata_modelo.get("r2"),
            "total_registros": metadata_modelo.get("total_registros"),
            "fecha_entrenamiento": metadata_modelo.get("fecha_entrenamiento"),
            "materias_entrenadas": metadata_modelo.get("materias_entrenadas", []),
        }
    })


@app.route("/catalogo/recargar", methods=["POST"])
def recargar_catalogo():
    """Fuerza la recarga del catálogo de temas desde la base de datos."""
    refrescar_catalogo_si_es_necesario(forzar=True)
    return jsonify({
        "mensaje": "Catálogo recargado exitosamente en memoria.",
        "total_temas": len(cache_catalogo["temas"]),
        "total_materias": len(cache_catalogo["materias"])
    })


def procesar_prediccion(materia_id, nota_actual, tema_actual_id=None):
    """Lógica central de predicción a nivel de TEMA con manejo de arranque en frío."""
    if nota_actual < 0 or nota_actual > 100:
        return jsonify({
            "error": "Nota fuera de rango",
            "mensaje": f"La nota debe estar comprendida entre 0 y 100 (recibido: {nota_actual})."
        }), 400

    refrescar_catalogo_si_es_necesario()

    # 1. Validar si la materia tiene datos históricos entrenados
    materias_entrenadas = metadata_modelo.get("materias_entrenadas", [])
    if materia_id not in materias_entrenadas:
        logger.info(f"ℹ️ Materia {materia_id} sin datos históricos suficientes (Cold Start).")
        return jsonify({
            "tiene_datos_historicos": False,
            "prediccion_sin_leer": None,
            "prediccion_con_leer": None,
            "beneficio": 0,
            "mensaje": "Esta materia o tema es nuevo en la gestión actual. No cuenta con datos históricos suficientes para una predicción confiable.",
            "siguiente_tema": None,
            "explicabilidad": None,
            "origen_modelo": "sin_datos_historicos"
        })

    # 2. Localizar el siguiente tema en la secuencia de la materia
    temas_materia = [t for t in cache_catalogo["temas"] if t["materia_id"] == materia_id]
    temas_materia = sorted(temas_materia, key=lambda x: x["orden"])

    siguiente_tema_obj = None
    orden_siguiente = 2

    if tema_actual_id:
        tema_actual_match = next((t for t in temas_materia if t["id"] == tema_actual_id), None)
        if tema_actual_match:
            orden_actual = tema_actual_match["orden"]
            siguiente_tema_obj = next((t for t in temas_materia if t["orden"] > orden_actual and t["estado"] != "omitido"), None)

    if not siguiente_tema_obj and temas_materia:
        # Por defecto, seleccionar el primer tema no omitido o el segundo
        siguiente_tema_obj = temas_materia[1] if len(temas_materia) > 1 else temas_materia[0]

    if siguiente_tema_obj:
        orden_siguiente = siguiente_tema_obj["orden"]

    # 3. Predicción con Scikit-Learn
    if modelo_pipeline is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    # Construir dataframes con las 4 características del pipeline
    cols = ["materia_id", "tema_orden", "nota_previa", "leyo_material"]
    df_con = pd.DataFrame([[materia_id, orden_siguiente, nota_actual, 1]], columns=cols)
    df_sin = pd.DataFrame([[materia_id, orden_siguiente, nota_actual, 0]], columns=cols)

    mejora_con = float(modelo_pipeline.predict(df_con)[0])
    mejora_sin = float(modelo_pipeline.predict(df_sin)[0])

    prediccion_con = round(max(0.0, min(100.0, nota_actual + mejora_con)), 1)
    prediccion_sin = round(max(0.0, min(100.0, nota_actual + mejora_sin)), 1)
    beneficio = round(prediccion_con - prediccion_sin, 1)

    nombre_sig = siguiente_tema_obj["nombre"] if siguiente_tema_obj else "el siguiente tema"
    pags_sig = siguiente_tema_obj["paginas_libro"] if siguiente_tema_obj else "el material asignado"

    return jsonify({
        "tiene_datos_historicos": True,
        "prediccion_sin_leer": prediccion_sin,
        "prediccion_con_leer": prediccion_con,
        "beneficio": beneficio,
        "mensaje": f"Si no te preparas: ~{prediccion_sin}/100 | Si estudias {pags_sig}: ~{prediccion_con}/100",
        "siguiente_tema": {
            "tema_id": siguiente_tema_obj["id"] if siguiente_tema_obj else None,
            "nombre": nombre_sig,
            "orden": orden_siguiente,
            "paginas_libro": pags_sig
        } if siguiente_tema_obj else None,
        "explicabilidad": metadata_modelo.get("feature_importances", {}),
        "origen_modelo": metadata_modelo.get("origen_modelo", "datos_sinteticos")
    })


@app.route("/predecir/<int:materia_id>/<int(signed=True):nota_actual>", methods=["GET"])
@app.route("/predecir/<int:materia_id>/<float(signed=True):nota_actual>", methods=["GET"])
def predecir_rendimiento(materia_id, nota_actual):
    """Endpoint Preventivo: Predice desempeño en el próximo tema con/sin preparación."""
    tema_actual_id = request.args.get("tema_actual_id", default=None, type=int)
    return procesar_prediccion(materia_id, float(nota_actual), tema_actual_id)


@app.route("/recomendar/<int:materia_id>/<int(signed=True):nota_inicial>", methods=["GET"])
@app.route("/recomendar/<int:materia_id>/<float(signed=True):nota_inicial>", methods=["GET"])
def recomendar(materia_id, nota_inicial):
    """Endpoint Correctivo: Recomienda el repaso del TEMA relevante de la materia."""
    nota_float = float(nota_inicial)
    if nota_float < 0 or nota_float > 100:
        return jsonify({
            "error": "Nota fuera de rango",
            "mensaje": "La nota debe estar entre 0 y 100."
        }), 400

    refrescar_catalogo_si_es_necesario()

    materias_entrenadas = metadata_modelo.get("materias_entrenadas", [])
    if materia_id not in materias_entrenadas:
        return jsonify({
            "recomendar": False,
            "tiene_datos_historicos": False,
            "mensaje": "Materia en recopilación de datos iniciales. No requiere recomendación por ahora."
        })

    temas_materia = [t for t in cache_catalogo["temas"] if t["materia_id"] == materia_id]
    tema_prioritario = temas_materia[0] if temas_materia else None

    cols = ["materia_id", "tema_orden", "nota_previa", "leyo_material"]
    orden = tema_prioritario["orden"] if tema_prioritario else 1
    df_con = pd.DataFrame([[materia_id, orden, nota_float, 1]], columns=cols)
    df_sin = pd.DataFrame([[materia_id, orden, nota_float, 0]], columns=cols)

    mejora_con = float(modelo_pipeline.predict(df_con)[0])
    mejora_sin = float(modelo_pipeline.predict(df_sin)[0])
    beneficio = round(mejora_con - mejora_sin, 1)

    if beneficio > 8.0 and tema_prioritario:
        return jsonify({
            "recomendar": True,
            "tiene_datos_historicos": True,
            "tema_id": tema_prioritario["id"],
            "tema_nombre": tema_prioritario["nombre"],
            "materia_nombre": tema_prioritario["materia_nombre"],
            "paginas_libro": tema_prioritario["paginas_libro"],
            "mejora_esperada_si_lee": round(mejora_con, 1),
            "beneficio_de_leer": beneficio,
            "mensaje": f"Te recomendamos repasar: {tema_prioritario['nombre']} ({tema_prioritario['paginas_libro']})",
            "explicabilidad": metadata_modelo.get("feature_importances", {})
        })

    return jsonify({
        "recomendar": False,
        "tiene_datos_historicos": True,
        "mensaje": "Tu rendimiento actual se mantiene en un nivel adecuado."
    })


@app.route("/riesgo/estudiante/<int:estudiante_id>/<int:materia_id>", methods=["GET"])
def evaluar_riesgo_estudiante(estudiante_id, materia_id):
    """
    Endpoint de Alerta Temprana (Docentes):
    Calcula el semáforo de riesgo (Verde, Amarillo, Rojo) usando calificaciones y entregas.
    ISO/IEC 27001: Consulta parametrizada (%s).
    """
    try:
        with obtener_conexion_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        ROUND(AVG(cal.nota), 2) as promedio_notas,
                        COUNT(ent.id) as total_entregas,
                        SUM(CASE WHEN ent.estado_entrega = 'atrasado' THEN 1 ELSE 0 END) as entregas_atrasadas,
                        SUM(CASE WHEN ent.estado_entrega = 'pendiente' THEN 1 ELSE 0 END) as entregas_pendientes
                    FROM entregas ent
                    JOIN calificaciones cal ON cal.entrega_id = ent.id
                    JOIN lecciones lec ON ent.leccion_id = lec.id
                    WHERE ent.estudiante_id = %s 
                      AND lec.materia_id = %s
                      AND ent.deleted_at IS NULL
                """, (estudiante_id, materia_id))
                res = cursor.fetchone()
    except Exception as e:
        logger.warning(f"Error consultando riesgo de estudiante en DB: {e}")
        res = None

    if not res or res["promedio_notas"] is None:
        return jsonify({
            "estudiante_id": estudiante_id,
            "materia_id": materia_id,
            "nivel_riesgo": "sin_datos",
            "semaforo": "gris",
            "promedio": 0,
            "porcentaje_atrasos": 0,
            "mensaje": "El estudiante aún no registra calificaciones en esta materia."
        })

    promedio = float(res["promedio_notas"])
    total = int(res["total_entregas"] or 1)
    atrasos = int(res["entregas_atrasadas"] or 0)
    pct_atrasos = round((atrasos / total) * 100, 1)

    # Lógica del semáforo
    if promedio < 51.0 or pct_atrasos >= 40.0:
        nivel = "rojo"
        mensaje = "Riesgo crítico de reprobación. Requiere tutoría inmediata y refuerzo pedagógico."
    elif promedio < 66.0 or pct_atrasos >= 20.0:
        nivel = "amarillo"
        mensaje = "Riesgo moderado. Se sugiere monitorear la puntualidad de entregas y lectura de resúmenes."
    else:
        nivel = "verde"
        mensaje = "Rendimiento óptimo. El estudiante mantiene buen ritmo académico."

    return jsonify({
        "estudiante_id": estudiante_id,
        "materia_id": materia_id,
        "nivel_riesgo": nivel,
        "semaforo": nivel,
        "promedio": promedio,
        "total_entregas": total,
        "entregas_atrasadas": atrasos,
        "porcentaje_atrasos": pct_atrasos,
        "mensaje": mensaje
    })


if __name__ == "__main__":
    logger.info(f"Iniciando Microservicio ML en http://{ML_HOST}:{ML_PORT} (Producción)")
    app.run(host=ML_HOST, port=ML_PORT, debug=False)
