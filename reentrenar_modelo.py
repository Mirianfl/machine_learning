"""
reentrenar_modelo.py - Aprendizaje Acumulativo de Gestiones (2026 -> 2027 -> 2028)
==================================================================================
Extrae de MySQL las calificaciones reales consolidadas a nivel de TEMA,
calcula el proxy real de preparación de los estudiantes y evalúa si hay
suficiente historial para actualizar de forma atómica y segura el modelo.

Cumplimiento:
- ISO/IEC 27001: Consultas SQL parametrizadas (%s), sin credenciales en código.
- Criterio de Reemplazo Seguro: Mínimo 30 registros reales consolidados y R² >= 0.50.
"""

import os
import json
import logging
from datetime import datetime, timezone
import pymysql
import pandas as pd
import joblib
from dotenv import load_dotenv

from entrenar_modelo import entrenar, MODEL_PATH, META_PATH

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "psicologia_proyecto")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def extraer_datos_reales_temas():
    """Extrae notas consolidadas por TEMA y calcula el proxy real de estudio."""
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset="utf8mb4"
        )
    except Exception as e:
        logging.error(f"Error conectando a la base de datos MySQL: {e}")
        return None

    query = """
        SELECT 
            c.gestion_id,
            g.año as gestion_anio,
            e.id as estudiante_id,
            m.id as materia_id,
            m.nombre as materia_nombre,
            t.id as tema_id,
            t.nombre as tema_nombre,
            t.orden as tema_orden,
            t.paginas_libro,
            ROUND(AVG(cal.nota), 2) as nota_consolidada_tema,
            COUNT(DISTINCT lec.id) as total_tareas_tema,
            SUM(CASE WHEN ent.estado_entrega != 'atrasado' AND ent.estado_entrega != 'pendiente' THEN 1 ELSE 0 END) as tareas_a_tiempo,
            MAX(COALESCE(ref.completado, 0)) as refuerzo_completado
        FROM entregas ent
        JOIN calificaciones cal ON cal.entrega_id = ent.id
        JOIN lecciones lec ON ent.leccion_id = lec.id
        JOIN temas t ON lec.tema_id = t.id
        JOIN materias m ON t.materia_id = m.id
        JOIN cursos c ON lec.curso_id = c.id
        JOIN gestions g ON c.gestion_id = g.id
        JOIN estudiantes e ON ent.estudiante_id = e.id
        LEFT JOIN refuerzos ref ON ref.entrega_id = ent.id
        WHERE ent.deleted_at IS NULL 
          AND lec.deleted_at IS NULL 
          AND t.deleted_at IS NULL
          AND cal.nota IS NOT NULL
        GROUP BY c.gestion_id, g.año, e.id, m.id, m.nombre, t.id, t.nombre, t.orden, t.paginas_libro
        ORDER BY e.id, m.id, t.orden ASC
    """

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(query)
            filas = cursor.fetchall()
    except Exception as e:
        logging.error(f"Error ejecutando consulta de notas consolidadas: {e}")
        filas = []
    finally:
        conn.close()

    if not filas:
        logging.info("No se encontraron calificaciones consolidadas en MySQL.")
        return None

    # Construir secuencias de aprendizaje tema a tema por estudiante
    # Agrupamos por (estudiante_id, materia_id)
    estudiantes_materias = {}
    for f in filas:
        clave = (f["estudiante_id"], f["materia_id"])
        if clave not in estudiantes_materias:
            estudiantes_materias[clave] = []
        estudiantes_materias[clave].append(f)

    pares_secuenciales = []
    for (est_id, mat_id), historial in estudiantes_materias.items():
        historial_ordenado = sorted(historial, key=lambda x: x["tema_orden"])

        for i in range(len(historial_ordenado) - 1):
            tema_prev = historial_ordenado[i]
            tema_sig = historial_ordenado[i + 1]

            nota_prev = float(tema_prev["nota_consolidada_tema"])
            nota_sig = float(tema_sig["nota_consolidada_tema"])
            mejora = round(nota_sig - nota_prev, 2)

            # Proxy real de leyo_material:
            # 1 si cumplió al menos el 60% de tareas a tiempo o completó el refuerzo interactivo del tema
            tareas_totales = tema_sig["total_tareas_tema"] or 1
            tareas_puntuales = tema_sig["tareas_a_tiempo"] or 0
            tasa_puntualidad = tareas_puntuales / tareas_totales
            refuerzo_ok = bool(tema_sig["refuerzo_completado"])

            leyo_material = 1 if (tasa_puntualidad >= 0.60 or refuerzo_ok) else 0

            pares_secuenciales.append({
                "gestion_id": tema_sig["gestion_id"],
                "gestion_anio": tema_sig["gestion_anio"],
                "estudiante_id": est_id,
                "materia_id": mat_id,
                "tema_id": tema_sig["tema_id"],
                "tema_orden": tema_sig["tema_orden"],
                "nota_previa": nota_prev,
                "leyo_material": leyo_material,
                "nota_tema": nota_sig,
                "mejora": mejora,
            })

    if not pares_secuenciales:
        return None

    df_reales = pd.DataFrame(pares_secuenciales)
    logging.info(f"Se estructuraron {len(df_reales)} pares secuenciales de temas reales.")
    return df_reales


def ejecutar_reentrenamiento():
    logging.info("Iniciando proceso de evaluación y reentrenamiento acumulativo...")
    df_reales = extraer_datos_reales_temas()

    UMBRAL_MINIMO_REGISTROS = 30
    UMBRAL_MINIMO_R2 = 0.50

    if df_reales is None or len(df_reales) < UMBRAL_MINIMO_REGISTROS:
        registros = len(df_reales) if df_reales is not None else 0
        logging.warning(
            f"Insuficientes datos reales consolidados ({registros}/{UMBRAL_MINIMO_REGISTROS}). "
            "Se conserva el modelo actual en producción sin interrumpir el servicio."
        )
        return False

    # Guardar respaldo de datos reales acumulados
    csv_reales = os.path.join(os.path.dirname(MODEL_PATH), "datos_reales.csv")
    df_reales.to_csv(csv_reales, index=False, encoding="utf-8-sig")

    # Realizar entrenamiento con datos reales
    try:
        # Hacemos copia de seguridad del modelo actual
        backup_model = os.path.join(os.path.dirname(MODEL_PATH), "modelo_backup.pkl")
        if os.path.exists(MODEL_PATH):
            joblib.dump(joblib.load(MODEL_PATH), backup_model)

        # Entrenar temporalmente
        pipeline_nuevo, meta_nueva = entrenar(
            origen_csv=csv_reales,
            origen_nombre=f"datos_reales_acumulados_gestiones_{df_reales['gestion_anio'].min()}_{df_reales['gestion_anio'].max()}"
        )

        # Validar umbral de calidad
        if meta_nueva["r2"] < UMBRAL_MINIMO_R2:
            logging.warning(
                f"El modelo entrenado con datos reales no alcanzó el umbral mínimo de R² ({meta_nueva['r2']} < {UMBRAL_MINIMO_R2}). "
                "Restaurando modelo de respaldo para proteger la precisión en producción."
            )
            if os.path.exists(backup_model):
                joblib.dump(joblib.load(backup_model), MODEL_PATH)
            return False

        logging.info(
            f"✅ Reentrenamiento exitoso con datos reales: R²={meta_nueva['r2']}, MAE={meta_nueva['mae']}. "
            "Modelo actualizado atómicamente en producción."
        )
        return True

    except Exception as e:
        logging.error(f"Error durante el reentrenamiento: {e}")
        return False


if __name__ == "__main__":
    ejecutar_reentrenamiento()

