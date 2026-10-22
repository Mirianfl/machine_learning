# api_ml.py - El cerebro ML expuesto como API para Laravel/React
# ============================================================
# Este archivo expone DOS endpoints:
# 1. /recomendar/<materia_id>/<nota_inicial> → Sistema Correctivo (ya existente)
# 2. /predecir/<materia_id>/<nota_actual>    → Sistema Preventivo (NUEVO)
# ============================================================

import joblib
import pymysql
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permite que Laravel (puerto 8000) y React llamen a esta API (puerto 5000)

# Cargamos el cerebro entrenado UNA sola vez al iniciar
modelo = joblib.load("modelo.pkl")

# Columnas con las que el modelo fue entrenado (evita la advertencia amarilla)
COLUMNAS = ["nota_inicial", "materia_id", "leyo_libro"]


def predecir(nota_inicial, materia_id, leyo):
    """
    Usa el modelo entrenado para predecir la mejora.
    - leyo=1: el estudiante leyó el resumen
    - leyo=0: el estudiante NO leyó el resumen
    """
    df = pd.DataFrame([[nota_inicial, materia_id, leyo]], columns=COLUMNAS)
    return modelo.predict(df)[0]


def obtener_catalogo():
    """Lee el catálogo real de libros y materias desde MySQL."""
    conexion = pymysql.connect(
        host="127.0.0.1", user="root", password="",
        database="preuniversitario_psicologia", charset="utf8mb4"
    )
    cursor = conexion.cursor()
    cursor.execute("""
        SELECT l.id, l.nombre, l.materia_id, m.nombre
        FROM libros l JOIN materias m ON l.materia_id = m.id
        WHERE l.deleted_at IS NULL AND m.deleted_at IS NULL
    """)
    filas = cursor.fetchall()
    conexion.close()
    return filas


# ═══════════════════════════════════════════════════════════
# ENDPOINT 1: SISTEMA CORRECTIVO (ya existente)
# Se usa cuando el estudiante ya tiene una nota baja.
# ═══════════════════════════════════════════════════════════
@app.route("/salud")
def salud():
    """Verifica que la API esté viva."""
    return jsonify({"estado": "ok", "mensaje": "El cerebro ML está vivo"})


@app.route("/recomendar/<int:materia_id>/<int:nota_inicial>")
def recomendar(materia_id, nota_inicial):
    """
    Sistema Correctivo: recomienda leer un libro-resumen
    cuando el estudiante ya tiene una nota baja.
    """
    mejora_si_lee = predecir(nota_inicial, materia_id, 1)
    mejora_si_no = predecir(nota_inicial, materia_id, 0)
    beneficio = mejora_si_lee - mejora_si_no

    libro = next((f for f in obtener_catalogo() if f[2] == materia_id), None)

    if beneficio > 10 and libro:
        return jsonify({
            "recomendar": True,
            "libro_id": libro[0],
            "libro_nombre": libro[1],
            "materia_nombre": libro[3],
            "mejora_esperada_si_lee": round(mejora_si_lee, 1),
            "beneficio_de_leer": round(beneficio, 1),
            "mensaje": f"Te recomendamos leer: {libro[1]}"
        })
    return jsonify({
        "recomendar": False,
        "mensaje": "Por ahora no se requiere recomendación"
    })


# ═══════════════════════════════════════════════════════════
# ENDPOINT 2: SISTEMA PREVENTIVO (NUEVO - para tu tesis)
# Se usa para predecir el rendimiento en el PRÓXIMO tema.
# ═══════════════════════════════════════════════════════════
@app.route("/predecir/<int:materia_id>/<int:nota_actual>")
def predecir_rendimiento(materia_id, nota_actual):
    """
    Sistema Preventivo/Predictivo:
    Predice la nota del estudiante en el PRÓXIMO tema de la misma materia,
    con y sin preparación (lectura del resumen).
    
    Parámetros:
    - materia_id: ID de la materia (para que el modelo use el contexto correcto)
    - nota_actual: Nota que el estudiante sacó en el tema anterior
    
    Retorna:
    - prediccion_sin_leer: Nota predicha si NO se prepara
    - prediccion_con_leer: Nota predicha si SÍ lee el resumen
    - beneficio: Diferencia entre ambas predicciones
    """
    try:
        # 1. El modelo predice cuánta MEJORA obtendría el estudiante
        #    (esto es lo que aprendió durante el entrenamiento)
        mejora_si_lee = predecir(nota_actual, materia_id, 1)
        mejora_si_no = predecir(nota_actual, materia_id, 0)
        
        
        
        # 3. Proyección de notas para el PRÓXIMO tema:
        #    La nota predicha = nota_actual + mejora_predicha
        #    
        #    Si NO se prepara: la nota sube poco (solo ~5 puntos)
        #    porque el tema nuevo es más difícil y no leyó.
        prediccion_sin_leer = nota_actual + mejora_si_no
        
        #    Si SÍ lee el resumen: la nota sube mucho (~32 puntos)
        #    porque el modelo aprendió que leer ayuda.
        prediccion_con_leer = nota_actual + mejora_si_lee

        
        
        # Limitamos entre 0 y 100 (una nota no puede ser negativa ni mayor a 100)
        prediccion_sin_leer = max(0, min(100, prediccion_sin_leer))
        prediccion_con_leer = max(0, min(100, prediccion_con_leer))

        # 4. AHORA SÍ, CALCULAMOS EL BENEFICIO REAL CON LAS NOTAS YA TOPADAS
        beneficio = prediccion_con_leer - prediccion_sin_leer
        
        return jsonify({
            "prediccion_sin_leer": round(prediccion_sin_leer, 1),
            "prediccion_con_leer": round(prediccion_con_leer, 1),
            "beneficio": round(beneficio, 1),
            "mensaje": (
                f"Si no te preparas: ~{round(prediccion_sin_leer, 1)}/100 | "
                f"Si lees el resumen: ~{round(prediccion_con_leer, 1)}/100"
            )
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)