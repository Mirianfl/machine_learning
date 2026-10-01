# api_ml.py - El cerebro ML expuesto como API para Laravel/React
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
    df = pd.DataFrame([[nota_inicial, materia_id, leyo]], columns=COLUMNAS)
    return modelo.predict(df)[0]

def obtener_catalogo():
    conexion = pymysql.connect(host="127.0.0.1", user="root", password="",
                               database="preuniversitario_psicologia", charset="utf8mb4")
    cursor = conexion.cursor()
    cursor.execute("""
        SELECT l.id, l.nombre, l.materia_id, m.nombre
        FROM libros l JOIN materias m ON l.materia_id = m.id
        WHERE l.deleted_at IS NULL AND m.deleted_at IS NULL
    """)
    filas = cursor.fetchall()
    conexion.close()
    return filas

@app.route("/salud")
def salud():
    return jsonify({"estado": "ok", "mensaje": "El cerebro ML está vivo"})

@app.route("/recomendar/<int:materia_id>/<int:nota_inicial>")
def recomendar(materia_id, nota_inicial):
    mejora_si_lee = predecir(nota_inicial, materia_id, 1)
    mejora_si_no  = predecir(nota_inicial, materia_id, 0)
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
    return jsonify({"recomendar": False,
                    "mensaje": "Por ahora no se requiere recomendación"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)