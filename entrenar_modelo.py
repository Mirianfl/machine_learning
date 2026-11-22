"""
entrenar_modelo.py - Entrenamiento con Scikit-Learn y Exportación de Metadatos
=============================================================================
Entrena un RandomForestRegressor con Pipeline y OneHotEncoder a nivel de TEMA.
Exporta:
1. modelo.pkl: Pipeline completo serializado.
2. metadata_modelo.json: Métricas CRISP-DM, XAI (feature_importances) y catálogo entrenado.
"""

import os
import json
from datetime import datetime, timezone
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "datos_sinteticos.csv")
MODEL_PATH = os.path.join(BASE_DIR, "modelo.pkl")
META_PATH = os.path.join(BASE_DIR, "metadata_modelo.json")

def entrenar(origen_csv=CSV_PATH, origen_nombre="datos_sinteticos_v6_temas"):
    if not os.path.exists(origen_csv):
        raise FileNotFoundError(f"No se encontró el archivo de datos: {origen_csv}")

    datos = pd.read_csv(origen_csv)
    print(f"=== Entrenando con {len(datos)} registros de {origen_nombre} ===")

    # 1. Definición de características a nivel de TEMA
    columnas_features = ["materia_id", "tema_orden", "nota_previa", "leyo_material"]
    X = datos[columnas_features].copy()
    y = datos["mejora"].copy()

    # 2. División Train / Test (80% / 20%) con semilla fija para reproducibilidad
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 3. Pipeline con ColumnTransformer (OneHotEncoder para materia_id, passthrough para numéricas)
    preprocesador = ColumnTransformer(
        transformers=[
            ("materia_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["materia_id"])
        ],
        remainder="passthrough"
    )

    regresor = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline_ml = Pipeline([
        ("preprocesador", preprocesador),
        ("regresor", regresor)
    ])

    pipeline_ml.fit(X_train, y_train)

    # 4. Evaluación de Métricas de Calidad
    y_pred = pipeline_ml.predict(X_test)
    mae = float(round(mean_absolute_error(y_test, y_pred), 2))
    r2 = float(round(r2_score(y_test, y_pred), 4))

    print(f"📊 Métricas de Evaluación:")
    print(f"   - Error Medio Absoluto (MAE): {mae} puntos")
    print(f"   - Coeficiente de Determinación (R²): {r2}")

    # 5. Explicabilidad (XAI - Feature Importances)
    nombres_features = pipeline_ml.named_steps["preprocesador"].get_feature_names_out()
    importancias = pipeline_ml.named_steps["regresor"].feature_importances_

    importancias_agrupadas = {
        "peso_nota_previa": 0.0,
        "peso_repaso_tema": 0.0,
        "peso_orden_tema": 0.0,
        "peso_materia": 0.0,
    }

    for nombre, imp in zip(nombres_features, importancias):
        if "nota_previa" in nombre:
            importancias_agrupadas["peso_nota_previa"] += imp
        elif "leyo_material" in nombre:
            importancias_agrupadas["peso_repaso_tema"] += imp
        elif "tema_orden" in nombre:
            importancias_agrupadas["peso_orden_tema"] += imp
        elif "materia_ohe" in nombre:
            importancias_agrupadas["peso_materia"] += imp

    # Normalizar importancias a 2 decimales
    for k in importancias_agrupadas:
        importancias_agrupadas[k] = round(float(importancias_agrupadas[k]), 4)

    # 6. Materias y Temas entrenados
    materias_unicas = sorted([int(m) for m in datos["materia_id"].unique()])
    temas_unicos = sorted([int(o) for o in datos["tema_orden"].unique()])

    metadata = {
        "version": "2.0.0",
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(),
        "origen_modelo": origen_nombre,
        "algoritmo": "RandomForestRegressor(n_estimators=100, random_state=42)",
        "total_registros": int(len(datos)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "mae": mae,
        "r2": r2,
        "feature_importances": importancias_agrupadas,
        "materias_entrenadas": materias_unicas,
        "temas_entrenados_orden": temas_unicos,
        "columnas_features": columnas_features,
    }

    # 7. Guardar artefactos
    joblib.dump(pipeline_ml, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ Modelo guardado en: {MODEL_PATH}")
    print(f"✅ Metadatos guardados en: {META_PATH}")
    return pipeline_ml, metadata

if __name__ == "__main__":
    entrenar()
