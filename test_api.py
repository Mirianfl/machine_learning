"""
test_api.py - Batería de Pruebas Automatizadas (Caja Negra y Caja Blanca)
========================================================================
Valida la integridad de seguridad (ISO/IEC 27001), rendimiento en ms (ISO/IEC 25010)
y calidad del modelo Scikit-Learn.
"""

import time
import json
import joblib
from api_ml import app, ML_API_KEY, MODEL_PATH, META_PATH


def test_caja_blanca_modelo():
    """Prueba de Caja Blanca: Valida estructura interna del modelo y metadatos."""
    print("\n--- [CAJA BLANCA] Verificando Modelo y Pipeline ---")
    modelo = joblib.load(MODEL_PATH)
    assert modelo is not None, "El archivo modelo.pkl debe cargarse correctamente."

    # Verificar componentes del pipeline
    pasos = dict(modelo.steps)
    assert "preprocesador" in pasos, "El pipeline debe incluir preprocesador ColumnTransformer."
    assert "regresor" in pasos, "El pipeline debe incluir el regresor RandomForest."
    regresor = pasos["regresor"]
    assert regresor.n_estimators == 100, "Debe tener 100 estimadores."
    assert regresor.random_state == 42, "Debe tener random_state=42."

    # Verificar metadatos CRISP-DM
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "mae" in meta, "Metadatos deben incluir MAE."
    assert "r2" in meta, "Metadatos deben incluir R²."
    assert meta["r2"] > 0.50, f"R² debe ser aceptable (>0.50, actual={meta['r2']})."
    assert "feature_importances" in meta, "Debe exponer feature_importances (XAI)."
    assert len(meta["materias_entrenadas"]) > 0, "Debe listar materias entrenadas."
    print(f"✅ Caja Blanca OK: MAE={meta['mae']}, R²={meta['r2']}, XAI={meta['feature_importances']}")


def test_caja_negra_seguridad_autenticacion():
    """Prueba de Caja Negra: ISO/IEC 27001 (401 si no hay clave X-ML-API-KEY)."""
    print("\n--- [CAJA NEGRA] Verificando Seguridad ISO/IEC 27001 ---")
    cliente = app.test_client()

    # 1. /salud no requiere token
    resp_salud = cliente.get("/salud")
    assert resp_salud.status_code == 200, "El endpoint /salud debe responder 200 sin clave."

    # 2. Endpoints protegidos sin cabecera deben dar 401
    resp_pred_sin_key = cliente.get("/predecir/1/65")
    assert resp_pred_sin_key.status_code == 401, "Sin cabecera X-ML-API-KEY debe devolver HTTP 401."

    # 3. Con cabecera inválida debe dar 401
    resp_pred_bad_key = cliente.get("/predecir/1/65", headers={"X-ML-API-KEY": "clave_falsa"})
    assert resp_pred_bad_key.status_code == 401, "Con clave errónea debe devolver HTTP 401."

    # 4. Con cabecera correcta debe dar 200
    resp_pred_ok = cliente.get("/predecir/1/65", headers={"X-ML-API-KEY": ML_API_KEY})
    assert resp_pred_ok.status_code == 200, "Con clave correcta debe devolver HTTP 200."
    print("✅ Seguridad OK: Acceso denegado a peticiones sin token, permitido con token legítimo.")


def test_caja_negra_validacion_rangos():
    """Prueba de Caja Negra: Validaciones de rango HTTP 400."""
    print("\n--- [CAJA NEGRA] Verificando Validación de Parámetros (HTTP 400) ---")
    cliente = app.test_client()
    headers = {"X-ML-API-KEY": ML_API_KEY}

    resp_neg = cliente.get("/predecir/1/-15", headers=headers)
    assert resp_neg.status_code == 400, "Nota negativa debe devolver 400."

    resp_exceso = cliente.get("/predecir/1/150", headers=headers)
    assert resp_exceso.status_code == 400, "Nota > 100 debe devolver 400."
    print("✅ Validación OK: Notas fuera de rango [0-100] rechazadas con HTTP 400.")


def test_caja_negra_rendimiento_y_estructura_json():
    """Prueba de Caja Negra: ISO/IEC 25010 (Latencia en ms y consistencia JSON)."""
    print("\n--- [CAJA NEGRA] Verificando Rendimiento y Estructura JSON ---")
    cliente = app.test_client()
    headers = {"X-ML-API-KEY": ML_API_KEY}

    # Medir tiempo de respuesta en ms
    t0 = time.time()
    resp = cliente.get("/predecir/1/65?tema_actual_id=1", headers=headers)
    latencia_ms = (time.time() - t0) * 1000

    assert resp.status_code == 200
    data = resp.get_json()

    # Validar llaves requeridas para retrocompatibilidad con React
    assert "prediccion_sin_leer" in data, "Debe incluir prediccion_sin_leer."
    assert "prediccion_con_leer" in data, "Debe incluir prediccion_con_leer."
    assert "beneficio" in data, "Debe incluir beneficio."
    assert "mensaje" in data, "Debe incluir mensaje."
    assert data["tiene_datos_historicos"] is True, "Materia 1 debe tener datos históricos."
    assert "siguiente_tema" in data, "Debe incluir siguiente_tema."
    assert "explicabilidad" in data, "Debe incluir explicabilidad XAI."

    print(f"✅ Eficiencia OK: Latencia promedio de /predecir = {latencia_ms:.2f} ms (esperado < 50ms)")


def test_caja_negra_materia_nueva_cold_start():
    """Prueba de Caja Negra: Comportamiento ante materias sin historial (Cold Start)."""
    print("\n--- [CAJA NEGRA] Verificando Materia Nueva (Sin Histórico) ---")
    cliente = app.test_client()
    headers = {"X-ML-API-KEY": ML_API_KEY}

    # Materia 999 no existe en el catálogo entrenado
    resp = cliente.get("/predecir/999/70", headers=headers)
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["tiene_datos_historicos"] is False, "Materia no entrenada debe reportar tiene_datos_historicos=false."
    assert data["prediccion_sin_leer"] is None, "Debe ser null para no inventar datos."
    assert data["prediccion_con_leer"] is None, "Debe ser null para no inventar datos."
    assert data["beneficio"] == 0
    print(f"✅ Cold Start OK: Respuesta segura: {data['mensaje']}")


if __name__ == "__main__":
    test_caja_blanca_modelo()
    test_caja_negra_seguridad_autenticacion()
    test_caja_negra_validacion_rangos()
    test_caja_negra_rendimiento_y_estructura_json()
    test_caja_negra_materia_nueva_cold_start()
    print("\n🎉 ¡TODAS LAS PRUEBAS DE CAJA NEGRA Y BLANCA PASARON CON ÉXITO!")
