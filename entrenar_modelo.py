# entrenar_modelo.py - El cerebro aprende con scikit-learn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 🔑 AQUI CAMBIAS LA FUENTE cuando tengas datos reales (1 sola línea)
FUENTE = "datos_sinteticos.csv"   # luego: "datos_reales.csv"

datos = pd.read_csv(FUENTE)
print(f"=== Entrenando con: {FUENTE} ({len(datos)} registros) ===")

# 1. Características (X) y objetivo (y)
X = datos[["nota_inicial", "materia_id", "leyo_libro"]]
y = datos["mejora"]

# 2. Dividir: 80% para aprender, 20% para examinar al modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. El algoritmo de Machine Learning (Bosque Aleatorio)
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 4. EVALUACIÓN (esto es lo que el jurado querrá ver)
pred = modelo.predict(X_test)
print("\n=== EVALUACIÓN DEL MODELO ===")
print(f"Error medio (MAE): {mean_absolute_error(y_test, pred):.2f} puntos")
print(f"R² (qué tanto explica): {r2_score(y_test, pred):.2f}")

# 5. Guardar el cerebro entrenado
joblib.dump(modelo, "modelo.pkl")
print("\n✅ Modelo guardado en modelo.pkl")

# 6. Recomendación por "beneficio": cuánto gana si lee vs si no lee
COLUMNAS = ["nota_inicial", "materia_id", "leyo_libro"]

def decidir_recomendacion(materia_id, nota_inicial):
    df_lee = pd.DataFrame([[nota_inicial, materia_id, 1]], columns=COLUMNAS)
    df_no  = pd.DataFrame([[nota_inicial, materia_id, 0]], columns=COLUMNAS)
    si_lee = modelo.predict(df_lee)[0]
    si_no_lee = modelo.predict(df_no)[0]
    return si_lee, (si_lee - si_no_lee)

print("\n=== PRUEBA: estudiante con nota 45 en materia 1 ===")
esperada, beneficio = decidir_recomendacion(1, 45)
print(f"Si lee el resumen: mejora esperada +{esperada:.0f} puntos")
print(f"Beneficio de leer vs no leer: +{beneficio:.0f} puntos")
print("→ El sistema SÍ le recomendará leer" if beneficio > 10 else "→ No se recomienda")