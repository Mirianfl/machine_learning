# generador_datos.py - v5: Simulador realista con temas secuenciales y tareas variables
# -----------------------------------------------------------------------------
# Este script es el "simulador de vuelo" de tu proyecto de grado:
# genera historiales de calificaciones sintéticos para entrenar al ML,
# usando tu catálogo REAL (materias + libros) y tus 20 estudiantes REALES.
# -----------------------------------------------------------------------------
import random       # Números aleatorios (para simular notas)
import pymysql      # Conector de Python a MySQL
import pandas as pd # Manejo de datos tipo tabla (como Excel)

# ═══════════════════════════════════════════════════════════════
# PASO 1: CONEXIÓN A LA BASE DE DATOS REAL DE LARAVEL
# ═══════════════════════════════════════════════════════════════
conexion = pymysql.connect(
    host="127.0.0.1",                          # Servidor MySQL local (XAMPP)
    user="root",                               # Usuario por defecto de XAMPP
    password="",                               # Contraseña vacía en XAMPP
    database="preuniversitario_psicologia",    # Tu base de datos
    charset="utf8mb4"                          # Para leer tildes y ñ correctamente
)
cursor = conexion.cursor()

# ═══════════════════════════════════════════════════════════════
# PASO 2: LEER EL CATÁLOGO REAL (libros + materias)
# El JOIN une libros con materias para saber qué libro corresponde
# a cada materia. El WHERE ignora registros borrados (deleted_at).
# ═══════════════════════════════════════════════════════════════
cursor.execute("""
    SELECT l.id, l.nombre, l.materia_id, m.nombre
    FROM libros l
    JOIN materias m ON l.materia_id = m.id
    WHERE l.deleted_at IS NULL AND m.deleted_at IS NULL
""")
catalogo = cursor.fetchall()   # Lista de tuplas: (libro_id, libro_nombre, materia_id, materia_nombre)

# Imprimimos el catálogo para VERIFICAR que leímos tu BD real
print("=== CATÁLOGO LEÍDO ===")
for c in catalogo:
    print(f"- Libro #{c[0]}: {c[1]}  (materia: {c[3]})")

# ═══════════════════════════════════════════════════════════════
# PASO 3: LEER LOS 20 ESTUDIANTES REALES DE TU BD
# El JOIN une estudiantes con users, porque el nombre y apellido
# están en la tabla users (estudiantes solo guarda el user_id).
# ═══════════════════════════════════════════════════════════════
cursor.execute("""
    SELECT e.id, u.nombre, u.apellido_paterno
    FROM estudiantes e
    JOIN users u ON e.user_id = u.id
    WHERE e.deleted_at IS NULL
""")
estudiantes = cursor.fetchall()   # Lista de tuplas: (estudiante_id, nombre, apellido)
conexion.close()                  # Cerramos la conexión: ya tenemos todo lo real

print(f"\n=== ESTUDIANTES DE TU BD: {len(estudiantes)} ===")

# ═══════════════════════════════════════════════════════════════
# PASO 4: GENERAR LOS HISTORIALES SIMULADOS (el "simulador")
# ═══════════════════════════════════════════════════════════════
random.seed(42)   # Semilla fija => mismos datos cada vez que corras (reproducibilidad)
filas = []        # Aquí acumularemos cada fila del historial
TAREAS_POR_COMBINACION = 15   # 15 tareas por cada (estudiante + materia)
                              # Total: 20 estudiantes × 5 materias × 15 = 1500 filas

# Bucle 1: recorre cada libro/materia REAL de tu catálogo
for libro_id, libro_nombre, materia_id, materia_nombre in catalogo:

    # Bucle 2: recorre cada estudiante REAL de tu BD
    for est_id, est_nombre, est_apellido in estudiantes:

        # Bucle 3: simula 15 tareas para esta combinación
        for i in range(TAREAS_POR_COMBINACION):

            # Nota inicial reprobatoria (entre 30 y 50)
            nota_inicial = random.randint(30, 50)

            # ¿El estudiante leyó el resumen después de reprobar? (50% de probabilidad)
            leyo = random.random() < 0.5

            # ── EL PATRÓN OCULTO que el ML deberá descubrir solo ──
            if leyo:
                mejora = random.randint(25, 40)   # Leer el resumen ayuda MUCHO
            else:
                mejora = random.randint(0, 10)    # No leer casi no ayuda

            # La nota final no puede pasar de 100
            nota_final = min(nota_inicial + mejora, 100)

            # Armamos la fila con datos REALES (ids y nombres) + notas SIMULADAS
            filas.append({
                "estudiante_id": est_id,                              # ID real de tu BD
                "estudiante_nombre": f"{est_nombre} {est_apellido}",  # Nombre real (ficticio pero de tu sistema)
                "materia_id": materia_id,                             # ID real
                "materia_nombre": materia_nombre,                     # Nombre real
                "libro_id": libro_id,                                 # ID real
                "libro_nombre": libro_nombre,                         # Nombre real
                "leyo_libro": int(leyo),                              # 1 = leyó, 0 = no leyó
                "nota_inicial": nota_inicial,                         # Simulada
                "nota_final": nota_final,                             # Simulada
                "mejora": nota_final - nota_inicial,                  # Cuánto mejoró
            })

# ═══════════════════════════════════════════════════════════════
# PASO 5: CONVERTIR A TABLA Y GUARDAR EN CSV
# index=False  → no guarda el número de fila de pandas
# utf-8-sig    → conserva tildes/ñ y abre bien en Excel
# ═══════════════════════════════════════════════════════════════
datos = pd.DataFrame(filas)
datos.to_csv("datos_sinteticos.csv", index=False, encoding="utf-8-sig")

# ═══════════════════════════════════════════════════════════════
# PASO 6: RESUMEN FINAL (comprobamos que el patrón oculto existe)
# groupby agrupa por "leyó" (1) y "no leyó" (0) y promedia la mejora
# ═══════════════════════════════════════════════════════════════
print("\n=== RESUMEN DEL SIMULADOR ===")
print(f"Total de historiales simulados: {len(datos)}")
print("\nMejora promedio según si leyó o no:")
print(datos.groupby("leyo_libro")["mejora"].mean().round(1))
print("\n✅ Datos sintéticos guardados en datos_sinteticos.csv")