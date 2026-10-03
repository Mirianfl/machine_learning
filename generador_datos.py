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

# 2. Leer los TEMAS REALES ordenados por secuencia (orden)
cursor.execute("""
    SELECT t.id, t.nombre, t.materia_id, m.nombre as materia_nombre, t.paginas_libro, t.orden
    FROM temas t
    JOIN materias m ON t.materia_id = m.id
    WHERE t.deleted_at IS NULL AND m.deleted_at IS NULL
    ORDER BY t.materia_id, t.orden ASC
""")
temas = cursor.fetchall()

# 3. Leer los 20 estudiantes reales
cursor.execute("""
    SELECT e.id, u.nombre, u.apellido_paterno
    FROM estudiantes e JOIN users u ON e.user_id = u.id
    WHERE e.deleted_at IS NULL
""")
estudiantes = cursor.fetchall()
conexion.close()

print(f"=== TEMAS CARGADOS: {len(temas)} ===")
print(f"=== ESTUDIANTES: {len(estudiantes)} ===")

# 4. Simulación del historial académico
random.seed(42) # Reproducibilidad
filas = []

for est_id, est_nombre, est_apellido in estudiantes:
    nombre_completo = f"{est_nombre} {est_apellido}"
    
    # Variable para recordar cómo le fue en el tema anterior (correlación secuencial)
    # Empieza en 70 (un estudiante promedio comienza bien)
    rendimiento_previo = 70 

    for tema_id, tema_nombre, materia_id, materia_nombre, paginas_libro, orden in temas:
        
        # ¿El estudiante leyó el material de ESTE tema antes de que empezara? (50% probabilidad)
        leyo_tema = random.random() < 0.5
        
        # ── EL PATRÓN OCULTO QUE EL ML DEBE APRENDER ──
        # El bono por leer se suma al rendimiento base del estudiante
        if leyo_tema:
            bono_lectura = random.randint(20, 35)  # Leer ayuda MUCHO
        else:
            bono_lectura = random.randint(-5, 10)  # No leer casi no ayuda (o baja un poco)
        
        # El rendimiento en este tema depende de: cómo le fue antes + si leyó + un poco de azar
        rendimiento_tema = rendimiento_previo + bono_lectura + random.randint(-10, 10)
        rendimiento_tema = max(30, min(100, rendimiento_tema)) # Limitar entre 30 y 100
        
        # Guardamos este rendimiento para que influya en el SIGUIENTE tema
        rendimiento_previo = rendimiento_tema
        
        # ── GENERAR TAREAS VARIABLES PARA ESTE TEMA (entre 1 y 3 tareas) ──
        num_tareas = random.randint(1, 3)
        
        for num_tarea in range(1, num_tareas + 1):
            # Cada tarea varía un poco alrededor del rendimiento del tema
            nota_tarea = rendimiento_tema + random.randint(-8, 8)
            nota_tarea = max(0, min(100, round(nota_tarea, 1))) # Nota entre 0 y 100, con 1 decimal
            
            filas.append({
                "estudiante_id": est_id,
                "estudiante_nombre": nombre_completo,
                "materia_id": materia_id,
                "materia_nombre": materia_nombre,
                "tema_id": tema_id,
                "tema_nombre": tema_nombre,
                "orden_tema": orden,
                "paginas_libro": paginas_libro,
                "leyo_tema": int(leyo_tema),          # 1 = leyó, 0 = no leyó
                "numero_tarea_en_tema": num_tarea,
                "nota": nota_tarea,
            })

# 5. Guardar en CSV
datos = pd.DataFrame(filas)
datos.to_csv("datos_sinteticos.csv", index=False, encoding="utf-8-sig")

# 6. Resumen para verificar el patrón
print("\n=== RESUMEN DEL SIMULADOR v5 ===")
print(f"Total de tareas simuladas: {len(datos)}")
print("\nPromedio de notas por tema según si leyó el material o no:")
promedios = datos.groupby("leyo_tema")["nota"].mean().round(1)
print(promedios)
print("\n✅ CSV generado con tareas variables (1 a 3 por tema) y correlación secuencial.")