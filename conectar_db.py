# conectar_db.py - Python lee la base de datos REAL de Laravel
import pymysql

# 1. Abrimos la conexión a TU base de datos
conexion = pymysql.connect(
    host="127.0.0.1",          # Tu computadora (localhost)
    user="root",               # Usuario de MySQL (XAMPP por defecto)
    password="",               # Contraseña (vacía en XAMPP)
    database="preuniversitario_psicologia",
    charset="utf8mb4"          # Para que lea tildes y ñ correctamente
)

cursor = conexion.cursor()

# 2. Leemos las MATERIAS reales
cursor.execute("SELECT nombre, codigo FROM materias WHERE deleted_at IS NULL")
print("=== MATERIAS ===")
for m in cursor.fetchall():
    print(f"- {m[0]} ({m[1]})")

# 3. Leemos los LIBROS-RESUMEN (el catálogo que creó el seeder)
cursor.execute("SELECT nombre FROM libros WHERE deleted_at IS NULL")
print("\n=== LIBROS / RESÚMENES (catálogo del ML) ===")
for l in cursor.fetchall():
    print(f"- {l[0]}")

# 4. Leemos ESTUDIANTES con nombre y celular (join con users)
cursor.execute("""
    SELECT u.nombre, u.apellido_paterno, u.celular
    FROM estudiantes e
    JOIN users u ON e.user_id = u.id
    WHERE e.deleted_at IS NULL
""")
print("\n=== ESTUDIANTES ===")
for e in cursor.fetchall():
    print(f"- {e[0]} {e[1]} | Cel: {e[2]}")

conexion.close()
print("\n✅ ¡Python se conectó a tu base de datos real!")