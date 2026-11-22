"""
generador_datos.py - Simulador de Historial Académico a Nivel de TEMA (v6)
========================================================================
Genera dataset sintético estructurado estrictamente por TEMA (la unidad
invariable entre gestiones 2026 -> 2027 -> 2028).

Features generadas para el modelo:
- materia_id: ID categórico de la materia
- tema_orden: Secuencia numérica del tema dentro de la materia (1, 2, 3...)
- nota_previa: Calificación previa consolidada del estudiante (0 - 100)
- leyo_material: Proxy de preparación del estudiante (1 = repasó, 0 = no repasó)
Target:
- mejora: Diferencia de rendimiento (nota_tema - nota_previa)
"""

import os
import random
import pymysql
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "psicologia_proyecto")

def cargar_datos_base():
    """Intenta cargar materias y temas desde MySQL; si falla, usa catálogo predeterminado."""
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset="utf8mb4"
        )
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT t.id, t.nombre, t.materia_id, m.nombre as materia_nombre, t.paginas_libro, t.orden
                FROM temas t
                JOIN materias m ON t.materia_id = m.id
                WHERE t.deleted_at IS NULL AND m.deleted_at IS NULL
                ORDER BY t.materia_id, t.orden ASC
            """)
            temas = cursor.fetchall()

            cursor.execute("""
                SELECT e.id, u.nombre, u.apellido_paterno
                FROM estudiantes e
                JOIN users u ON e.user_id = u.id
                WHERE e.deleted_at IS NULL
            """)
            estudiantes = cursor.fetchall()
        conn.close()

        if temas and estudiantes:
            print(f"✅ Conexión exitosa a MySQL: {len(temas)} temas y {len(estudiantes)} estudiantes cargados.")
            return temas, estudiantes
    except Exception as e:
        print(f"⚠️ Aviso: No se pudo conectar a MySQL ({e}). Usando catálogo base de respaldo.")

    # Catálogo de respaldo representativo de la carrera
    temas_respaldo = [
        (1, "Tema 1: Qué es la Psicología", 1, "INTRODUCCIÓN A LA PSICOLOGÍA", "págs. 5-8", 1),
        (2, "Tema 2: Historia y Evolución", 1, "INTRODUCCIÓN A LA PSICOLOGÍA", "págs. 9-20", 2),
        (3, "Tema 3: Procesos Psicológicos Básicos", 1, "INTRODUCCIÓN A LA PSICOLOGÍA", "págs. 21-30", 3),
        (4, "Tema 4: Motivación y Emoción", 1, "INTRODUCCIÓN A LA PSICOLOGÍA", "págs. 31-33", 4),
        (5, "Tema 5: Áreas de Estudio", 1, "INTRODUCCIÓN A LA PSICOLOGÍA", "págs. 34-37", 5),
        (6, "Tema 6: Nuevas Áreas Emergentes", 1, "INTRODUCCIÓN A LA PSICOLOGÍA", "págs. 38-42", 6),
        (7, "Tema 1: Definición de Salud", 2, "REALIDAD NACIONAL Y SALUD MENTAL", "págs. 43-44", 1),
        (8, "Tema 2: Salud Mental en Bolivia", 2, "REALIDAD NACIONAL Y SALUD MENTAL", "págs. 45-47", 2),
        (9, "Tema 3: Factores Psicosociales", 2, "REALIDAD NACIONAL Y SALUD MENTAL", "págs. 48-52", 3),
        (10, "Tema 4: Políticas de Salud", 2, "REALIDAD NACIONAL Y SALUD MENTAL", "págs. 53-58", 4),
    ]
    estudiantes_respaldo = [(i, f"Estudiante_{i}", "Simulado") for i in range(1, 31)]
    return temas_respaldo, estudiantes_respaldo


def generar_dataset(num_estudiantes=50):
    temas, estudiantes = cargar_datos_base()
    random.seed(42)

    # Agrupar temas por materia
    materias_dict = {}
    for t in temas:
        m_id = t[2]
        if m_id not in materias_dict:
            materias_dict[m_id] = []
        materias_dict[m_id].append(t)

    filas = []

    # Generamos datos para una población simulada representativa
    total_estudiantes = max(len(estudiantes), num_estudiantes)

    for est_idx in range(1, total_estudiantes + 1):
        for m_id, lista_temas in materias_dict.items():
            lista_ordenada = sorted(lista_temas, key=lambda x: x[5])  # ordenar por 'orden'

            # Nota base con la que inicia la materia el estudiante (distribución normal centrada en 60)
            nota_actual = round(max(30.0, min(95.0, random.gauss(60, 15))), 1)

            for i, tema in enumerate(lista_ordenada):
                tema_id, tema_nombre, materia_id, materia_nombre, paginas_libro, orden = tema

                # ¿El estudiante preparó / repasó este tema? (probabilidad 55%)
                leyo_material = 1 if random.random() < 0.55 else 0

                # Impacto pedagógico del repaso de material (heurística)
                if leyo_material == 1:
                    bono_repaso = random.uniform(16.0, 28.0)
                else:
                    bono_repaso = random.uniform(-8.0, 6.0)

                # Efecto de dificultad progresiva según el orden del tema
                dificultad = -0.8 * (orden - 1)
                ruido = random.uniform(-5.0, 5.0)

                nueva_nota = nota_actual + bono_repaso + dificultad + ruido
                nueva_nota = round(max(10.0, min(100.0, nueva_nota)), 1)
                mejora = round(nueva_nota - nota_actual, 1)

                filas.append({
                    "estudiante_id": est_idx,
                    "materia_id": materia_id,
                    "tema_id": tema_id,
                    "tema_orden": orden,
                    "nota_previa": nota_actual,
                    "leyo_material": leyo_material,
                    "nota_tema": nueva_nota,
                    "mejora": mejora,
                })

                # La nueva nota pasa a ser la nota previa del siguiente tema
                nota_actual = nueva_nota

    df = pd.DataFrame(filas)
    archivo_csv = os.path.join(os.path.dirname(__file__), "datos_sinteticos.csv")
    df.to_csv(archivo_csv, index=False, encoding="utf-8-sig")

    print(f"✅ Dataset a nivel de TEMA generado: {len(df)} registros guardados en {archivo_csv}")
    print("\n--- Promedios por leyo_material ---")
    print(df.groupby("leyo_material")[["nota_previa", "nota_tema", "mejora"]].mean().round(2))
    return df


if __name__ == "__main__":
    generar_dataset()
