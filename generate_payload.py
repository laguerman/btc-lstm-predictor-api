# generate_payload.py

import numpy as np
import json
import joblib

print("🚀 Generando archivo 'payload.json' con datos de prueba...")

try:
    # 1. Cargar el scaler y los datos de prueba
    scaler = joblib.load('data/scaler_classification.pkl')
    X_test_scaled = np.load('data/X_test_clf.npy')

    # 2. Tomar una muestra (la primera ventana de 60 días)
    sample_scaled = X_test_scaled[0]

    # 3. Des-escalar la muestra para simular la entrada real que espera la API
    sample_unscaled = scaler.inverse_transform(sample_scaled)

    # 4. Crear el diccionario para el JSON
    datos_para_api = {"datos": sample_unscaled.tolist()}

    # 5. Guardar el diccionario en un archivo JSON
    file_path = "payload.json"
    with open(file_path, "w") as f:
        json.dump(datos_para_api, f, indent=2) # Con indentación para que sea legible

    print(f"✅ ¡Éxito! Archivo '{file_path}' creado/actualizado.")
    print("\n📋 Próximos pasos:")
    print("1. Abre el archivo 'payload.json' en VS Code.")
    print("2. Copia TODO su contenido.")
    print("3. Pégalo en la interfaz de la API en Render y haz clic en 'Execute'.")

except FileNotFoundError as e:
    print(f"❌ ERROR: No se pudo encontrar un archivo necesario: {e}")
    print("Asegúrate de haber ejecutado los scripts de preparación y entrenamiento primero.")
except Exception as e:
    print(f"❌ Ocurrió un error inesperado: {e}")