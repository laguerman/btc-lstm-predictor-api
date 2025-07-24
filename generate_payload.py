# generate_payload.py

import numpy as np
import pandas as pd
import json
import joblib
from ta.volatility import AverageTrueRange

print("🚀 Generando archivo 'payload.json' con datos de prueba consistentes...")

try:
    # 1. Cargamos el MISMO scaler que usa la API para saber el número de features
    scaler = joblib.load('data/scaler.pkl')
    N_FEATURES = scaler.n_features_in_
    print(f"El modelo espera {N_FEATURES} características por cada paso de tiempo.")

    # 2. Cargamos el dataset con todas las features para obtener datos NO escalados
    df_features = pd.read_csv('data/btc_features.csv', index_col=0, parse_dates=True)

    # 3. Re-aplicamos las mismas transformaciones que en prepare_data.py
    #    para asegurar que el número de columnas sea idéntico.
    df_features['price_diff'] = df_features['Close'].diff()
    atr_indicator = AverageTrueRange(high=df_features['High'], low=df_features['Low'], close=df_features['Close'], window=14)
    df_features['ATR'] = atr_indicator.average_true_range()
    df_features.dropna(inplace=True)

    # Ordenamos las columnas EXACTAMENTE igual que para el entrenamiento
    cols = ['Close'] + [col for col in df_features.columns if col != 'Close']
    df_final = df_features[cols]

    # Verificamos que el número de columnas coincida
    if df_final.shape[1] != N_FEATURES:
        print(f"❌ ¡ALERTA DE INCONSISTENCIA! El scaler espera {N_FEATURES} features, pero hemos generado {df_final.shape[1]}.")
        exit()

    # 4. Tomamos los últimos 60 días de datos NO escalados como nuestra muestra
    sample_unscaled = df_final.tail(60)

    # 5. Creamos el diccionario para el JSON
    datos_para_api = {"datos": sample_unscaled.to_numpy().tolist()}

    # 6. Guardamos el JSON en un archivo
    file_path = "payload.json"
    with open(file_path, "w") as f:
        json.dump(datos_para_api, f, indent=2)

    print(f"\n✅ ¡Éxito! Archivo '{file_path}' creado/actualizado con la forma correcta (60, {N_FEATURES}).")
    print("\n📋 Próximos pasos:")
    print("1. Abre el archivo 'payload.json'.")
    print("2. Copia TODO su contenido.")
    print("3. Pégalo en la interfaz de la API y haz clic en 'Execute'.")

except Exception as e:
    print(f"❌ Ocurrió un error inesperado: {e}")