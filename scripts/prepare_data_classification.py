# scripts/prepare_data_classification.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

def preparar_datos_clasificacion(input_path='data/btc_features_v3.csv', ventana=60, carpeta_salida='data'):
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    # --- CAMBIO CLAVE 1: Crear el objetivo de clasificación ---
    # Creamos la columna 'target'. Será 1 si el precio de mañana (Close.shift(-1)) es mayor que el de hoy.
    # Será 0 en caso contrario.
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # La última fila tendrá un NaN en 'target', la eliminamos.
    df.dropna(inplace=True)
    
    print("Ejemplo de datos con la nueva columna 'target':")
    print(df[['Close', 'target']].head())

    # Seleccionamos las características (features) para el modelo. Excluimos 'target'.
    features = [col for col in df.columns if col != 'target']
    # El objetivo (target) es la última columna ahora.
    target_col_index = len(features) 
    
    # Normalizar solo las características, no el target.
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])

    # Convertimos a array de numpy para crear secuencias
    datos_completos = df_scaled.to_numpy()

    # --- CAMBIO CLAVE 2: Crear secuencias apuntando a 'target' ---
    X, y = [], []
    for i in range(len(datos_completos) - ventana):
        # 'X' son las ventanas de características normalizadas
        X.append(datos_completos[i:i+ventana, :-1]) # Todas las columnas menos la última (target)
        # 'y' es el valor de 'target' del día siguiente
        y.append(datos_completos[i + ventana, -1]) # La última columna

    X, y = np.array(X), np.array(y)

    # Dividir en train/test
    dividir = int(len(X) * 0.8)
    X_train, X_test = X[:dividir], X[dividir:]
    y_train, y_test = y[:dividir], y[dividir:]
    
    # Guardar el scaler y los nuevos datos
    os.makedirs(carpeta_salida, exist_ok=True)
    joblib.dump(scaler, os.path.join(carpeta_salida, 'scaler_classification.pkl'))
    
    np.save(os.path.join(carpeta_salida, 'X_train_clf.npy'), X_train)
    np.save(os.path.join(carpeta_salida, 'X_test_clf.npy'), X_test)
    np.save(os.path.join(carpeta_salida, 'y_train_clf.npy'), y_train)
    np.save(os.path.join(carpeta_salida, 'y_test_clf.npy'), y_test)

    print("\n✅ Datos para clasificación preparados y guardados.")
    print(f"Forma de X_train: {X_train.shape}, Forma de y_train: {y_train.shape}")

if __name__ == '__main__':
    preparar_datos_clasificacion()