import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from ta.volatility import AverageTrueRange # Importamos ATR

def preparar_datos_v2(input_path='data/btc_quant_ready.csv', ventana=60, carpeta_salida='data'):
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    # --- INGENIERÍA DE CARACTERÍSTICAS (AÑADIMOS NUEVAS FEATURES) ---
    # 1. Añadir price_diff (momentum)
    df['price_diff'] = df['Close'].diff()
    # 2. Añadir ATR (volatilidad)
    atr_indicator = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR'] = atr_indicator.average_true_range()
    
    df.dropna(inplace=True) # Eliminar NaNs generados por las nuevas features

    # El objetivo 'Close' debe ser la primera columna para des-escalar fácilmente
    cols = ['Close'] + [col for col in df.columns if col != 'Close']
    df = df[cols]
    
    print(f"Versión 2 - Columnas para el modelo: {list(df.columns)}")

    scaler = MinMaxScaler()
    datos_normalizados = scaler.fit_transform(df)

    # El objetivo vuelve a ser el precio de cierre ('Close'), que ahora está en el índice 0
    X, y = [], []
    for i in range(len(datos_normalizados) - ventana):
        X.append(datos_normalizados[i:i+ventana])
        y.append(datos_normalizados[i + ventana, 0])

    X, y = np.array(X), np.array(y)

    dividir = int(len(X) * 0.8)
    X_train, X_test = X[:dividir], X[dividir:]
    y_train, y_test = y[:dividir], y[dividir:]

    os.makedirs(carpeta_salida, exist_ok=True)
    joblib.dump(scaler, os.path.join(carpeta_salida, 'scaler_v2.pkl'))
    
    # Guardamos los nuevos archivos con un sufijo _v2
    np.save(os.path.join(carpeta_salida, 'X_train_v2.npy'), X_train)
    np.save(os.path.join(carpeta_salida, 'X_test_v2.npy'), X_test)
    np.save(os.path.join(carpeta_salida, 'y_train_v2.npy'), y_train)
    np.save(os.path.join(carpeta_salida, 'y_test_v2.npy'), y_test)

    print("✅ V2 - Datos con nuevas características preparados y guardados.")

if __name__ == '__main__':
    preparar_datos_v2()