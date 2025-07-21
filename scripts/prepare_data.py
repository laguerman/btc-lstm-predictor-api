import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from ta.volatility import AverageTrueRange

def preparar_datos(input_path='data/btc_features.csv', ventana=60, carpeta_salida='data'):
    print("Iniciando preparación de datos para el modelo...")
    try:
        # CORRECCIÓN: Leer el archivo con index_col=0
        df = pd.read_csv(input_path, index_col=0, parse_dates=True) 
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        if 'price_diff' not in df.columns:
            df['price_diff'] = df['Close'].diff()
        if 'ATR' not in df.columns and all(c in df.columns for c in ['High', 'Low', 'Close']):
            atr_indicator = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            df['ATR'] = atr_indicator.average_true_range()
        
        df.dropna(inplace=True)

        if df.empty or len(df) <= ventana:
            print("ERROR: No hay suficientes datos para crear secuencias después del preprocesamiento.")
            return

        cols = ['Close'] + [col for col in df.columns if col != 'Close']
        df = df[cols]
        
        scaler = MinMaxScaler()
        datos_normalizados = scaler.fit_transform(df)

        X, y = [], []
        for i in range(len(datos_normalizados) - ventana):
            X.append(datos_normalizados[i:i+ventana])
            y.append(datos_normalizados[i + ventana, 0])

        X, y = np.array(X), np.array(y)
        
        dividir = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X[:dividir], X[dividir:], y[:dividir], y[dividir:]

        os.makedirs(carpeta_salida, exist_ok=True)
        joblib.dump(scaler, os.path.join(carpeta_salida, 'scaler.pkl'))
        
        np.save(os.path.join(carpeta_salida, 'X_train.npy'), X_train)
        np.save(os.path.join(carpeta_salida, 'X_test.npy'), X_test)
        np.save(os.path.join(carpeta_salida, 'y_train.npy'), y_train)
        np.save(os.path.join(carpeta_salida, 'y_test.npy'), y_test)

        print("✅ Datos preparados y guardados (scaler.pkl, X_train.npy, etc.).")

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de entrada en {input_path}")
    except Exception as e:
        print(f"ERROR al preparar los datos: {e}")

if __name__ == '__main__':
    preparar_datos()