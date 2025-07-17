# scripts/calc_indicators.py

import pandas as pd
import pandas_ta as ta
import os

def generar_features_v3(input_path='data/btc_raw.csv', output_path='data/btc_features_v3.csv'):
    # Cargar los datos crudos
    df = pd.read_csv(input_path, index_col='Date', parse_dates=True)
    
    # --- CAMBIO CLAVE: Usar una Estrategia de pandas_ta ---
    # Creamos una estrategia personalizada. 'Common' es una estrategia que
    # incluye una buena mezcla de indicadores comunes y potentes.
    # Esto generará automáticamente decenas de columnas de features.
    CustomStrategy = ta.Strategy(
        name="Common_Indicators",
        description="RSI, MACD, BBands, OBV, etc.",
        ta=[
            {"kind": "rsi"},
            {"kind": "macd"},
            {"kind": "bbands"},
            {"kind": "obv"}, # On-Balance Volume, muy importante
            {"kind": "atr"}, # Average True Range
            {"kind": "adx"}, # Average Directional Index
            {"kind": "stoch"}, # Stochastic Oscillator
        ]
    )
    
    # Aplicamos la estrategia al DataFrame.
    # ¡Pandas_ta se encarga de todo!
    df.ta.strategy(CustomStrategy)
    
    # Limpiamos las filas que tengan valores NaN (generados al principio)
    df.dropna(inplace=True)
    
    print(f"✅ Se generaron {len(df.columns)} columnas de características.")
    print("Primeras 5 columnas como ejemplo:", list(df.columns[:5]))

    # Guardamos el nuevo dataset enriquecido
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Archivo V3 con features enriquecidas guardado en: {output_path}")

if __name__ == '__main__':
    generar_features_v3()