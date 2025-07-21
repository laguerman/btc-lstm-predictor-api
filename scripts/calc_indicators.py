import pandas as pd
import pandas_ta as ta
import os

def generar_indicadores(input_path='data/btc_raw.csv', output_path='data/btc_features.csv'):
    print("Calculando indicadores técnicos...")
    try:
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)

        cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        CustomStrategy = ta.Strategy(
            name="Estrategia_Personalizada",
            ta=[{"kind": "rsi"}, {"kind": "macd"}, {"kind": "bbands"}, {"kind": "obv"}, {"kind": "atr"}, {"kind": "adx"}, {"kind": "stoch"}]
        )
        df.ta.strategy(CustomStrategy)
        df.dropna(inplace=True)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path)
        print(f"Archivo con características guardado en: {output_path}")

    except Exception as e:
        print(f"ERROR al calcular indicadores: {e}")

if __name__ == '__main__':
    generar_indicadores()