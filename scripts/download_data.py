import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def descargar_datos(ticker='BTC-USD', start='2015-01-01', carpeta_salida='data'):
    """Descarga datos históricos y los guarda en un CSV."""
    end = datetime.today().strftime('%Y-%m-%d')
    print(f"Descargando datos para {ticker} desde {start} hasta {end}...")
    
    try:
        btc_data = yf.download(ticker, start=start, end=end)
        
        os.makedirs(carpeta_salida, exist_ok=True)
        ruta_salida = os.path.join(carpeta_salida, 'btc_raw.csv')
        btc_data.to_csv(ruta_salida)
        
        print(f"Datos guardados exitosamente en: {ruta_salida}")
    except Exception as e:
        print(f"❌ Error al descargar los datos: {e}")

if __name__ == "__main__":
    descargar_datos()