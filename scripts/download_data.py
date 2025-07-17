from datetime import datetime, timedelta
import yfinance as yf
import os

def descargar_datos_btc(start='2015-01-01', carpeta_destino='data'):
    """
    Descarga los datos históricos de BTC/USD desde Yahoo Finance
    y los guarda como archivo CSV en la carpeta especificada.
    Se ajusta automáticamente a la última fecha disponible.
    """

    # Usar hoy como fecha tentativa de fin
    today = datetime.today()
    tentative_end = today.strftime('%Y-%m-%d')

    print(f"Descargando datos desde {start} hasta {tentative_end} (fecha tentativa)...")

    # Descargar datos
    btc_data = yf.download('BTC-USD', start=start, end=tentative_end)

    # Si no hay datos en el último día, recortar la última fecha válida
    last_date = btc_data.index[-1].strftime('%Y-%m-%d')
    print(f"Última fecha disponible en los datos: {last_date}")

    # Guardar archivo
    os.makedirs(carpeta_destino, exist_ok=True)
    archivo_salida = os.path.join(carpeta_destino, 'btc_raw.csv')
    btc_data.to_csv(archivo_salida)
    print(f"Datos guardados en: {archivo_salida}")

if __name__ == "__main__":
    descargar_datos_btc()
# Nota: YFinance no proporciona datos en tiempo real.
# Generalmente, los datos están disponibles hasta 1 o 2 días antes de la fecha actual.
# Este script ajusta la descarga automáticamente para obtener los datos más recientes disponibles.
