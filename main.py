from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from pathlib import Path

# --- 2. Construir rutas absolutas a los artefactos ---
# Esto asegura que siempre encontraremos los archivos, sin importar
# desde qué directorio se ejecute el script.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "lstm_model.h5"
SCALER_PATH = BASE_DIR / "data" / "scaler.pkl"

class InputData(BaseModel):
    datos: List[List[float]] = Field(..., example=[[0.5]*14]*60)

model = None
scaler = None
N_FEATURES = 0

try:
    print("Cargando Modelo y Scaler con rutas absolutas...")
    print(f"Buscando modelo en: {MODEL_PATH}")
    print(f"Buscando scaler en: {SCALER_PATH}")

    if MODEL_PATH.exists() and SCALER_PATH.exists():
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        N_FEATURES = scaler.n_features_in_
        print(f"✅ Modelo y Scaler cargados. El modelo espera {N_FEATURES} características.")
    else:
        print(f"❌ Error: No se encontraron los archivos en las rutas especificadas.")
        
except Exception as e:
    print(f"❌ Error crítico al cargar los artefactos: {e}")

app = FastAPI(
    title="📈 API de Predicción de Precio de BTC (Modelo Final)",
    description="""
    API que utiliza un modelo LSTM para predecir el precio de cierre de Bitcoin.
    Un proyecto de Luciano y Study.
    """,
    version="3.0.0"
)

@app.post("/predecir_precio/", tags=["Predicciones"])
async def predecir_precio(input_data: InputData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo no operativo. Revisar logs del servidor.")

    datos_array = np.array(input_data.datos)
    
    if datos_array.shape != (60, N_FEATURES):
        raise HTTPException(status_code=400, detail=f"Datos de entrada inválidos. Se esperaba (60, {N_FEATURES}).")

    # Los datos que llegan ya están sin escalar, el scaler lo usamos para des-normalizar
    # NOTA: La API espera datos brutos, no escalados. El escalado se hace internamente.
    # Pero nuestro modelo V2 fue entrenado con datos escalados, así que debemos escalar la entrada.
    datos_scaled = scaler.transform(datos_array) # 👈 AÑADIMOS ESTE PASO

    datos_reshaped = np.reshape(datos_scaled, (1, datos_scaled.shape[0], datos_scaled.shape[1])) # 👈 USAMOS DATOS ESCALADOS
    prediction_scaled = model.predict(datos_reshaped)

    dummy_array = np.zeros((1, N_FEATURES))
    dummy_array[0, 0] = prediction_scaled[0, 0]
    prediction_real = scaler.inverse_transform(dummy_array)[0, 0]

    return {"prediccion_btc_usd": round(float(prediction_real), 2)}

@app.get("/", tags=["Status"])
async def read_root():
    return {"status": "API del Modelo Final está funcionando. Visita /docs para probarla."}