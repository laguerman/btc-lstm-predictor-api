from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

class InputData(BaseModel):
    datos: List[List[float]] = Field(..., example=[[0.5]*14]*60)

MODEL_PATH = 'models/lstm_model_v2.h5'
SCALER_PATH = 'data/scaler.pkl'

model = None
scaler = None
N_FEATURES = 0

try:
    print("Cargando Modelo V2 y Scaler para la API...")
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        N_FEATURES = scaler.n_features_in_
        print(f"✅ Modelo V2 y Scaler cargados. El modelo espera {N_FEATURES} características.")
    else:
        print(f"❌ Error: No se encontraron los archivos del Modelo V2 o del Scaler V2.")
        
except Exception as e:
    print(f"❌ Error crítico al cargar los artefactos: {e}")

app = FastAPI(
    title="📈 API de Predicción de Precio de BTC (Modelo V2)",
    description="""
    API que utiliza un modelo LSTM V2 (R² de 0.97) para predecir el precio
    de cierre de Bitcoin para el día siguiente. Un proyecto de laguerman y Study.
    """,
    version="2.0.0"
)

@app.post("/predecir_precio/", tags=["Predicciones"])
async def predecir_precio(input_data: InputData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo no operativo.")

    datos_array = np.array(input_data.datos)
    
    if datos_array.shape != (60, N_FEATURES):
        raise HTTPException(status_code=400, detail=f"Datos de entrada inválidos. Se esperaba (60, {N_FEATURES}).")

    datos_reshaped = np.reshape(datos_array, (1, datos_array.shape[0], datos_array.shape[1]))
    prediction_scaled = model.predict(datos_reshaped)

    dummy_array = np.zeros((1, N_FEATURES))
    dummy_array[0, 0] = prediction_scaled[0, 0]
    prediction_real = scaler.inverse_transform(dummy_array)[0, 0]

    return {"prediccion_btc_usd": round(float(prediction_real), 2)}

@app.get("/", tags=["Status"])
async def read_root():
    return {"status": "API del Modelo V2 está funcionando. Visita /docs para probarla."}