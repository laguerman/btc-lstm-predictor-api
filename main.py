# main.py

# --- 1. Importaciones ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
import pandas as pd

# --- 2. Modelo Pydantic para la Entrada de Datos ---
# Se asegura que los datos de entrada tengan la estructura correcta.
class InputData(BaseModel):
    # Field se usa para añadir validaciones y ejemplos en la documentación
    datos: List[List[float]] = Field(..., example=[[0.5]*15]*60)

# --- 3. Carga Global de Artefactos (Modelo y Scaler) ---
# Se cargan una sola vez al iniciar la API para máxima eficiencia.
MODEL_PATH = 'models/lstm_classification_model.h5'
SCALER_PATH = 'data/scaler_classification.pkl'

model = None
scaler = None
N_FEATURES = 0

try:
    print("Cargando modelo y scaler para la API...")
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        N_FEATURES = scaler.n_features_in_
        print(f"✅ Modelo y scaler cargados. El modelo espera {N_FEATURES} características.")
        model.summary() # Muestra un resumen de la arquitectura del modelo
    else:
        print("❌ Error: No se encontraron los archivos del modelo o del scaler.")
        
except Exception as e:
    print(f"❌ Error crítico al cargar los artefactos: {e}")

# --- 4. Creación de la Instancia de la App FastAPI ---
app = FastAPI(
    title="🤖 API de Predicción Direccional de BTC",
    description="""
    Esta API utiliza un modelo LSTM para predecir si el precio de Bitcoin 
    subirá o bajará en el próximo período de 24 horas.
    Creada por Luciano - ¡Un proyecto de Quant-AI!
    """,
    version="1.0.0"
)

# --- 5. Definición del Endpoint de Predicción ---
@app.post("/predecir_direccion/", tags=["Predicciones"])
async def predecir_direccion(input_data: InputData):
    """
    Recibe una secuencia de 60 días de datos de mercado y devuelve
    la predicción de dirección ('Sube' o 'Baja/Mantiene') y la 
    probabilidad de que la predicción sea 'Sube'.
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo no está operativo. Contacte al administrador.")

    datos_array = np.array(input_data.datos)

    # Validación de la forma de los datos
    if datos_array.shape != (60, N_FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f"Datos de entrada inválidos. Se esperaba una forma (60, {N_FEATURES}), pero se recibió {datos_array.shape}."
        )

    # Escalar los datos de entrada
    datos_scaled = scaler.transform(datos_array)

    # Reshape para el modelo LSTM (1 muestra, 60 timesteps, N features)
    datos_reshaped = np.reshape(datos_scaled, (1, datos_scaled.shape[0], datos_scaled.shape[1]))

    # Realizar la predicción
    prediction_prob = model.predict(datos_reshaped)[0][0]

    # Tomar la decisión
    decision = "Sube" if prediction_prob > 0.5 else "Baja/Mantiene"

    return {
        "decision_predicha": decision,
        "confianza_de_subida": round(float(prediction_prob), 4)
    }

# --- 6. Endpoint Raíz (para verificar el estado de la API) ---
@app.get("/", tags=["Status"])
async def read_root():
    """
    Endpoint raíz para verificar que la API está funcionando.
    """
    return {"status": "¡La API del modelo de clasificación está viva y coleando! Visita /docs para probarla."}