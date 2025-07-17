# scripts/evaluate_model.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os

# --- 1. Cargar el modelo, datos y scaler V2 ---
print("Cargando modelo y datos V2...")
model = load_model('models/lstm_model_v2.h5')
X_test = np.load('data/X_test_v2.npy')
y_test_scaled = np.load('data/y_test_v2.npy')
scaler = joblib.load('data/scaler_v2.pkl')

# --- 2. Hacer Predicciones ---
print("Haciendo predicciones con el modelo V2...")
predictions_scaled = model.predict(X_test)

# --- 3. Des-normalizar los Datos ---
# El scaler fue ajustado a un df donde 'Close' era la primera columna.
n_features = scaler.n_features_in_
dummy_array_pred = np.zeros((len(predictions_scaled), n_features))
dummy_array_pred[:, 0] = predictions_scaled.flatten()
predictions_real = scaler.inverse_transform(dummy_array_pred)[:, 0]

dummy_array_test = np.zeros((len(y_test_scaled), n_features))
dummy_array_test[:, 0] = y_test_scaled.flatten()
y_test_real = scaler.inverse_transform(dummy_array_test)[:, 0]

# --- 4. Evaluar y Visualizar ---
rmse = np.sqrt(mean_squared_error(y_test_real, predictions_real))
mae = mean_absolute_error(y_test_real, predictions_real)
r2 = r2_score(y_test_real, predictions_real)

print("\n📊 Métricas de Evaluación (Modelo V2):")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 MAE:  {mae:.2f}")
print(f"🔹 R²:   {r2:.4f}")

plt.figure(figsize=(15, 7))
plt.plot(y_test_real, color='blue', label='Precio Real de BTC')
plt.plot(predictions_real, color='red', linestyle='--', label='Predicción del Modelo V2')
plt.title('Predicción de Precio de BTC (Modelo V2 con Features Mejoradas)')
plt.xlabel('Tiempo (días en el conjunto de prueba)')
plt.ylabel('Precio de Bitcoin (USD)')
plt.legend()
plt.grid(True)

os.makedirs('results', exist_ok=True)
plt.savefig('results/prediction_model_v2.png')
print("\n✅ Gráfico V2 guardado.")
plt.show()