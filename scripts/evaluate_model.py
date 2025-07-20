# scripts/evaluate_model.py

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os

def evaluar_modelo():
    """Carga el modelo final, evalúa su rendimiento y genera un gráfico."""
    print("Iniciando evaluación del modelo final...")
    try:
        model = load_model('models/lstm_model.h5')
        X_test = np.load('data/X_test.npy')
        y_test_scaled = np.load('data/y_test.npy')
        scaler = joblib.load('data/scaler.pkl')
    except FileNotFoundError:
        print("❌ Error: Archivos de modelo o datos no encontrados. Asegúrate de entrenar el modelo primero.")
        return

    print("Haciendo predicciones...")
    predictions_scaled = model.predict(X_test)

    n_features = scaler.n_features_in_
    dummy_pred = np.zeros((len(predictions_scaled), n_features))
    dummy_pred[:, 0] = predictions_scaled.flatten()
    predictions_real = scaler.inverse_transform(dummy_pred)[:, 0]

    dummy_test = np.zeros((len(y_test_scaled), n_features))
    dummy_test[:, 0] = y_test_scaled.flatten()
    y_test_real = scaler.inverse_transform(dummy_test)[:, 0]

    rmse = np.sqrt(mean_squared_error(y_test_real, predictions_real))
    mae = mean_absolute_error(y_test_real, predictions_real)
    r2 = r2_score(y_test_real, predictions_real)

    print(f"\n📊 Métricas de Evaluación Final:")
    print(f"🔹 RMSE: {rmse:.2f}")
    print(f"🔹 MAE:  {mae:.2f}")
    print(f"🔹 R²:   {r2:.4f}")

    plt.figure(figsize=(15, 7))
    plt.plot(y_test_real, color='blue', label='Precio Real de BTC')
    plt.plot(predictions_real, color='red', linestyle='--', label='Predicción del Modelo')
    plt.title('Predicción de Precio de BTC (Modelo Final)')
    plt.legend()
    plt.grid(True)

    os.makedirs('results', exist_ok=True)
    plt.savefig('results/final_prediction_chart.png')
    print("\n✅ Gráfico final guardado en 'results/final_prediction_chart.png'.")
    plt.show()

if __name__ == "__main__":
    evaluar_modelo()