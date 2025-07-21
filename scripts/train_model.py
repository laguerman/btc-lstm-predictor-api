import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

def entrenar_modelo():
    print("Cargando datos para entrenamiento...")
    try:
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
    except FileNotFoundError:
        print("ERROR: Archivos de datos (.npy) no encontrados. Ejecuta 'prepare_data.py' primero.")
        return

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("Iniciando entrenamiento del modelo...")
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[early_stop], verbose=1)

    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_model.h5')
    print("✅ Modelo final entrenado y guardado en 'models/lstm_model.h5'.")

if __name__ == "__main__":
    entrenar_modelo()