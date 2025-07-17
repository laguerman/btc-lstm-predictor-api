# scripts/train_model.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

print("Cargando datos V2 (con features mejoradas)...")
# --- Cargamos los archivos de la versión 2 ---
X_train = np.load('data/X_train_v2.npy')
y_train = np.load('data/y_train_v2.npy')
X_test = np.load('data/X_test_v2.npy')
y_test = np.load('data/y_test_v2.npy')

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Iniciando entrenamiento del modelo V2...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop], verbose=1)

os.makedirs('models', exist_ok=True)
# --- Guardamos el modelo V2 ---
model.save('models/lstm_model_v2.h5')
print("✅ Modelo V2 entrenado y guardado.")