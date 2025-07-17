# scripts/train_classification_model.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

print("Cargando datos para el modelo de clasificación...")
X_train = np.load('data/X_train_clf.npy')
y_train = np.load('data/y_train_clf.npy')
X_test = np.load('data/X_test_clf.npy')
y_test = np.load('data/y_test_clf.npy')

# --- CAMBIOS EN LA ARQUITECTURA Y COMPILACIÓN ---
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    # Capa de salida para clasificación binaria
    Dense(units=1, activation='sigmoid') # 👈 'sigmoid' para dar una probabilidad
])

# Compilamos con loss y métricas para clasificación
model.compile(optimizer='adam', 
              loss='binary_crossentropy', # 👈 Loss para clasificación
              metrics=['accuracy']) # 👈 Métrica para ver el % de acierto

early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)

print("Iniciando entrenamiento del modelo de clasificación...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop], verbose=1)

os.makedirs('models', exist_ok=True)
model.save('models/lstm_classification_model.h5')
print("\n✅ Modelo de clasificación entrenado y guardado.")