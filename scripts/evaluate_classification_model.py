import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cargar el modelo y los datos de prueba
print("Cargando modelo y datos de clasificación para evaluación...")
model = load_model('models/lstm_classification_model.h5')
X_test = np.load('data/X_test_clf.npy')
y_test = np.load('data/y_test_clf.npy')

# Hacer predicciones (el modelo devuelve probabilidades)
y_pred_probs = model.predict(X_test)

# Convertir probabilidades a clases (0 o 1) usando un umbral de 0.5
y_pred_classes = (y_pred_probs > 0.5).astype("int32")

# --- Calcular y Mostrar Métricas ---
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"\n📊 Accuracy (Precisión Global): {accuracy:.4f}")
print("\n📋 Reporte de Clasificación:")
print(classification_report(y_test, y_pred_classes, target_names=['Baja/Mantiene', 'Sube']))

# --- Generar Matriz de Confusión ---
print("\n[[ Verdaderos Neg.  Falsos Pos. ]]")
print("[[ Falsos Neg.     Verdaderos Pos. ]]")
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Baja/Mantiene', 'Sube'], 
            yticklabels=['Baja/Mantiene', 'Sube'])
plt.xlabel('Predicción del Modelo')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')

# Guardamos el gráfico de la matriz de confusión
os.makedirs('results', exist_ok=True)
plt.savefig('results/confusion_matrix.png')
print("\n✅ Matriz de confusión guardada en 'results/confusion_matrix.png'.")

plt.show()