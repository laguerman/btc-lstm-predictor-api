# 🤖 API de Predicción Direccional de BTC con LSTM

![Estado de la API](https://img.shields.io/website?up_message=online&down_message=offline&url=https%3A%2F%2Fbtc-lstm-predictor-api-lguerman.onrender.com%2F)
![Python Version](https://img.shields.io/badge/python-3.10-blue)
![Framework](https://img.shields.io/badge/Framework-FastAPI-green)
![Modelo](https://img.shields.io/badge/Modelo-TensorFlow%2FKeras-orange)

Este repositorio contiene el código fuente de una API web que utiliza un modelo de Deep Learning (LSTM) para predecir la dirección del precio de Bitcoin (BTC) para el día siguiente. El proyecto abarca el ciclo de vida completo de un proyecto de Machine Learning, desde la recolección y procesamiento de datos hasta el entrenamiento, evaluación y despliegue de un modelo como un servicio web en la nube.

---

## 🚀 API en Vivo

¡Puedes probar la API ahora mismo! Está desplegada en Render y es accesible a través de la siguiente URL:

**[https://btc-lstm-predictor-api-lguerman.onrender.com/docs](https://btc-lstm-predictor-api-lguerman.onrender.com/docs)**

### Ejemplo de Uso con `curl`

```bash
# Nota: Reemplaza el "[...]" con los datos de entrada en formato JSON.
curl -X 'POST' \
  'https://btc-lstm-predictor-api-lguerman.onrender.com/predecir_direccion/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{ "datos": [[...]] }'
```

---

## 📋 Características del Proyecto

*   **Pipeline de Datos Automatizado:** Un script se ejecuta automáticamente al iniciar el proyecto para descargar los datos más recientes de Yahoo Finance, calcular indicadores técnicos y preprocesar la información.
*   **Modelo LSTM de Clasificación:** Se entrenó una Red Neuronal Recurrente (LSTM) para un problema de clasificación binaria (predecir si el precio 'Sube' o 'Baja/Mantiene').
*   **API Robusta con FastAPI:** El modelo se sirve a través de una API web rápida y moderna, con documentación interactiva generada automáticamente.
*   **Despliegue Continuo en la Nube:** El proyecto está alojado en GitHub y se despliega automáticamente en Render con cada `push` a la rama `main`.
*   **Gestión de Entorno Profesional:** Utiliza entornos virtuales (`venv`) y un archivo `requirements.txt` para una reproducibilidad total.

---

## 🧠 Resultados y Conclusiones del Modelo

Tras varias iteraciones de modelado (predicción de precios, retornos logarítmicos, diferencias y finalmente clasificación), se concluyó que predecir la dirección del mercado basándose únicamente en datos históricos de precio y sus derivados es extremadamente complejo.

El modelo final de clasificación alcanzó una **precisión (accuracy) de aproximadamente el 50%**, similar a un resultado aleatorio.

**Matriz de Confusión del Modelo Final:**
![Matriz de Confusión](https://github.com/laguerman/btc-lstm-predictor-api/blob/main/results/prediction_classification_model.png?raw=true)
*(Nota: Necesitarás subir la imagen de la matriz a la carpeta `results` en GitHub para que esto funcione).*

Este resultado valida empíricamente la Hipótesis del Mercado Eficiente (forma débil) y demuestra la necesidad de incorporar fuentes de datos alternativas (ej. on-chain, sentimiento) para obtener una ventaja predictiva real.

---

## 🛠️ Stack Tecnológico

*   **Lenguaje:** Python 3.10
*   **Machine Learning:** TensorFlow/Keras, Scikit-Learn
*   **Análisis de Datos:** Pandas, NumPy, TA-Lib
*   **API:** FastAPI, Uvicorn
*   **Despliegue:** Render, Git, GitHub

---

## 🚀 Cómo Ejecutar en Local

1.  Clonar el repositorio:
    ```bash
    git clone https://github.com/laguerman/btc-lstm-predictor-api.git
    cd btc-lstm-predictor-api
    ```
2.  Crear y activar un entorno virtual con Python 3.10:
    ```bash
    python -m venv .venv
    source .venv/Scripts/activate
    ```
3.  Instalar las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
4.  Lanzar la API:
    ```bash
    uvicorn main:app --reload
    ```
    La API estará disponible en `http://127.0.0.1:8000`.

---

## 🔮 Futuras Mejoras

*   **Modelo V4:** Integrar datos on-chain y de sentimiento para mejorar la precisión del modelo.
*   **Modelado por Regímenes:** Desarrollar un sistema que detecte el estado del mercado (tendencia, lateral) y utilice modelos especializados para cada régimen.