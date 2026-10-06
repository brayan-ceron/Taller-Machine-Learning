# Pipelines ML · Irradiancia

Aplicación web hecha con **Flask** para comparar modelos de clasificación de machine learning que predicen la **irradiancia solar** a partir de datos satelitales (MODIS y Landsat).

La irradiancia se dividió en tres clases (**Baja, Media y Alta**). La app permite poner dos modelos lado a lado y ver cuál clasifica mejor, tanto en mapas como en métricas.

## Enlaces

- **Aplicación en línea:** https://taller-machine-learning.onrender.com/
- **Video demostrativo (YouTube):** https://www.youtube.com/watch?v=5NaqtUy6aHk

> La app está en el plan gratuito de Render. Si lleva un rato sin visitas, la primera carga puede tardar cerca de un minuto.

## Funcionalidades

1. **Comparar dos modelos (A y B).** Se elige cada modelo con un selector. Pueden diferir en el dataset (MODIS o Landsat), en el preprocesamiento (MinMaxScaler, Normalizer, etc.) o en el clasificador (KSVC, KRidge, etc.).
2. **Mapas interactivos.** Cada punto es una ubicación. Se pueden colorear por:
   - **Predicha:** la clase que predice el modelo.
   - **Real:** la clase verdadera.
   - **Aciertos:** dónde el modelo acierta o falla.
3. **Clasificar una posición.** Al hacer clic en el mapa, o al escribir latitud y longitud y pulsar *Consultar*, ambos modelos clasifican esa ubicación y la app indica si coinciden.
4. **Métricas de calidad (validación cruzada).** Accuracy, F1 macro, MCC, AUC (OvR) y un score compuesto, con la diferencia entre B y A.
5. **Matrices de confusión.** Una por modelo, para ver en qué clases acierta y con cuáles se confunde.
6. **Almacenar modelos.** Se pueden añadir nuevos modelos subiendo sus metadatos (`.json`), sus predicciones (`_predicciones.csv`) y, opcionalmente, el pipeline entrenado (`.joblib`).

## Tecnologías

- Python 3
- Flask
- pandas y numpy
- scikit-learn y joblib (para los pipelines)
- gunicorn (servidor en producción)
- HTML, CSS y JavaScript en la carpeta `templates/`

## Requisitos previos

- Python 3.10 o superior
- Git

## Ejecución en local

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/brayan-ceron/Taller-Machine-Learning.git
   cd Taller-Machine-Learning
   ```


2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Ejecutar la aplicación:
   ```bash
   python app.py
   ```

## Despliegue en Render (gratis)

1. Subir el proyecto a GitHub. `app.py` y `requirements.txt` deben estar en la raíz del repositorio.
2. Crear una cuenta en https://render.com e iniciar sesión con GitHub.
3. Pulsar **New + → Web Service** y seleccionar este repositorio.
4. Configurar:
   - **Language:** Python 3
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
5. Pulsar **Create Web Service**. Al terminar el despliegue, Render entrega una URL pública.
6. Cada `git push` a `main` vuelve a desplegar la app automáticamente.

## Estructura del proyecto

├── app.py               # Servidor Flask y rutas de la app
├── KRidge.py            # Código del modelo Kernel Ridge
├── requirements.txt     # Dependencias
├── models/              # Modelos almacenados (metadatos, predicciones, pipelines)
├── templates/           # Plantillas HTML
└── static/              # Archivos estáticos (CSS, JavaScript)
```

## Autor

Brayan Cerón
