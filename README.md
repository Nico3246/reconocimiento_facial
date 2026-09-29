# Reconocimiento facial

Sistema de reconocimiento facial en tiempo real desarrollado en Python. El proyecto detecta rostros con un modelo SSD basado en Caffe, genera representaciones faciales con OpenFace y entrena un clasificador SVM para identificar personas desde una webcam.

## Características

- Detección de rostros con OpenCV DNN y un modelo SSD Caffe.
- Generación de embeddings faciales de 128 dimensiones mediante OpenFace.
- Entrenamiento de un clasificador SVM a partir de los embeddings.
- Reconocimiento en tiempo real utilizando la cámara del equipo.
- Visualización del nombre y la probabilidad estimada sobre cada rostro detectado.
- Identificación como `Desconocido` cuando la confianza del clasificador es inferior al umbral configurado.

## Flujo del sistema

```text
dataset/
   ↓
Detección facial
OpenCV + Caffe SSD
   ↓
Extracción de embeddings
OpenFace
   ↓
output/embeddings.pickle
   ↓
Entrenamiento SVM
scikit-learn
   ↓
recognizer.pickle + le.pickle
   ↓
Reconocimiento en tiempo real
Webcam + OpenCV
```

## Tecnologías

- Python
- OpenCV
- OpenFace
- Caffe
- NumPy
- imutils
- scikit-learn
- Pickle

## Estructura principal

```text
reconocimiento_facial/
├── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── openface/
│   └── nn4.small2.v1.t7
├── output/
│   ├── embeddings.pickle
│   ├── le.pickle
│   └── recognizer.pickle
├── entrenar_modelo.py
├── generar_embeddings.py
├── identificar_caras.py
├── requirements.txt
└── .gitignore
```

El directorio `dataset/` no se incluye en el repositorio. Debe crearse localmente para entrenar el sistema con rostros propios.

