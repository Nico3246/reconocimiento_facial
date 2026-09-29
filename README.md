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

## Instalación

Se recomienda utilizar un entorno virtual:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

Instala las dependencias:

```bash
pip install -r requirements.txt
pip install scikit-learn
```

> `entrenar_modelo.py` utiliza scikit-learn para crear el clasificador SVM. Actualmente esta dependencia no figura en `requirements.txt`, por lo que debe instalarse adicionalmente.

## Preparar el dataset

Crea una carpeta por cada persona dentro de `dataset/`:

```text
dataset/
├── Persona_1/
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── foto3.jpg
└── Persona_2/
    ├── foto1.jpg
    ├── foto2.jpg
    └── foto3.jpg
```

El nombre de cada carpeta se utilizará como etiqueta de identidad durante el entrenamiento.

Para obtener mejores resultados conviene utilizar varias fotografías de cada persona, con variaciones de iluminación, ángulo y expresión.

## Uso

### 1. Generar embeddings

```bash
python generar_embeddings.py
```

El script:

1. Recorre las imágenes de `dataset/`.
2. Localiza el rostro con el detector Caffe.
3. Genera un embedding con OpenFace.
4. Guarda todos los vectores y etiquetas en:

```text
output/embeddings.pickle
```

### 2. Entrenar el clasificador

```bash
python entrenar_modelo.py
```

Se entrena un SVM lineal y se generan:

```text
output/recognizer.pickle
output/le.pickle
```

### 3. Ejecutar reconocimiento en tiempo real

```bash
python identificar_caras.py
```

La aplicación utiliza la cámara configurada como dispositivo `0`, detecta los rostros de cada fotograma y muestra la identidad estimada junto con la probabilidad del clasificador.

Pulsa **ESC** para cerrar la aplicación.

## Modelos utilizados

El repositorio contiene los modelos necesarios para el pipeline actual:

- **SSD Caffe** para detección de rostros:
  - `face_detector/deploy.prototxt`
  - `face_detector/res10_300x300_ssd_iter_140000.caffemodel`
- **OpenFace** para generar embeddings:
  - `openface/nn4.small2.v1.t7`

## Umbrales actuales

La detección facial solo procesa detecciones con una confianza superior a `0.5`.

Durante la identificación, una predicción con probabilidad inferior a `0.4` se muestra como `Desconocido`.

Estos valores pueden ajustarse en los scripts según las condiciones de uso y el dataset empleado.

## Privacidad

Las fotografías utilizadas para entrenar identidades no forman parte del repositorio. Mantener el dataset fuera de Git evita publicar imágenes personales accidentalmente.

Los embeddings y clasificadores almacenados en `output/` también representan información derivada de datos biométricos y deben tratarse con precaución si el proyecto se utiliza con datos reales.

## Limitaciones

- El rendimiento depende de la calidad y variedad de las imágenes de entrenamiento.
- Cambios fuertes de iluminación, orientación o distancia pueden reducir la precisión.
- El clasificador solo puede identificar correctamente personas representadas en su conjunto de entrenamiento.
- El sistema está planteado como proyecto educativo y experimental, no como solución de identificación biométrica para entornos críticos.
