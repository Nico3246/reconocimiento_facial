import os
import cv2
import pickle
import numpy as np
from imutils import paths

# Cargar modelos
detector = cv2.dnn.readNetFromCaffe("face_detector/deploy.prototxt",
                                    "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
embedder = cv2.dnn.readNetFromTorch("openface/nn4.small2.v1.t7")

# Rutas
dataset_path = "dataset"
output_path = "output/embeddings.pickle"

imagePaths = list(paths.list_images(dataset_path))
knownEmbeddings = []
knownNames = []
total = 0

for imagePath in imagePaths:
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[WARNING] No se pudo leer la imagen: {imagePath}")
        continue
    (h, w) = image.shape[:2]

    # Detección de rostro
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    if detections.shape[2] > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]

            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1
            print(f"[INFO] Procesado {name}: {imagePath}")

# Guardar embeddings
data = {"embeddings": knownEmbeddings, "names": knownNames}
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "wb") as f:
    f.write(pickle.dumps(data))

print(f"[INFO] Total de embeddings generados: {total}")
