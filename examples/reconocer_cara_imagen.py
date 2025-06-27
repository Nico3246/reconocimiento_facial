import cv2
import os

# Construir las rutas hacia los archivos del modelo
prototxt_path = os.path.sep.join(["../face_detector", "deploy.prototxt"])
weights_path = os.path.sep.join(["../face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])

# Cargar el modelo desde disco
net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

# Cargar la imagen desde disco
image = cv2.imread("example.jpg")
(h, w) = image.shape[:2]

# Crear un blob a partir de la imagen
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))

# Pasar el blob por la red y obtener las detecciones
net.setInput(blob)
detections = net.forward()

# Recorrer todas las detecciones
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filtrar detecciones débiles
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (startX, startY, endX, endY) = box.astype("int")

        # Dibujar la caja y la confianza
        text = f"{confidence:.2f}"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Mostrar la imagen final con las detecciones
cv2.imshow("Output", image)
cv2.waitKey(0)
