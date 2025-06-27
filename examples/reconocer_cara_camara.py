import cv2
import os
import imutils

# Rutas al modelo
prototxt_path = os.path.sep.join(["../face_detector", "deploy.prototxt"])
weights_path = os.path.sep.join(["../face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])

# Cargar modelo
net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

# Iniciar cámara web
video = cv2.VideoCapture(0)  # 0 = cámara principal

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Crear blob y procesar
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Dibujar las detecciones
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            text = f"{confidence:.2f}"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Mostrar resultado
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
