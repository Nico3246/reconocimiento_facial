import cv2
import pickle
import numpy as np

# Cargar modelos
print("[INFO] Cargando modelos...")
detector = cv2.dnn.readNetFromCaffe("face_detector/deploy.prototxt",
                                    "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
embedder = cv2.dnn.readNetFromTorch("openface/nn4.small2.v1.t7")

recognizer = pickle.load(open("output/recognizer.pickle", "rb"))
le = pickle.load(open("output/le.pickle", "rb"))

# Iniciar cámara
print("[INFO] Iniciando cámara...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]

            # Si la probabilidad es menor que 60%, marcar como desconocido
            if proba < 0.4:
                name = "Desconocido"
            else:
                name = le.classes_[j]

            text = f"{name}: {proba * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("Reconocimiento Facial", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
