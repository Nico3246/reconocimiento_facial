import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import os

# Cargar embeddings
with open("output/embeddings.pickle", "rb") as f:
    data = pickle.load(f)

# Codificar las etiquetas (nombres)
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# Entrenar el modelo SVM
print("[INFO] Entrenando el clasificador...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# Crear carpeta si no existe
os.makedirs("output", exist_ok=True)

# Guardar el clasificador y el codificador de etiquetas
with open("output/recognizer.pickle", "wb") as f:
    f.write(pickle.dumps(recognizer))

with open("output/le.pickle", "wb") as f:
    f.write(pickle.dumps(le))

print("[INFO] Modelo guardado en 'output/recognizer.pickle'")
